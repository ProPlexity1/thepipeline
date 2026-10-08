"""ThePipeline's agent: a local chat model that can research, plan and drive
the app's generators through tools.

Conversations are saved after every step (power cuts lose at most the reply in
progress). Each user message starts a "turn": the model may call tools several
times before giving its answer. Everything the model produces is streamed to
the UI as agent_event messages.
"""
import ipaddress
import json
import re
import socket
import threading
import time
import uuid
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import parse_qs, unquote, urlparse

import requests

import llm
from model_registry import MODELS_DIR

AGENT_DIR = MODELS_DIR.parent / "agent"
CONV_DIR = AGENT_DIR / "conversations"
SETTINGS_PATH = MODELS_DIR.parent / "settings.json"
MAX_TOOL_ROUNDS = 16
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36"

SYSTEM_PROMPT = """You are the director inside "ThePipeline", a desktop app that makes videos, images and voice-overs entirely on the user's own computer with open AI models.

Your job: understand what the user wants to make, then plan it and drive the app's tools to make it.
- Ask a short clarifying question when the request is genuinely ambiguous (audience, length, style); otherwise make sensible choices and say what you chose.
- Before queuing more than one long generation, show the plan (shots, models, estimated time) and wait for the user's go-ahead.
- Write generation prompts in rich, concrete visual language: subject, action, setting, lighting, camera, style. For models that make sound (MiniMax H3), also describe the sound: dialogue in quotes, effects, music.
- Keep characters consistent by repeating the same detailed description in every shot.
- Use web_search and read_webpage when facts, references or current information would improve the result, and mention your sources briefly.
- Videos take minutes each on this PC; be realistic about time. Generation runs while you are asleep, so after queuing jobs, tell the user what will happen and finish your reply.
- Be concise and friendly. Use short paragraphs and lists. Never invent tool results.

How to make a finished video (production):
1. Agree on the idea, audience, length, aspect and visual style (photorealistic, anime, 3D animation...). State the style in every prompt.
2. Design characters once, in precise visual detail (age, face, hair, build, clothing, colours), and reuse the exact wording in every shot.
3. Write the voice-over script first if there is narration (about 2.5 spoken words per second; numbers written as words).
4. Break it into 4-8 shots. Each shot: a keyframe_prompt (one photograph: subject, pose, setting, lighting, lens, framing) and a camera move (people: push_in/pull_out; places: pan, orbit, rise).
5. Show the plan as a numbered shot list with the script and an estimated time, and wait for approval.
6. Call start_production. Default mode "image": sharp 2K stills with cinematic camera moves, a natural voice-over, the logo and an exact end card, done in minutes. Use mode "video" only when people must visibly act (walk, talk, gesture) and the user accepts 15-20 minutes per shot.
7. Then finish your reply. The app posts in this chat by itself when the video is ready.

Getting details right:
- Never put text, phone numbers, addresses or logos in image prompts; generators misspell them. Put exact text in end_card_lines and the logo in logo_image.
- Describe only what should appear. Words like "no holster" make the model draw a holster.
- For the user's logo or product photo: pass its gallery name (attached images are listed in the chat) as logo_image, or as a shot's image to use it directly.

When something fails:
- Read the error and its what_to_do hint, fix the cause (wrong name, missing model, bad argument) and retry once or twice with corrected arguments.
- If a model isn't installed, use list_models to pick another installed one, or tell the user exactly which model to install from the Models page.
- Never claim something worked unless a tool result says so."""


# ── settings ─────────────────────────────────────────────────────────────────

def load_settings() -> dict:
    defaults = {"allow_web_research": True, "agent_model": None, "brave_api_key": ""}
    try:
        return {**defaults, **json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))}
    except (OSError, ValueError):
        return defaults


def save_settings(values: dict) -> dict:
    s = {**load_settings(), **values}
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=2), encoding="utf-8")
    tmp.replace(SETTINGS_PATH)
    return s


# ── web research ─────────────────────────────────────────────────────────────

def _public_http_url(url: str) -> str:
    """Only http(s) URLs that resolve to public addresses: the agent must never
    be steered into this PC's own services or the local network."""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise ValueError("Only http and https web addresses can be opened.")
    try:
        infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80))
    except socket.gaierror:
        raise ValueError(f"Couldn't find the website {p.hostname}.")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ValueError("That address points at a local or private network, which the agent may not open.")
    return url


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "iframe", "aside"}
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skip:
            self.parts.append(data)

    def text(self) -> str:
        t = "".join(self.parts)
        t = re.sub(r"[ \t\r\f\v]+", " ", t)
        return re.sub(r"\n\s*\n+", "\n\n", t).strip()


def _search_brave(query: str, key: str, n: int) -> list[dict]:
    r = requests.get("https://api.search.brave.com/res/v1/web/search", params={"q": query, "count": n},
                     headers={"X-Subscription-Token": key, "Accept": "application/json"}, timeout=20)
    r.raise_for_status()
    return [{"title": x.get("title", ""), "url": x.get("url", ""),
             "snippet": unescape(re.sub(r"<[^>]+>", "", x.get("description", "")))}
            for x in (r.json().get("web") or {}).get("results", [])[:n]]


def _search_duckduckgo(query: str, n: int) -> list[dict]:
    r = requests.post("https://html.duckduckgo.com/html/", data={"q": query},
                      headers={"User-Agent": USER_AGENT}, timeout=20)
    if r.status_code != 200:  # 202 = DuckDuckGo's bot check; we don't try to get around it
        return []
    clean = lambda s: unescape(re.sub(r"<[^>]+>", "", s)).strip()
    results = []
    for m in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=<a[^>]+class="result__a"|$)',
                         r.text, re.S):
        href, title, rest = m.group(1), m.group(2), m.group(3)
        if "uddg=" in href:
            href = unquote(parse_qs(urlparse(href).query).get("uddg", [href])[0])
        if href.startswith("//"):
            href = "https:" + href
        if "duckduckgo.com/y.js" in href:  # ads
            continue
        sn = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', rest, re.S)
        results.append({"title": clean(title), "url": href, "snippet": clean(sn.group(1)) if sn else ""})
        if len(results) >= n:
            break
    return results


def _search_marginalia(query: str, n: int) -> list[dict]:
    """Marginalia's public search API (keyless; independent, non-commercial web index)."""
    r = requests.get(f"https://api.marginalia.nu/public/search/{requests.utils.quote(query)}",
                     params={"count": n}, headers={"User-Agent": "ThePipeline/1.0 (local desktop app)"}, timeout=20)
    r.raise_for_status()
    return [{"title": x.get("title", ""), "url": x.get("url", ""), "snippet": x.get("description", "")}
            for x in r.json().get("results", [])[:n]]


def _search_wikipedia(query: str, n: int) -> list[dict]:
    r = requests.get("https://en.wikipedia.org/w/api.php",
                     params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": n},
                     headers={"User-Agent": "ThePipeline/1.0 (local desktop app)"}, timeout=20)
    r.raise_for_status()
    return [{"title": x["title"], "url": "https://en.wikipedia.org/wiki/" + x["title"].replace(" ", "_"),
             "snippet": unescape(re.sub(r"<[^>]+>", "", x.get("snippet", "")))}
            for x in r.json().get("query", {}).get("search", [])]


_STOP = set("a an and are as at be by for from how i in is it of on or that the this to what when where which who why with "
             "best good tips guide tutorial ways way make making using use do does can should".split())


def _keywords(query: str, n: int = 5) -> str:
    words = [w for w in re.findall(r"[\w'-]+", query.lower()) if w not in _STOP]
    return " ".join(words[:n])


def web_search(query: str, max_results: int = 6) -> dict:
    """Brave (if the user added a key), else DuckDuckGo, else Wikipedia."""
    key = (load_settings().get("brave_api_key") or "").strip()
    attempts = ([("Brave Search", lambda: _search_brave(query, key, max_results))] if key else []) + [
        ("DuckDuckGo", lambda: _search_duckduckgo(query, max_results)),
        ("Marginalia", lambda: _search_marginalia(query, max_results)),
        # Long natural-language queries often match nothing; retry with the key words.
        ("Marginalia", lambda: _search_marginalia(_keywords(query, 4), max_results)),
        ("Wikipedia", lambda: _search_wikipedia(query, max_results)),
        ("Wikipedia", lambda: _search_wikipedia(_keywords(query, 3), max_results)),
    ]
    errors = []
    for name, fn in attempts:
        try:
            results = fn()
        except requests.RequestException as e:
            errors.append(f"{name}: {str(e)[:80]}")
            continue
        if results:
            return {"query": query, "source": name, "results": results}
    return {"query": query, "results": [], "note": "No results. " + "; ".join(errors)}


def read_webpage(url: str, max_chars: int = 12000) -> dict:
    _public_http_url(url)
    with requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20, stream=True, allow_redirects=True) as r:
        _public_http_url(r.url)  # redirects must stay public too
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        if "html" not in ctype and "text" not in ctype:
            return {"url": r.url, "error": f"Not a web page ({ctype or 'unknown type'})."}
        raw = b""
        for chunk in r.iter_content(65536):
            raw += chunk
            if len(raw) > 3_000_000:
                break
        html = raw.decode(r.encoding or "utf-8", errors="replace")
    ex = _TextExtractor()
    ex.feed(html)
    text = ex.text()
    return {"url": url, "title": ex.title.strip(), "text": text[:max_chars],
            "truncated": len(text) > max_chars}


# ── tools ────────────────────────────────────────────────────────────────────

class Tool:
    def __init__(self, name: str, description: str, parameters: dict, fn: Callable[..., dict],
                 needs_web: bool = False):
        self.name, self.description, self.parameters, self.fn, self.needs_web = \
            name, description, parameters, fn, needs_web

    def schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                  "parameters": self.parameters}}


TOOLS: dict[str, Tool] = {}


def register_tool(tool: Tool):
    TOOLS[tool.name] = tool


register_tool(Tool(
    "web_search", "Search the web. Returns titles, links and snippets. Use for facts, references, trends and inspiration.",
    {"type": "object", "properties": {"query": {"type": "string", "description": "What to search for"}},
     "required": ["query"]},
    lambda query: web_search(query), needs_web=True))
register_tool(Tool(
    "read_webpage", "Read the main text of a web page (for example a search result).",
    {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
    lambda url: read_webpage(url), needs_web=True))


def active_tools() -> list[Tool]:
    web = load_settings().get("allow_web_research", True)
    return [t for t in TOOLS.values() if web or not t.needs_web]


# ── conversations ────────────────────────────────────────────────────────────

_conv_locks: dict[str, threading.Lock] = {}


def _conv_path(conv_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{12}", conv_id):
        raise ValueError("Invalid conversation id")
    return CONV_DIR / f"{conv_id}.json"


def save_conversation(conv: dict):
    CONV_DIR.mkdir(parents=True, exist_ok=True)
    path = _conv_path(conv["id"])
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(conv, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_conversation(conv_id: str) -> dict:
    return json.loads(_conv_path(conv_id).read_text(encoding="utf-8"))


def new_conversation(title: str = "") -> dict:
    conv = {"id": uuid.uuid4().hex[:12], "title": title or "New chat",
            "created_at": datetime.now().isoformat(timespec="seconds"), "messages": []}
    save_conversation(conv)
    return conv


def list_conversations() -> list[dict]:
    out = []
    for p in sorted(CONV_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            c = json.loads(p.read_text(encoding="utf-8"))
            out.append({"id": c["id"], "title": c.get("title", ""), "created_at": c.get("created_at"),
                        "updated_at": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
                        "messages": len(c.get("messages", []))})
        except (OSError, ValueError, KeyError):
            continue
    return out


def delete_conversation(conv_id: str):
    _conv_path(conv_id).unlink(missing_ok=True)


def _to_api_messages(conv: dict) -> list[dict]:
    """Stored messages -> OpenAI chat format (images as data URLs)."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT + f"\n\nToday is {datetime.now():%A %d %B %Y}."}]
    for m in conv["messages"]:
        if m["role"] == "user":
            if m.get("images"):
                parts = [{"type": "text", "text": m.get("content", "")}]
                parts += [{"type": "image_url", "image_url": {"url": u}} for u in m["images"]]
                msgs.append({"role": "user", "content": parts})
            else:
                msgs.append({"role": "user", "content": m.get("content", "")})
        elif m["role"] == "assistant":
            entry = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("tool_calls"):
                entry["tool_calls"] = [{"id": c["id"], "type": "function",
                                        "function": {"name": c["name"], "arguments": c["arguments"]}}
                                       for c in m["tool_calls"]]
            msgs.append(entry)
        elif m["role"] == "tool":
            msgs.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
    return msgs


# ── turns ────────────────────────────────────────────────────────────────────

class Turn:
    """One user message and everything the agent does in reply."""
    active: dict[str, "Turn"] = {}
    context = threading.local()  # .conv_id while a tool runs, so tools can report back to this chat

    def __init__(self, conv_id: str, model_id: str, emit: Callable[[dict], None]):
        self.conv_id, self.model_id, self.emit = conv_id, model_id, emit
        self.cancel = threading.Event()

    def _event(self, event: str, **data):
        self.emit({"type": "agent_event", "conv_id": self.conv_id, "event": event, **data})

    def run(self, text: str, images: list[str]):
        lock = _conv_locks.setdefault(self.conv_id, threading.Lock())
        if not lock.acquire(blocking=False):
            self._event("error", error="The agent is still answering the previous message.")
            return
        Turn.active[self.conv_id] = self
        try:
            conv = load_conversation(self.conv_id)
            conv["messages"].append({"role": "user", "content": text, "images": images,
                                     "at": datetime.now().isoformat(timespec="seconds")})
            if conv.get("title") in (None, "", "New chat"):
                conv["title"] = (text.strip().splitlines() or ["New chat"])[0][:60] or "Image"
            save_conversation(conv)
            self._respond(conv)
        except Exception as e:
            self._event("error", error=str(e)[:500])
        finally:
            Turn.active.pop(self.conv_id, None)
            lock.release()

    def _wait_for_gpu(self):
        """The GPU is busy with a video/image/voice job and the agent is asleep: wait (visibly)
        and continue by ourselves. If the agent is already awake mid-reply, the GPU waits for us."""
        announced = False
        while llm.server.gpu_busy() and not llm.server.running() and not self.cancel.is_set():
            if not announced:
                self._event("status", status="waiting",
                            message="Waiting for the graphics card to finish the current job; I'll answer automatically.")
                announced = True
            time.sleep(3)

    def _respond(self, conv: dict):
        tools = active_tools()
        rounds = 0
        crashes = 0
        counted = False  # counted as a reply in progress (the GPU scheduler waits for us)
        try:
            self._loop(conv, tools)
        finally:
            if self._counted:
                llm.server.end_turn()
        self._event("done")

    _counted = False

    def _hold(self, on: bool):
        if on and not self._counted:
            llm.server.begin_turn()
            self._counted = True
        elif not on and self._counted:
            llm.server.end_turn()
            self._counted = False

    def _loop(self, conv: dict, tools: list):
        rounds = 0
        crashes = 0
        while rounds < MAX_TOOL_ROUNDS and not self.cancel.is_set():
            self._wait_for_gpu()
            if self.cancel.is_set():
                break
            if not llm.server.running() or llm.server.model_id != self.model_id:
                self._event("status", status="waking", message="Waking up the agent…")
            self._event("assistant_start")
            self._hold(True)
            try:
                with llm.server.turn_lock:
                    msg = llm.server.chat_stream(
                        self.model_id, _to_api_messages(conv), [t.schema() for t in tools] or None,
                        on_delta=lambda kind, piece: self._event("delta", kind=kind, text=piece),
                        cancelled=self.cancel.is_set)
            except llm.AgentBusy:
                self._hold(False)
                continue  # a GPU job started just now; wait for it, then carry on
            except Exception as e:
                # The chat engine crashed or hung: restart it and try once more.
                self._hold(False)
                crashes += 1
                llm.server.stop()
                if crashes > 1:
                    raise RuntimeError(f"The agent's model stopped responding ({str(e)[:200]}). Try again in a moment.")
                self._event("status", status="waking", message="Restarting the agent after a hiccup…")
                continue
            rounds += 1
            self._finish_round(conv, msg, tools)
            if not msg["tool_calls"] or self.cancel.is_set():
                break

    def _finish_round(self, conv: dict, msg: dict, tools: list):
        if True:
            entry = {"role": "assistant", "content": msg["content"], "reasoning": msg["reasoning"],
                     "at": datetime.now().isoformat(timespec="seconds")}
            if msg["tool_calls"]:
                for c in msg["tool_calls"]:
                    c["id"] = c["id"] or f"call_{uuid.uuid4().hex[:8]}"
                entry["tool_calls"] = msg["tool_calls"]
            conv["messages"].append(entry)
            save_conversation(conv)
            if not msg["tool_calls"] or self.cancel.is_set():
                return
            for call in msg["tool_calls"]:
                result = self._call_tool(call, {t.name: t for t in tools})
                conv["messages"].append({"role": "tool", "tool_call_id": call["id"], "name": call["name"],
                                         "content": json.dumps(result, ensure_ascii=False)[:20000]})
                save_conversation(conv)

    def _call_tool(self, call: dict, tools: dict[str, Tool]) -> dict:
        name = call["name"]
        try:
            args = json.loads(call["arguments"] or "{}")
        except ValueError:
            args = None
        self._event("tool_call", id=call["id"], name=name, arguments=args or {})
        t0 = time.time()
        if args is None:
            result = {"error": "The tool arguments were not valid JSON."}
        elif name not in tools:
            result = {"error": f"There is no tool called {name}. Available: {', '.join(sorted(tools))}."}
        else:
            Turn.context.conv_id = self.conv_id
            try:
                result = tools[name].fn(**args)
            except TypeError as e:
                result = {"error": f"Wrong arguments for {name}: {e}"}
            except Exception as e:
                result = {"error": str(e)[:400]}
            finally:
                Turn.context.conv_id = None
        if isinstance(result, dict) and result.get("error"):
            # Push the model to recover instead of giving up or inventing results.
            result["what_to_do"] = ("Work out why this failed from the error. Fix the arguments and call the tool again, "
                                    "or use list_models / list_jobs to check what is available. If it truly can't be "
                                    "fixed, tell the user plainly what went wrong and what they can do.")
        self._event("tool_result", id=call["id"], name=name, result=result, seconds=round(time.time() - t0, 1))
        return result


def post_notice(conv_id: str, text: str, emit: Callable[[dict], None]):
    """Add a message from the agent to a chat without a model call (e.g. 'your video is ready')."""
    try:
        conv = load_conversation(conv_id)
    except (ValueError, OSError):
        return
    conv["messages"].append({"role": "assistant", "content": text, "notice": True,
                             "at": datetime.now().isoformat(timespec="seconds")})
    save_conversation(conv)
    emit({"type": "agent_event", "conv_id": conv_id, "event": "notice", "text": text})


def start_turn(conv_id: str, model_id: str, text: str, images: list[str], emit: Callable[[dict], None]):
    load_conversation(conv_id)  # validates the id / existence before going async
    turn = Turn(conv_id, model_id, emit)
    threading.Thread(target=turn.run, args=(text, images), daemon=True).start()


def cancel_turn(conv_id: str):
    t = Turn.active.get(conv_id)
    if t:
        t.cancel.set()
