import { Fragment, type ReactNode } from 'react';

/**
 * Small Markdown renderer for chat replies: headings, lists, bold, italics,
 * inline code, code blocks, links and quotes. It builds React elements (never
 * raw HTML), so model output can't inject markup.
 */
export default function Markdown({ text }: { text: string }) {
  const blocks: ReactNode[] = [];
  const lines = text.replace(/\r\n/g, '\n').split('\n');
  let i = 0;
  let key = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (/^```/.test(line)) {
      const body: string[] = [];
      i++;
      while (i < lines.length && !/^```/.test(lines[i])) body.push(lines[i++]);
      i++;
      blocks.push(
        <pre key={key++} className="my-2 overflow-x-auto rounded-lg border border-border-dim bg-bg-secondary p-3 text-xs leading-relaxed text-text-secondary">
          <code>{body.join('\n')}</code>
        </pre>);
      continue;
    }

    const h = /^(#{1,4})\s+(.*)$/.exec(line);
    if (h) {
      const size = h[1].length <= 2 ? 'text-base' : 'text-sm';
      blocks.push(<p key={key++} className={`mb-1 mt-3 font-semibold text-text-primary first:mt-0 ${size}`}>{inline(h[2])}</p>);
      i++;
      continue;
    }

    if (/^\s*([-*•]|\d+[.)])\s+/.test(line)) {
      const ordered = /^\s*\d+[.)]/.test(line);
      const items: ReactNode[] = [];
      while (i < lines.length && /^\s*([-*•]|\d+[.)])\s+/.test(lines[i])) {
        const indent = /^\s*/.exec(lines[i])![0].length;
        const content = lines[i].replace(/^\s*([-*•]|\d+[.)])\s+/, '');
        items.push(<li key={items.length} className={indent >= 2 ? 'ml-4' : ''}>{inline(content)}</li>);
        i++;
      }
      const List = ordered ? 'ol' : 'ul';
      blocks.push(
        <List key={key++} className={`my-1.5 space-y-1 pl-5 ${ordered ? 'list-decimal' : 'list-disc'} marker:text-text-muted`}>
          {items}
        </List>);
      continue;
    }

    // Tables: | a | b |, then a |---|---| separator row.
    if (/^\s*\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
      const cells = (l: string) => l.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((c) => c.trim());
      const head = cells(line);
      i += 2;
      const rows: string[][] = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) rows.push(cells(lines[i++]));
      blocks.push(
        <div key={key++} className="my-2 overflow-x-auto rounded-lg border border-border-dim">
          <table className="w-full border-collapse text-xs">
            <thead className="bg-bg-secondary">
              <tr>{head.map((h, j) => <th key={j} className="border-b border-border-dim px-2.5 py-1.5 text-left font-semibold text-text-primary">{inline(h)}</th>)}</tr>
            </thead>
            <tbody>
              {rows.map((r, ri) => (
                <tr key={ri} className="align-top odd:bg-bg-secondary/40">
                  {head.map((_, j) => <td key={j} className="border-t border-border-dim/60 px-2.5 py-1.5 text-text-secondary">{inline(r[j] || '')}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>);
      continue;
    }

    if (/^>\s?/.test(line)) {
      const quote: string[] = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) quote.push(lines[i++].replace(/^>\s?/, ''));
      blocks.push(
        <blockquote key={key++} className="my-2 border-l-2 border-accent-purple/50 pl-3 text-text-secondary">
          {inline(quote.join(' '))}
        </blockquote>);
      continue;
    }

    if (/^\s*(---|\*\*\*)\s*$/.test(line)) {
      blocks.push(<hr key={key++} className="my-3 border-border-dim" />);
      i++;
      continue;
    }

    if (!line.trim()) {
      i++;
      continue;
    }

    // Always take the current line (it matched nothing above), so the loop
    // can't stall on e.g. a table row whose separator hasn't streamed in yet.
    const para: string[] = [lines[i++]];
    while (i < lines.length && lines[i].trim() && !/^(```|#{1,4}\s|>\s?|\s*\||\s*([-*•]|\d+[.)])\s+)/.test(lines[i])) {
      para.push(lines[i++]);
    }
    blocks.push(<p key={key++} className="my-1.5 first:mt-0">{inline(para.join('\n'))}</p>);
  }
  return <>{blocks}</>;
}

const TOKEN = /(`[^`]+`|\*\*[^*]+\*\*|(?<!\w)__[^_]+__(?!\w)|\*[^*\s][^*]*\*|(?<!\w)_[^_\s][^_]*_(?!\w)|\[[^\]]+\]\([^)\s]+\)|https?:\/\/[^\s)]+|\n)/g;

function inline(text: string): ReactNode {
  const out: ReactNode[] = [];
  let last = 0;
  let k = 0;
  for (const m of text.matchAll(TOKEN)) {
    if (m.index! > last) out.push(text.slice(last, m.index));
    const t = m[0];
    if (t === '\n') out.push(<br key={k++} />);
    else if (t.startsWith('`')) out.push(<code key={k++} className="rounded bg-bg-tertiary px-1 py-0.5 text-[0.85em] text-accent-cyan">{t.slice(1, -1)}</code>);
    else if (t.startsWith('**') || t.startsWith('__')) out.push(<strong key={k++} className="font-semibold text-text-primary">{inline(t.slice(2, -2))}</strong>);
    else if (t.startsWith('[')) {
      const mm = /^\[([^\]]+)\]\(([^)\s]+)\)$/.exec(t)!;
      out.push(<LinkText key={k++} label={mm[1]} href={mm[2]} />);
    } else if (t.startsWith('http')) out.push(<LinkText key={k++} label={t} href={t} />);
    else out.push(<em key={k++}>{inline(t.slice(1, -1))}</em>);
    last = m.index! + t.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return <Fragment>{out}</Fragment>;
}

/** Links are shown, not followed: the app window shouldn't navigate away. Click copies the address. */
function LinkText({ label, href }: { label: string; href: string }) {
  return (
    <span
      title={`${href} (click to copy)`}
      onClick={() => navigator.clipboard?.writeText(href).catch(() => {})}
      className="cursor-pointer text-accent-blue underline decoration-accent-blue/40 underline-offset-2 hover:decoration-accent-blue"
    >
      {label}
    </span>
  );
}
