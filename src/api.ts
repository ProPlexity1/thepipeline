// Single place that knows how to reach the local AI engine. The desktop shell
// picks a free port and a per-launch secret token; every request carries it.

let base = '';
let token = '';

export function configureApi(port: number, apiToken: string) {
  base = `http://127.0.0.1:${port}`;
  token = apiToken;
}

export function apiReady() {
  return base !== '';
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export async function api<T = any>(path: string, init: RequestInit = {}): Promise<T> {
  if (!base) throw new ApiError(0, 'AI engine not started');
  const headers = new Headers(init.headers);
  headers.set('x-neuralcut-token', token);
  if (init.body && !headers.has('content-type')) headers.set('content-type', 'application/json');
  const res = await fetch(base + path, { ...init, headers });
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) {
    const detail = data?.detail;
    throw new ApiError(res.status, typeof detail === 'string' ? detail : `Request failed (${res.status})`);
  }
  return data as T;
}

export const post = <T = any>(path: string, body?: unknown) =>
  api<T>(path, { method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) });

export const del = <T = any>(path: string) => api<T>(path, { method: 'DELETE' });

export function wsUrl() {
  return `${base.replace('http', 'ws')}/ws?token=${encodeURIComponent(token)}`;
}

export async function healthy(): Promise<any | null> {
  if (!base) return null;
  try {
    const res = await fetch(`${base}/health`);
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}

export const formatBytes = (b: number) =>
  b >= 1024 ** 3 ? `${(b / 1024 ** 3).toFixed(1)} GB` : b >= 1024 ** 2 ? `${(b / 1024 ** 2).toFixed(0)} MB` : `${Math.max(0, Math.round(b / 1024))} KB`;

export const formatDuration = (s: number) => {
  if (!isFinite(s) || s <= 0) return '';
  if (s < 60) return `${Math.round(s)}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${Math.round(s % 60)}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
};

/** URL for a file in the audio library. */
export function audioUrl(name: string) {
  return `${base}/audio/${encodeURIComponent(name)}/file?token=${encodeURIComponent(token)}`;
}

/** URL for an image in the image gallery. */
export function imageUrl(name: string) {
  return `${base}/images/${encodeURIComponent(name)}/file?token=${encodeURIComponent(token)}`;
}

/** URL the <video> element can stream a gallery file from (token-checked, range-capable). */
export function videoUrl(name: string) {
  return `${base}/outputs/${encodeURIComponent(name)}/file?token=${encodeURIComponent(token)}`;
}
