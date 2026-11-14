#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OXH UI — minimal web interface for OX-Neuro Hyper (adaptive rate always on).

Run:    python3 ox_ui.py [--port 8000]
Open:   http://127.0.0.1:8000

Two actions: compress (images -> .oxh) and decompress (.oxh -> .jpg).
Accepts files and whole folders via drag&drop; unsupported formats are
skipped silently. GPU is used automatically; on VRAM exhaustion the codec
falls back to CPU (inside ox_neuro.py).
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote

HERE = os.path.dirname(os.path.abspath(__file__))
OX_NEURO = os.path.join(HERE, 'ox_neuro.py')
CKPT = os.path.join(HERE, 'ox_neuro.pt')
MAX_UPLOAD = 4 * 1024 * 1024 * 1024  # 4 ГБ

IMG_EXT = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tif', '.tiff')

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OXH — OX-Neuro Hyper</title>
<style>
  :root { --bg:#0d1117; --card:#161b22; --border:#30363d; --fg:#e6edf3;
          --dim:#8b949e; --acc:#4c8dff; --ok:#3fb950; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
  .wrap { max-width:720px; margin:0 auto; padding:24px 16px 64px; }
  h1 { font-size:26px; margin:8px 0 2px; }
  h1 .x { color:var(--acc); }
  .sub { color:var(--dim); margin-bottom:20px; }
  .badge { display:inline-block; padding:2px 10px; border-radius:12px;
           font-size:12px; margin-left:8px; vertical-align:middle; }
  .badge.gpu { background:#12341c; color:var(--ok); border:1px solid #238636; }
  .badge.cpu { background:#3a2f0b; color:#d29922; border:1px solid #9e6a03; }
  .card { background:var(--card); border:1px solid var(--border);
          border-radius:12px; padding:18px; margin-bottom:16px; }
  #drop { border:2px dashed var(--border); border-radius:12px; padding:36px 16px;
          text-align:center; color:var(--dim); cursor:pointer;
          transition:border-color .15s, background .15s; }
  #drop.over { border-color:var(--acc); background:#101a2a; }
  #drop b { color:var(--fg); }
  .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
  input[type=number] { background:#0d1117; color:var(--fg);
    border:1px solid var(--border); border-radius:8px; padding:8px 10px;
    width:76px; }
  button { background:var(--acc); color:#fff; border:0; border-radius:8px;
           padding:9px 20px; font-size:15px; cursor:pointer; }
  button.ghost { background:#21262d; border:1px solid var(--border); }
  button:disabled { opacity:.45; cursor:default; }
  #list { margin-top:12px; }
  .file-row { display:flex; justify-content:space-between; gap:10px;
    padding:6px 10px; border-bottom:1px solid var(--border); font-size:14px; }
  .file-row .nm { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .file-row .st { color:var(--dim); white-space:nowrap; }
  .file-row .st.ok { color:var(--ok); }
  .file-row .st.err { color:#f85149; }
  #log { font:12px/1.5 ui-monospace,Menlo,Consolas,monospace; color:var(--dim);
         white-space:pre-wrap; max-height:160px; overflow:auto; margin-top:8px; }
  #prog { height:4px; background:#21262d; border-radius:2px; overflow:hidden;
          margin-top:10px; }
  #prog i { display:block; height:100%; width:40%; background:var(--acc);
            border-radius:2px; animation:slide 1s infinite linear; }
  @keyframes slide { from { margin-left:-40%; } to { margin-left:100%; } }
  .hidden { display:none; }
</style>
</head>
<body>
<div class="wrap">
  <h1>OX<span class="x">·</span>Neuro Hyper <span id="hw" class="badge cpu">…</span></h1>
  <div class="sub">neural image compression · per-patch adaptive rate · hyperprior</div>

  <div class="card">
    <div id="drop">
      <b>Drag &amp; drop files or a folder here</b><br>or click to browse
      <input type="file" id="file" multiple class="hidden">
    </div>
    <div id="prog" class="hidden"><i></i></div>
    <div id="list"></div>
  </div>

  <div class="card">
    <div class="row">
      <span style="color:var(--dim)">quality</span>
      <input type="number" id="q" value="100" min="1" max="100">
      <button id="bc">Compress</button>
      <button id="bd" class="ghost">Decompress</button>
    </div>
    <div id="log"></div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
let files = [];       // File[]
let saveDir = null;   // FileSystemDirectoryHandle или null (обычное скачивание)

fetch('/api/info').then(r => r.json()).then(i => {
  const b = $('hw');
  if (i.cuda) { b.textContent = 'GPU: ' + i.gpu; b.className = 'badge gpu'; }
  else { b.textContent = 'CPU'; }
}).catch(() => {});

// ---------- приём файлов и папок ----------
const drop = $('drop');
drop.onclick = () => $('file').click();
drop.ondragover = e => { e.preventDefault(); drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = async e => {
  e.preventDefault(); drop.classList.remove('over');
  const out = [];
  const items = e.dataTransfer.items;
  const entries = [];
  for (const it of items) {
    const en = it.webkitGetAsEntry && it.webkitGetAsEntry();
    if (en) entries.push(en);
  }
  async function walk(en) {
    if (en.isFile) {
      await new Promise(res => en.file(f => { out.push(f); res(); }));
    } else if (en.isDirectory) {
      const rd = en.createReader();
      let batch;
      do {
        batch = await new Promise(res => rd.readEntries(res));
        for (const sub of batch) await walk(sub);
      } while (batch.length);
    }
  }
  for (const en of entries) await walk(en);
  if (!out.length && e.dataTransfer.files.length)
    out.push(...e.dataTransfer.files);
  addFiles(out);
};
$('file').onchange = e => addFiles([...e.target.files]);

function addFiles(fs) {
  files = files.concat(fs);
  renderList();
  busy(false);
  log('added ' + fs.length + ' file(s), total ' + files.length);
}
function renderList() {
  $('list').innerHTML = files.map(f =>
    '<div class="file-row"><span class="nm">' + esc(f.name) +
    '</span><span class="st" data-name="' + esc(f.name) + '">' + fmt(f.size) +
    '</span></div>').join('');
}
function mark(name, text, cls) {
  const el = document.querySelector('.st[data-name="' + CSS.escape(name) + '"]');
  if (el) { el.textContent = text; el.className = 'st ' + cls; }
}
function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
function fmt(n) {
  if (n < 1024) return n + ' Б';
  if (n < 1048576) return (n / 1024).toFixed(1) + ' КБ';
  if (n < 1073741824) return (n / 1048576).toFixed(2) + ' МБ';
  return (n / 1073741824).toFixed(2) + ' ГБ';
}
function log(msg) {
  const t = new Date().toLocaleTimeString();
  $('log').innerHTML += '[' + t + '] ' + esc(msg) + '\n';
  $('log').scrollTop = $('log').scrollHeight;
}
function busy(on) {
  $('prog').classList.toggle('hidden', !on);
  $('bc').disabled = $('bd').disabled = on || !files.length;
}

// ---------- куда сохранять ----------
async function ensureSaveDir() {
  if (saveDir || !window.showDirectoryPicker) return;   // fallback: скачивание
  try { saveDir = await showDirectoryPicker({ mode: 'readwrite' }); }
  catch (e) { saveDir = null; throw e; }   // пользователь отменил — прервать очередь
}
async function saveFile(name, blob) {
  if (saveDir) {
    const fh = await saveDir.getFileHandle(name, { create: true });
    const w = await fh.createWritable();
    await w.write(blob); await w.close();
  } else {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = name; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  }
}

// ---------- очередь ----------
async function run(op) {
  if (!files.length) { log('drop files or a folder first'); return; }
  busy(true);
  let done = 0, skipped = 0;
  try {
    for (const f of files) {
      const fd = new FormData();
      fd.append('file', f);
      fd.append('q', $('q').value);
      try {
        const r = await fetch('/api/' + op, { method: 'POST', body: fd });
        if (r.status === 415) {           // неподдерживаемый формат — молча
          mark(f.name, 'skipped', ''); skipped++; continue;
        }
        if (!r.ok) { mark(f.name, 'error', 'err'); log(f.name + ': ' + await r.text()); continue; }
        if (done === 0) await ensureSaveDir();    // после первого — спросить куда
        const st = JSON.parse(r.headers.get('X-OX-Stats') || '{}');
        const blob = await r.blob();
        await saveFile(st.out_name || f.name, blob);
        mark(f.name, op === 'compress'
          ? fmt(st.in) + ' → ' + fmt(st.out) + ' (' + st.ratio + '%)'
          : 'done', 'ok');
        done++;
      } catch (e) {
        if (!saveDir && window.showDirectoryPicker && done === 0) {
          log('folder selection cancelled — queue stopped'); break;
        }
        mark(f.name, 'error', 'err'); log(f.name + ': ' + e);
      }
    }
    log((op === 'compress' ? 'compressed' : 'decompressed') + ': ' + done +
        ', skipped: ' + skipped);
    if (done) files = [], renderList();
  } finally { busy(false); }
}
$('bc').onclick = () => run('compress');
$('bd').onclick = () => run('decompress');
busy(false);
</script>
</body>
</html>
"""


# ------------------------------------------------------------- логика ------
def parse_form(handler):
    length = int(handler.headers.get('Content-Length', 0))
    if length <= 0 or length > MAX_UPLOAD:
        raise RuntimeError('empty or too large request body')
    body = handler.rfile.read(length)
    ct = handler.headers.get('Content-Type', '')
    msg = BytesParser(policy=email_policy).parsebytes(
        f'Content-Type: {ct}\r\nMIME-Version: 1.0\r\n\r\n'.encode() + body)
    fields, files = {}, {}
    for part in msg.iter_parts():
        name = part.get_param('name', header='content-disposition')
        fname = part.get_filename()
        data = part.get_payload(decode=True)
        if fname:
            files[name] = (os.path.basename(fname), data)
        else:
            fields[name] = data.decode('utf-8', 'replace')
    return fields, files


class Skip(Exception):
    """Unsupported format — skipped silently (HTTP 415)."""


def do_compress(fields, files):
    name, data = files['file']
    if not name.lower().endswith(IMG_EXT):
        raise Skip(name)
    if not os.path.exists(CKPT):
        raise RuntimeError('no ox_neuro.pt model — see README.md')
    q = fields.get('q', '100')
    with tempfile.TemporaryDirectory(prefix='oxh_') as td:
        src = os.path.join(td, name)
        dst = src + '.oxh'
        with open(src, 'wb') as f:
            f.write(data)
        t0 = time.time()
        r = subprocess.run([sys.executable, OX_NEURO, 'c', src, dst,
                            '--q', q, '--adaptive'],
                           capture_output=True, timeout=86400)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode('utf-8', 'replace')[-400:])
        out = open(dst, 'rb').read()
        dt = time.time() - t0
    n, on = len(data), len(out)
    stats = {
        'in': n, 'out': on,
        'ratio': round(100.0 * on / max(n, 1), 1),
        'time': round(dt, 1),
        'out_name': name + '.oxh',
    }
    return out, stats


def do_decompress(fields, files):
    name, data = files['file']
    # только последний формат: OXN3/OXN4 (Hyper); остальное — молча пропуск
    if not name.lower().endswith('.oxh') or data[:4] not in (b'OXN3', b'OXN4'):
        raise Skip(name)
    base = name[:-4]
    with tempfile.TemporaryDirectory(prefix='oxh_') as td:
        src = os.path.join(td, name)
        dst = os.path.join(td, base + '.jpg')
        with open(src, 'wb') as f:
            f.write(data)
        t0 = time.time()
        r = subprocess.run([sys.executable, OX_NEURO, 'd', src, dst],
                           capture_output=True, timeout=86400)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode('utf-8', 'replace')[-400:])
        out = open(dst, 'rb').read()
        dt = time.time() - t0
    stats = {'in': len(data), 'out': len(out), 'time': round(dt, 1),
             'out_name': base + '.jpg'}
    return out, stats


# ------------------------------------------------------------- HTTP --------
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body=b'', ctype='text/plain; charset=utf-8',
              extra=None):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):
        if self.path == '/':
            self._send(200, PAGE.encode(), 'text/html; charset=utf-8')
        elif self.path == '/api/info':
            cuda, gpu = False, ''
            try:
                import torch
                cuda = torch.cuda.is_available()
                if cuda:
                    gpu = torch.cuda.get_device_name(0)
            except Exception:
                pass
            self._send(200, json.dumps({'cuda': cuda, 'gpu': gpu}).encode(),
                       'application/json')
        else:
            self._send(404, b'not found')

    def do_POST(self):
        try:
            fields, files = parse_form(self)
            if 'file' not in files:
                raise RuntimeError('no file in request')
            if self.path == '/api/compress':
                out, stats = do_compress(fields, files)
            elif self.path == '/api/decompress':
                out, stats = do_decompress(fields, files)
            else:
                self._send(404, b'not found')
                return
            self._send(200, out, 'application/octet-stream',
                       {'X-OX-Stats': json.dumps(stats),
                        'Content-Disposition':
                            f"attachment; filename*=UTF-8''{quote(stats['out_name'])}"})
        except Skip:
            self._send(415, b'unsupported')
        except Exception as e:
            import re
            msg = re.sub(r'[\x00-\x08\x0b-\x1f]', '', str(e))
            self._send(500, msg.encode('utf-8', 'replace'))

    def log_message(self, fmt, *args):
        sys.stderr.write('oxh-ui: ' + fmt % args + '\n')


def main():
    port = 8000
    host = '127.0.0.1'
    if '--port' in sys.argv:
        port = int(sys.argv[sys.argv.index('--port') + 1])
    if '--host' in sys.argv:
        host = sys.argv[sys.argv.index('--host') + 1]
    if not os.path.exists(CKPT):
        print('warning: no ox_neuro.pt — see README.md', file=sys.stderr)
    print(f'OXH UI: http://{host}:{port}')
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == '__main__':
    main()
