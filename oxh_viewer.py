#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OXH Viewer — viewer for .oxh files (OX-Neuro Hyper).

Run:     python3 oxh_viewer.py file.oxh
Opens a local page: wheel/button zoom, drag to pan, arrow keys / buttons
to move between .oxh files in the same folder. Unsupported files are
skipped silently.

For "double-click on .oxh" association see oxh-viewer.desktop and README.md.
"""

import os
import subprocess
import sys
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, quote

HERE = os.path.dirname(os.path.abspath(__file__))
OX_NEURO = os.path.join(HERE, 'ox_neuro.py')

ROOT = os.getcwd()          # ограничимся папкой первого файла
_cache = {}                 # path -> (mtime, size, bytes jpg)

PAGE = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>OXH Viewer</title>
<style>
  html, body { margin:0; height:100%; background:#0d1117; overflow:hidden; }
  #stage { width:100%; height:100%; position:relative; cursor:grab; }
  #stage:active { cursor:grabbing; }
  img { user-select:none; -webkit-user-drag:none; transform-origin:0 0;
        position:absolute; left:0; top:0; }
  #bar { position:fixed; top:10px; left:50%; transform:translateX(-50%);
         background:#161b22cc; border:1px solid #30363d; border-radius:10px;
         padding:6px 12px; color:#e6edf3; font:14px system-ui; z-index:9;
         display:flex; gap:10px; align-items:center; }
  #bar button { background:#21262d; color:#e6edf3; border:1px solid #30363d;
                border-radius:6px; padding:3px 12px; cursor:pointer;
                font-size:15px; }
  #bar button:hover { background:#2d333d; }
  #name { color:#8b949e; max-width:40vw; overflow:hidden;
          text-overflow:ellipsis; white-space:nowrap; }
  #status { position:fixed; inset:0; display:flex; flex-direction:column;
            gap:14px; align-items:center; justify-content:center;
            color:#8b949e; font:15px system-ui; background:#0d1117cc;
            z-index:10; }
  #status .spin { width:36px; height:36px; border:3px solid #30363d;
            border-top-color:#4c8dff; border-radius:50%;
            animation: sp 0.8s linear infinite; }
  @keyframes sp { to { transform: rotate(360deg); } }
  .hidden { display:none !important; }
</style>
</head>
<body>
<div id="bar">
  <button id="prev" title="←">&#8592;</button>
  <span id="name">…</span>
  <button id="next" title="→">&#8594;</button>
  <button id="zm">&#8722;</button>
  <span id="zz">100%</span>
  <button id="zp">+</button>
  <button id="fit">fit</button>
</div>
<div id="stage"><img id="im" src=""></div>
<div id="status" class="hidden"><div class="spin"></div>
  <div>decoding&hellip; <span id="el">0</span> s</div></div>
<script>
let cur = new URLSearchParams(location.search).get('f') || '';
let scale = 1, tx = 0, ty = 0;
let showT = null, tickT = null, t0 = 0;
function statusOn() {
  showT = setTimeout(() => {
    $('status').classList.remove('hidden');
    t0 = Date.now();
    tickT = setInterval(() => $('el').textContent = ((Date.now() - t0) / 1000).toFixed(0), 500);
  }, 300);
}
function statusOff() {
  clearTimeout(showT); clearInterval(tickT);
  $('status').classList.add('hidden');
}
const im = $('im'), stage = $('stage');
function $(id){ return document.getElementById(id); }
function apply() {
  im.style.transform = 'translate(' + tx + 'px,' + ty + 'px) scale(' + scale + ')';
  $('zz').textContent = Math.round(scale * 100) + '%';
}
function fit() {
  const r = stage.getBoundingClientRect();
  scale = Math.min(r.width / im.naturalWidth, r.height / im.naturalHeight, 1);
  tx = (r.width - im.naturalWidth * scale) / 2;
  ty = (r.height - im.naturalHeight * scale) / 2;
  apply();
}
function load(f) {
  cur = f;
  statusOn();
  im.onload = () => { statusOff(); if (im.naturalWidth) fit(); };
  im.onerror = () => { statusOff(); $('name').textContent = f.split('/').pop() + ' — failed'; };
  im.src = '/img?f=' + encodeURIComponent(f) + '&t=' + Date.now();
  $('name').textContent = f.split('/').pop();
  history.replaceState(null, '', '/?f=' + encodeURIComponent(f));
}
async function step(d) {
  const r = await fetch('/api/nav?f=' + encodeURIComponent(cur) + '&d=' + d);
  const j = await r.json();
  if (j.f) load(j.f);
}
stage.onwheel = e => {
  e.preventDefault();
  const k = e.deltaY < 0 ? 1.15 : 1 / 1.15;
  const r = stage.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  tx = mx - (mx - tx) * k; ty = my - (my - ty) * k;
  scale = Math.min(40, Math.max(0.02, scale * k));
  apply();
};
let drag = null;
stage.onmousedown = e => { drag = { x: e.clientX - tx, y: e.clientY - ty }; };
window.onmousemove = e => { if (drag) { tx = e.clientX - drag.x; ty = e.clientY - drag.y; apply(); } };
window.onmouseup = () => drag = null;
$('prev').onclick = () => step(-1);
$('next').onclick = () => step(1);
$('zm').onclick = () => { scale = Math.max(0.02, scale / 1.25); apply(); };
$('zp').onclick = () => { scale = Math.min(40, scale * 1.25); apply(); };
$('fit').onclick = fit;
window.onkeydown = e => {
  if (e.key === 'ArrowLeft') step(-1);
  if (e.key === 'ArrowRight') step(1);
  if (e.key === '+' || e.key === '=') $('zp').click();
  if (e.key === '-') $('zm').click();
  if (e.key === '0') fit();
};
if (cur) load(cur);
</script>
</body>
</html>
"""


def decode_oxh(path):
    """Decode .oxh to JPEG bytes (cached by mtime/size)."""
    st = os.stat(path)
    key = (st.st_mtime, st.st_size)
    if path in _cache and _cache[path][0] == key:
        return _cache[path][1]
    with tempfile.TemporaryDirectory(prefix='oxhv_') as td:
        dst = os.path.join(td, 'out.jpg')
        r = subprocess.run([sys.executable, OX_NEURO, 'd', path, dst],
                           capture_output=True, timeout=86400)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode('utf-8', 'replace')[-300:])
        data = open(dst, 'rb').read()
    _cache.clear()
    _cache[path] = (key, data)
    return data


def list_oxh(folder):
    """Sorted .oxh files of the folder; everything else is skipped silently."""
    try:
        names = sorted(f for f in os.listdir(folder)
                       if f.lower().endswith('.oxh'))
    except OSError:
        names = []
    out = []
    for f in names:
        p = os.path.join(folder, f)
        try:
            with open(p, 'rb') as fh:
                if fh.read(4) in (b'OXN1', b'OXN2', b'OXN3', b'OXN4',
                                  b'OXN5', b'OXN6', b'OXN7', b'OXN8'):
                    out.append(p)
        except OSError:
            pass
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body=b'', ctype='text/html; charset=utf-8'):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):
        q = parse_qs(urlparse(self.path).query)
        path = urlparse(self.path).path
        if path == '/':
            self._send(200, PAGE.encode())
        elif path == '/img':
            f = q.get('f', [''])[0]
            full = os.path.normpath(os.path.join(ROOT, f.lstrip('/')))
            if not (full.startswith(ROOT) and full.lower().endswith('.oxh')):
                self._send(403, b'forbidden', 'text/plain')
                return
            try:
                self._send(200, decode_oxh(full), 'image/jpeg')
            except Exception as e:
                self._send(500, str(e).encode('utf-8', 'replace'), 'text/plain')
        elif path == '/api/nav':
            f = q.get('f', [''])[0]
            d = int(q.get('d', ['1'])[0])
            full = os.path.normpath(os.path.join(ROOT, f.lstrip('/')))
            folder = os.path.dirname(full)
            files = list_oxh(folder)
            nxt = ''
            if full in files and len(files) > 1:
                i = files.index(full)
                nxt = files[(i + d) % len(files)]
            rel = os.path.relpath(nxt, ROOT) if nxt else ''
            self._send(200, ('{"f": "%s"}' % rel.replace('\\', '/')).encode(),
                       'application/json')
        else:
            self._send(404, b'not found', 'text/plain')

    def log_message(self, fmt, *args):
        pass


def main():
    global ROOT
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    first = os.path.abspath(sys.argv[1])
    if not first.lower().endswith('.oxh'):
        print('not an .oxh file:', first)
        return 1
    ROOT = os.path.dirname(first)
    rel = os.path.basename(first)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    port = server.server_address[1]
    url = f'http://127.0.0.1:{port}/?f={quote(rel)}'
    print('OXH Viewer:', url)
    threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    server.serve_forever()


if __name__ == '__main__':
    sys.exit(main() or 0)
