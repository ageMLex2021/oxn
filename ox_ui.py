#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OXH UI — minimal web interface for OX-Neuro Hyper.

Run:    python3 ox_ui.py [--port 8000]
Open:   http://127.0.0.1:8000

Two actions: compress (images -> .oxh) and decompress (.oxh -> .jpg).
Accepts files and whole folders via drag&drop; unsupported formats are
skipped silently. GPU is used automatically; on VRAM exhaustion the codec
falls back to CPU (inside ox_neuro.py).
"""

import functools
import gc
import io
import json
import os
import sys
import threading
import time
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote

HERE = os.path.dirname(os.path.abspath(__file__))
OX_NEURO = os.path.join(HERE, 'ox_neuro.py')
CKPT = os.path.join(HERE, 'ox_neuro.pt')
CTX7 = os.path.join(HERE, 'ox_ctx7.pt')
CANON8 = os.path.join(HERE, 'oxn8_canon.npz')
CANON14 = os.path.join(HERE, 'oxn14_canon.npz')
CANON15 = os.path.join(HERE, 'oxn15_canon.npz')
REC_CKPT = os.path.join(HERE, 'ox_neuro_rec.pt')
PICO_CKPT = os.path.join(HERE, 'oxn_pico.pt')
PICO2_CKPT = os.path.join(HERE, 'oxn_pico_v4_soup50.pt')
PICO5_CKPT = os.path.join(HERE, 'oxn_pico_v5_sparse_frozen_7k.pt')
PICO_MOE_CKPT = os.path.join(HERE, 'oxn_pico_v8_moe6_final.pt')
OXN13_CTX = os.path.join(HERE, 'ox_ctx13.pt')
TINY_CKPT = os.path.join(HERE, 'ox_neuro_tiny.pt')
TINY_CTX7 = os.path.join(HERE, 'ox_ctx9_tiny.pt')
TINY2_CKPT = os.path.join(HERE, 'ox_tiny_v4.pt')
TINY2_CTX7 = os.path.join(HERE, 'ox_tiny_v4_ctx.pt')
TINY2_BOOST = os.path.join(HERE, 'ox_tiny_v4_boost.pt')
MAX_UPLOAD = 4 * 1024 * 1024 * 1024  # 4 ГБ
_CODEC = None
_ACTIVE_MODEL = None
_RESIDENT = False
_RESIDENT_STATES = {}
_CODEC_LOCK = threading.Lock()

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
  input[type=number], select { background:#0d1117; color:var(--fg);
    border:1px solid var(--border); border-radius:8px; padding:8px 10px; }
  input[type=number] { width:76px; }
  button { background:var(--acc); color:#fff; border:0; border-radius:8px;
           padding:9px 20px; font-size:15px; cursor:pointer; }
  button.ghost { background:#21262d; border:1px solid var(--border); }
  button:disabled { opacity:.45; cursor:default; }
  .rm { background:transparent; color:var(--dim); border:0; padding:0 4px;
        font-size:16px; line-height:1; cursor:pointer; }
  .rm:hover { color:#f85149; }
  #list { margin-top:12px; max-height:320px; overflow:auto; }
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
  <div class="sub">neural image compression · deterministic integer OXN8 · hyperprior</div>

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
      <button id="sd" class="ghost">choose save dir</button>
      <span id="sdpath" style="color:var(--dim)">not set — files go to Downloads</span>
    </div>
  </div>

  <div class="card">
    <div class="row">
      <label style="color:var(--dim)">model
        <select id="md">
          <option value="oxn8">OXN8 (stable)</option>
          <option value="oxn17fast">OXN17 fast · single encode</option>
          <option value="oxn17">OXN17 best-of · smallest file, 2× slower</option>
          <option value="pico">PICO reimplementation · research</option>
          <option value="picoauto">PICO auto · low/high best profile</option>
          <option value="pico2">PICO v4 perceptual · OXP6</option>
          <option value="pico5">PICO v5 sparse low-rate · OXP7</option>
          <option value="picomoe">PICO MoE 6 rate experts · OXP9</option>
          <option value="tiny">OXNT tiny · 88× fewer params</option>
          <option value="tiny2">Tiny v4 perceptual · OXNV + scale boost</option>
        </select>
      </label>
      <span style="color:var(--dim)">quality</span>
      <input type="number" id="q" value="100" min="1" max="100">
      <label style="color:var(--dim)" title="Только tiny/PICO: 0 = обычный quality, >0 = подобрать q под целевой битрейт (несколько проходов)">target bpp</label>
      <input type="number" id="tb" value="0" min="0" max="24" step="0.01" style="width:90px">
      <label style="color:var(--dim)" title="Включено: адаптивная степень по патчам. Выключено: плоская степень. Оба режима используют детерминированный OXN8.">
        <input type="checkbox" id="ad" checked> adaptive OXN8 (off = flat OXN8)
      </label>
      <label style="color:var(--dim)" title="Быстрый выбор адаптивной степени маленькой нейросетью; цена — около 0.5 дБ PSNR">
        <input type="checkbox" id="bs" checked> boost speed for adaptive
      </label>
      <label style="color:var(--dim)" title="Не пережимать файлы, чей результат уже есть в папке сохранения; отжать — перезаписывать">
        <input type="checkbox" id="sk" checked> skip existing
      </label>
      <label style="color:var(--dim)" title="После сжатия декодировать поток обратно и сверить (медленно на больших файлах)">
        <input type="checkbox" id="vf"> verify
      </label>
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
let activeModel = 'oxn8';
let modelBusy = false;

fetch('/api/info').then(r => r.json()).then(i => {
  if (i.model) {
    activeModel = i.model;
    $('md').value = i.model;
    updateMode();
  }
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
$('file').onchange = e => { addFiles([...e.target.files]); e.target.value = ''; };
function updateMode() {
  const model = $('md').value;
  $('tb').disabled = !model.startsWith('pico') && !model.startsWith('tiny');
  $('q').disabled = !$('tb').disabled && Number($('tb').value) > 0;
  $('ad').disabled = model.startsWith('pico');
  $('bs').disabled = !$('ad').checked || model !== 'oxn8';
  $('bs').title = model.startsWith('tiny') ? 'Tiny использует собственный scale predictor' :
    model.startsWith('oxn17') ? 'OXN17 использует собственный encoder без OXN8 boost' : '';
}
$('ad').onchange = updateMode;
$('tb').oninput = updateMode;
$('md').onchange = async () => {
  const previous = activeModel;
  const requested = $('md').value;
  modelBusy = true;
  busy(true);
  updateMode();
  try {
    const r = await fetch('/api/model', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model: requested})
    });
    if (!r.ok) throw new Error(await r.text());
    activeModel = requested;
    log('model loaded: ' + requested);
  } catch (e) {
    $('md').value = previous;
    log('model load failed: ' + e);
    updateMode();
  } finally {
    modelBusy = false;
    busy(false);
  }
};
updateMode();

function addFiles(fs) {
  files = files.concat(fs);
  renderList();
  busy(false);
  log('added ' + fs.length + ' file(s), total ' + files.length);
}
function renderList() {
  $('list').innerHTML = files.map((f, i) =>
    '<div class="file-row"><span class="nm">' + esc(f.name) +
    '</span><span class="st" data-name="' + esc(f.name) + '">' + fmt(f.size) +
    '</span><button class="rm" data-i="' + i + '" title="убрать из очереди">&times;</button></div>').join('');
}
$('list').onclick = e => {
  const b = e.target.closest('.rm');
  if (!b) return;
  files.splice(+b.dataset.i, 1);
  renderList();
  busy(false);
};
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
  $('bc').disabled = $('bd').disabled = on || modelBusy || !files.length;
  $('sd').disabled = on || modelBusy;
  $('md').disabled = on || modelBusy;
}

// ---------- куда сохранять ----------
$('sd').onclick = async () => {
  if (!window.showDirectoryPicker) {
    log('folder picker not supported by this browser — files go to Downloads');
    return;
  }
  try {
    saveDir = await showDirectoryPicker({ mode: 'readwrite' });
    $('sdpath').textContent = saveDir.name;
  } catch (e) {
    if (e && (e.name === 'NotAllowedError' || e.name === 'SecurityError'))
      log('browser blocks this folder (system dir like Downloads/Desktop) — pick a subfolder inside it');
    /* отмена — остаётся прежняя папка или Downloads */
  }
};
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
      if ($('sk').checked && saveDir) {
        const outName = op === 'compress' ? f.name + '.oxh'
          : f.name.replace(/\.oxh$/i, '') + '.jpg';
        try {
          await saveDir.getFileHandle(outName);   // существует — не жмём повторно
          mark(f.name, 'exists — skipped', ''); skipped++; continue;
        } catch (e) { /* NotFoundError: результата ещё нет */ }
      }
      const fd = new FormData();
      fd.append('file', f);
      fd.append('q', $('q').value);
      fd.append('target_bpp', $('tb').value);
      fd.append('model', $('md').value);
      fd.append('adaptive', $('ad').checked ? '1' : '0');
      fd.append('boost', $('bs').checked ? '1' : '0');
      fd.append('verify', $('vf').checked ? '1' : '0');
      try {
        const r = await fetch('/api/' + op, { method: 'POST', body: fd });
        if (r.status === 415) {           // неподдерживаемый формат — молча
          mark(f.name, 'skipped', ''); skipped++; continue;
        }
        if (!r.ok) { mark(f.name, 'error', 'err'); log(f.name + ': ' + await r.text()); continue; }
        const st = JSON.parse(r.headers.get('X-OX-Stats') || '{}');
        const blob = await r.blob();
        await saveFile(st.out_name || f.name, blob);
        mark(f.name, op === 'compress'
          ? fmt(st.in) + ' → ' + fmt(st.out) + ' (' + st.ratio + '%)'
          : 'done', 'ok');
        done++;
      } catch (e) {
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


def _codec():
    global _CODEC
    if _CODEC is None:
        import ox_neuro
        _CODEC = ox_neuro
    return _CODEC


def _model_key(model):
    if model in ('oxn8', 'tiny', 'tiny2', 'pico', 'pico2', 'pico5', 'picomoe'):
        return model
    if model in ('oxn17', 'oxn17fast'):
        return 'oxn17'
    raise ValueError(f'unknown model: {model}')


_MODEL_FILES = {
    'oxn8': (CKPT, CTX7),
    'oxn17': (REC_CKPT, OXN13_CTX),
    'tiny': (TINY_CKPT, TINY_CTX7),
    'tiny2': (TINY2_CKPT, TINY2_CTX7, TINY2_BOOST),
    'pico': (PICO_CKPT,),
    'pico2': (PICO2_CKPT,),
    'pico5': (PICO5_CKPT,),
    'picomoe': (PICO_MOE_CKPT,),
}


def _unload_models_locked():
    global _ACTIVE_MODEL
    _ACTIVE_MODEL = None
    _RESIDENT_STATES.clear()
    if _CODEC is not None:
        ox = _CODEC
        ox._MODEL_CACHE.clear()
        ox._LEGACY_CPU_MODEL_CACHE = None
        ox._BOOST = None
        ox._CTX = None
        ox._CTX7 = None
        ox._CTX10 = None
        ox._QCTX7 = None
    pico = sys.modules.get('oxn_pico')
    if pico is not None:
        pico._RUNTIME_CACHE.clear()
    gc.collect()
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _clear_aux_models_locked():
    if _CODEC is not None:
        ox = _CODEC
        ox._LEGACY_CPU_MODEL_CACHE = None
        ox._CTX = None
        ox._CTX7 = None
        ox._CTX10 = None
        ox._QCTX7 = None
    gc.collect()
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_model_locked(key):
    if key in ('pico', 'pico2', 'pico5', 'picomoe'):
        import torch
        import oxn_pico
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        fast = oxn_pico._default_fast(dev)
        ckpt = PICO_MOE_CKPT if key == 'picomoe' else PICO5_CKPT if key == 'pico5' else \
            PICO2_CKPT if key == 'pico2' else PICO_CKPT
        oxn_pico._runtime(ckpt, fast)
        return dev
    ox = _codec()
    if key in ('tiny', 'tiny2'):
        ox.CTX7_FILE = TINY2_CTX7 if key == 'tiny2' else ox.TINY_CTX7
        model, dev = ox._load(TINY2_CKPT if key == 'tiny2' else TINY_CKPT)
        ox._qctx7_net(model.latent)
        if key == 'tiny2':
            if ox.BOOST_FILE != TINY2_BOOST:
                ox.BOOST_FILE = TINY2_BOOST
                ox._BOOST = None
            ox._boost_net(dev)
    else:
        ox.CTX7_FILE = ox.OXN8_CTX7
        if key == 'oxn17':
            model, dev = ox._load(REC_CKPT)
            ox._ctx10_net(model.latent, dev, OXN13_CTX)
        else:
            model, dev = ox._load(CKPT)
            ox._qctx7_net(model.latent)
            default_boost = os.path.join(HERE, 'ox_boost.pt')
            if ox.BOOST_FILE != default_boost:
                ox.BOOST_FILE = default_boost
                ox._BOOST = None
            ox._boost_net(dev)
    return dev


def _capture_model_state(key):
    if _CODEC is None:
        return {}
    ox = _CODEC
    if key == 'oxn17':
        return {'CTX10': ox._CTX10}
    if key in ('oxn8', 'tiny', 'tiny2'):
        state = {'CTX7': ox._CTX7, 'QCTX7': ox._QCTX7}
        if key in ('oxn8', 'tiny2'):
            state.update({'BOOST': ox._BOOST, 'BOOST_FILE': ox.BOOST_FILE})
        return state
    return {}


def _restore_model_state(key):
    if key in ('pico', 'pico2', 'pico5', 'picomoe') or _CODEC is None:
        return
    ox = _CODEC
    ox.CTX7_FILE = TINY2_CTX7 if key == 'tiny2' else \
        ox.TINY_CTX7 if key == 'tiny' else ox.OXN8_CTX7
    state = _RESIDENT_STATES[key]
    ox._CTX7 = state.get('CTX7')
    ox._QCTX7 = state.get('QCTX7')
    ox._CTX10 = state.get('CTX10')
    if 'BOOST' in state:
        ox._BOOST = state['BOOST']
        ox.BOOST_FILE = state['BOOST_FILE']


def _activate_model_locked(model, stream=False):
    global _ACTIVE_MODEL
    key = _model_key(model)
    if key == _ACTIVE_MODEL and not stream:
        return key
    required = _MODEL_FILES[key]
    if not all(os.path.exists(path) for path in required):
        raise RuntimeError('selected model is not trained yet')
    if not _RESIDENT:
        _unload_models_locked()
    else:
        _clear_aux_models_locked()
    if _RESIDENT and key in _RESIDENT_STATES:
        _restore_model_state(key)
        _ACTIVE_MODEL = key
        return key
    started = time.time()
    dev = _load_model_locked(key)
    if _RESIDENT:
        _RESIDENT_STATES[key] = _capture_model_state(key)
    _ACTIVE_MODEL = key
    print(f'model {key} loaded on {dev} in {time.time() - started:.2f}s', flush=True)
    return key


def _activate_model(model, stream=False):
    with _CODEC_LOCK:
        return _activate_model_locked(model, stream)


def _init_models():
    global _RESIDENT
    files = {path for paths in _MODEL_FILES.values() for path in paths
             if os.path.exists(path)}
    total = sum(os.path.getsize(path) for path in files)
    try:
        free = os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    except (AttributeError, OSError, ValueError):
        free = 4 * 1024 ** 3
    _RESIDENT = total <= min(2 * 1024 ** 3, free // 8)
    if _RESIDENT:
        for key in _MODEL_FILES:
            if all(os.path.exists(path) for path in _MODEL_FILES[key]):
                _activate_model(key)
    _activate_model('oxn8')
    print(f'model cache: {"resident" if _RESIDENT else "exclusive"}', flush=True)


def _codec_call(ox, fn, *args, **kwargs):
    global _RESIDENT
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        if 'CUDA' not in str(e) or 'memory' not in str(e) or ox._FORCE_CPU:
            raise
        ox._FORCE_CPU = True
        _RESIDENT = False
        _unload_models_locked()
        import torch
        torch.cuda.empty_cache()
        return fn(*args, **kwargs)


def _release_workspace():
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _release_after(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        finally:
            _release_workspace()
    return wrapped


@_release_after
def do_compress(fields, files):
    name, data = files['file']
    if not name.lower().endswith(IMG_EXT):
        raise Skip(name)
    adaptive = fields.get('adaptive', '1') == '1'
    model = fields.get('model', 'oxn8')
    q = int(fields.get('q', '100'))
    target_bpp = float(fields.get('target_bpp', 0) or 0)
    if model == 'picoauto':
        low = (target_bpp > 0 and target_bpp <= 0.25) or (target_bpp == 0 and q <= 10)
        model = 'pico5' if low else 'pico2'
    if not os.path.exists(CANON8):
        raise RuntimeError('no oxn8_canon.npz — see README.md')
    if model in ('tiny', 'tiny2'):
        required = (TINY2_CKPT, TINY2_CTX7, TINY2_BOOST) if model == 'tiny2' else \
            (TINY_CKPT, TINY_CTX7)
    elif model.startswith('oxn17'):
        required = (REC_CKPT, OXN13_CTX, CANON14, CANON15)
    elif model in ('pico', 'pico2', 'pico5', 'picomoe'):
        required = (PICO_MOE_CKPT,) if model == 'picomoe' else \
            (PICO5_CKPT,) if model == 'pico5' else \
            (PICO2_CKPT,) if model == 'pico2' else (PICO_CKPT,)
    else:
        required = (CKPT, CTX7)
    if not all(os.path.exists(p) for p in required):
        raise RuntimeError('selected model is not trained yet')
    if target_bpp > 0 and model not in ('tiny', 'tiny2', 'pico', 'pico2', 'pico5', 'picomoe'):
        raise ValueError('target bpp is available for tiny and PICO')
    ox = _codec()
    from PIL import Image, ImageOps
    rgb = ox.np.asarray(ImageOps.exif_transpose(
        Image.open(io.BytesIO(data))).convert('RGB'))
    t0 = time.time()
    with _CODEC_LOCK:
        _activate_model_locked(model)
        ox._BOOST_ON = adaptive and model == 'oxn8' and fields.get('boost') == '1'
        ox._VERIFY = fields.get('verify') == '1'
        ox.CTX7_FILE = TINY2_CTX7 if model == 'tiny2' else \
            ox.TINY_CTX7 if model == 'tiny' else ox.OXN8_CTX7
        if model in ('tiny', 'tiny2') and target_bpp > 0:
            out = _codec_call(ox, ox.neuro_compress_tiny_v2_target_bpp, rgb,
                              target_bpp, TINY2_CKPT if model == 'tiny2' else TINY_CKPT,
                              TINY2_CTX7 if model == 'tiny2' else TINY_CTX7,
                              magic=b'OXNV' if model == 'tiny2' else b'OXNT',
                              use_boost=model == 'tiny2')
        elif model in ('tiny', 'tiny2'):
            out = _codec_call(ox, ox.neuro_compress_tiny_v2, rgb, q, not adaptive,
                              TINY2_CKPT if model == 'tiny2' else TINY_CKPT,
                              TINY2_CTX7 if model == 'tiny2' else TINY_CTX7,
                              magic=b'OXNV' if model == 'tiny2' else b'OXNT',
                              use_boost=model == 'tiny2')
        elif model == 'oxn17fast':
            out = _codec_call(ox, ox.neuro_compress_v17_fast, rgb, q, not adaptive)
        elif model == 'oxn17':
            out = _codec_call(ox, ox.neuro_compress_v17, rgb, q, not adaptive)
        elif model in ('pico', 'pico2', 'pico5', 'picomoe'):
            import oxn_pico
            ckpt = PICO_MOE_CKPT if model == 'picomoe' else PICO5_CKPT if model == 'pico5' else \
                PICO2_CKPT if model == 'pico2' else PICO_CKPT
            version = 9 if model == 'picomoe' else 7 if model == 'pico5' else 6 if model == 'pico2' else 5
            if target_bpp > 0:
                out = oxn_pico.compress_target_bpp(
                    rgb, target_bpp, ckpt, fields.get('verify') == '1',
                    stream_version=version)
            else:
                out = oxn_pico.compress(rgb, q, ckpt, fields.get('verify') == '1',
                                        stream_version=version)
        else:
            out = _codec_call(ox, ox.neuro_compress_v8, rgb, q, None, not adaptive)
    dt = time.time() - t0
    n, on = len(data), len(out)
    stats = {
        'in': n, 'out': on,
        'ratio': round(100.0 * on / max(n, 1), 1),
        'time': round(dt, 1),
        'out_name': name + '.oxh',
    }
    return out, stats


@_release_after
def do_decompress(fields, files):
    name, data = files['file']
    # Формат выбирает декодер по magic-байтам, включая legacy OXN1/OXN2.
    if not name.lower().endswith('.oxh') or data[:4] not in (
            b'OXN1', b'OXN2', b'OXN3', b'OXN4', b'OXN5', b'OXN6', b'OXN7',
            b'OXN8', b'OXN9', b'OXNT', b'OXNV', b'OXNA', b'OXNB', b'OXNC', b'OXND', b'OXNE',
            b'OXNG', b'OXNH', b'OXNI', b'OXP1', b'OXP2', b'OXP3', b'OXP4', b'OXP5', b'OXP6', b'OXP7', b'OXP8', b'OXP9'):
        raise Skip(name)
    base = name[:-4]
    ox = _codec()
    t0 = time.time()
    with _CODEC_LOCK:
        if data[:4] in (b'OXP1', b'OXP2', b'OXP3', b'OXP4', b'OXP5', b'OXP6', b'OXP7', b'OXP8', b'OXP9'):
            model = 'picomoe' if data[:4] in (b'OXP8', b'OXP9') else \
                'pico5' if data[:4] == b'OXP7' else 'pico2' if data[:4] == b'OXP6' else 'pico'
            _activate_model_locked(model, stream=True)
            import oxn_pico
            ckpt = PICO_MOE_CKPT if model == 'picomoe' else PICO5_CKPT if model == 'pico5' else \
                PICO2_CKPT if model == 'pico2' else PICO_CKPT
            rec = oxn_pico.decompress(data, ckpt)
        else:
            model = ('tiny2' if data[:4] == b'OXNV' else
                     'tiny' if data[:4] in (b'OXN9', b'OXNT') else
                     'oxn17' if data[:4] in (b'OXNH', b'OXNI') else 'oxn8')
            _activate_model_locked(model, stream=True)
            if model == 'tiny2':
                rec = _codec_call(ox, ox.neuro_decompress, data, TINY2_CKPT, TINY2_CTX7)
            else:
                rec = _codec_call(ox, ox.neuro_decompress, data)
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rec).save(buf, 'JPEG', quality=95, subsampling=0)
    out = buf.getvalue()
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
            self._send(200, json.dumps({
                'cuda': cuda, 'gpu': gpu, 'model': _ACTIVE_MODEL, 'resident': _RESIDENT,
                'tiny': os.path.exists(TINY_CKPT) and os.path.exists(TINY_CTX7),
                'tiny2': all(os.path.exists(p) for p in (TINY2_CKPT, TINY2_CTX7, TINY2_BOOST)),
                'oxn17': all(os.path.exists(p) for p in
                             (REC_CKPT, OXN13_CTX, CANON14, CANON15)),
                'pico': os.path.exists(PICO_CKPT),
                'pico2': os.path.exists(PICO2_CKPT),
                'pico5': os.path.exists(PICO5_CKPT),
                'picomoe': os.path.exists(PICO_MOE_CKPT)
            }).encode(), 'application/json')
        else:
            self._send(404, b'not found')

    def do_POST(self):
        try:
            if self.path == '/api/model':
                length = int(self.headers.get('Content-Length', 0))
                if length <= 0 or length > 4096:
                    raise ValueError('invalid model request')
                fields = json.loads(self.rfile.read(length))
                key = _activate_model(fields['model'])
                self._send(200, json.dumps({'model': key}).encode(),
                           'application/json')
                return
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
    if not os.path.exists(CKPT) or not os.path.exists(CTX7) or not os.path.exists(CANON8):
        print('warning: no ox_neuro.pt / ox_ctx7.pt / oxn8_canon.npz — see README.md',
              file=sys.stderr)
    else:
        _init_models()
    print(f'OXH UI: http://{host}:{port}')
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == '__main__':
    main()
