"""The main work area: an editor-style canvas (Gradio 6 HTML component).

Replaces Upscayl's ImageViewer / SliderView / LensViewer / ProgressBar with one
surface that keeps its own view state:

- fit-to-view on load, wheel zoom around the cursor, drag to pan,
  0 = fit, 1 = 100 %, +/- zoom, double-click toggles fit / 100 %
- drop, paste (Ctrl+V) or Open / double-click the empty canvas to load an image
- after a run the same canvas shows a before/after slider (divider in screen
  space, both images under one zoom/pan) or a magnifier lens
- Upscayl's progress overlay (spinner, message, bar, STOP) drawn on the canvas

The HTML template is a constant skeleton so Gradio's re-render diff never
touches it; everything is drawn imperatively from `props.value` in a
`watch('value')` callback, which keeps zoom/pan across progress updates.

Python -> JS: the component value (see `value()`).
JS -> Python events: image_upload {path, name}, stop, view {view}, open_folder,
selection {path, w, h, bbox} (mask PNG; path null = no selection), crop {crop: [x, y, w, h] | null}.
"""

from __future__ import annotations

import json
import time
import urllib.parse
from pathlib import Path

import gradio as gr

from . import toolbar

JS_DIR = Path(__file__).resolve().parent / "js"


def file_url(path: str) -> str:
    # Cache-bust so a re-run with the same output name refreshes.
    return f"/gradio_api/file={urllib.parse.quote(path)}?t={int(time.time() * 1000)}"


def value(mode: str = "empty", *, src: str = "", before: str = "", after: str = "", name: str = "",
          dims: str = "", out_dims: str = "", view: str = "slider", batch: bool = False, text: str = "",
          download_name: str = "", progress: dict | None = None, reset_view: bool = False,
          orig: str = "", view_box: list[int] | None = None, focus: list[float] | None = None) -> dict:
    """Build the canvas value. Paths are server paths; they are turned into URLs here."""
    return {
        "mode": mode,                       # empty | image | compare | batch
        "src": file_url(src) if src else "",
        "before": file_url(before) if before else "",
        "after": file_url(after) if after else "",
        "name": name,
        "dims": dims,
        "out_dims": out_dims,
        "view": view,                       # slider | lens
        "batch": batch,                     # empty-state wording
        "text": text,                       # batch mode message
        "download_name": download_name,
        "progress": progress,               # None | {"msg", "pct" (0..1 or None), "batch"}
        "nonce": time.time() if reset_view else None,
        # The full original the selection tools analyse, and the part of it the canvas shows
        # ([x, y, w, h] in its pixels; a crop after a cropped run). Defaults: the shown image.
        "orig": file_url(orig) if orig else "",
        "view_box": view_box,
        # A selection result's box in the shown image (fractions [x0, y0, x1, y1]): the slider
        # divider starts just left of it, so the whole selected area shows the upscaled pixels.
        "focus": focus,
    }


def with_progress(base: dict, msg: str, pct: float | None = None, batch: str = "") -> dict:
    return {**base, "progress": {"msg": msg, "pct": pct, "batch": batch}, "nonce": base.get("nonce")}


HTML_TEMPLATE = '<div class="ds-cv" tabindex="0"></div>'

CSS_TEMPLATE = """
.ds-cv { position: relative; width: 100%; height: calc(100vh - 162px); min-height: 440px; overflow: hidden;
  border-radius: 12px; outline: none; user-select: none; touch-action: none;
  background-color: #12161b;
  background-image: linear-gradient(45deg, #171c22 25%, transparent 25%), linear-gradient(-45deg, #171c22 25%, transparent 25%),
    linear-gradient(45deg, transparent 75%, #171c22 75%), linear-gradient(-45deg, transparent 75%, #171c22 75%);
  background-size: 24px 24px; background-position: 0 0, 0 12px, 12px -12px, -12px 0; }
.ds-cv.ds-pan { cursor: grab; } .ds-cv.ds-panning { cursor: grabbing; }
.ds-cv-stage { position: absolute; left: 0; top: 0; transform-origin: 0 0; will-change: transform; }
.ds-cv-stage img { position: absolute; left: 0; top: 0; width: 100%; height: 100%; display: block; pointer-events: none;
  -webkit-user-drag: none; }
.ds-cv-stage.ds-px img { image-rendering: pixelated; }
.ds-cv-clip { position: absolute; inset: 0; overflow: hidden; }
.ds-cv-divider { position: absolute; top: 0; bottom: 0; width: 2px; margin-left: -1px; background: rgba(255,255,255,.9);
  box-shadow: 0 0 8px rgba(0,0,0,.6); cursor: ew-resize; z-index: 4; }
.ds-cv-divider::before { content: ""; position: absolute; inset: 0 -10px; }
.ds-cv-knob { position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%); width: 36px; height: 36px; border-radius: 50%;
  background: rgba(255,255,255,.95); color: #222; display: flex; align-items: center; justify-content: center; font-size: 17px;
  box-shadow: 0 2px 10px rgba(0,0,0,.5); }
.ds-cv-tag { position: absolute; bottom: 92px; padding: 3px 9px; border-radius: 6px; background: rgba(0,0,0,.6); color: #fff;
  font-size: 12px; font-weight: 600; z-index: 3; pointer-events: none; }
.ds-cv-hud { position: absolute; bottom: 14px; left: calc(var(--ds-inset, 0px) + 10px); z-index: 5; display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
  max-width: calc(100% - var(--ds-inset, 0px) - 20px); pointer-events: none; }
.ds-cv.ds-hud-docked .ds-cv-hud { top: var(--ds-hud-y); bottom: auto; left: var(--ds-hud-x); max-width: var(--ds-hud-w); }
.ds-cv.ds-hud-docked .ds-cv-chip { max-width: 100%; }
.ds-cv-chip { background: rgba(10,12,16,.72); color: #e5e7eb; backdrop-filter: blur(6px); border: 1px solid rgba(255,255,255,.08);
  border-radius: 8px; padding: 4px 10px; font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 46vw; }
.ds-cv-chip b { color: #fff; }
.ds-cv-bar { position: absolute; bottom: 10px; left: calc(var(--ds-inset, 0px) + (100% - var(--ds-inset, 0px) - var(--ds-inset-r, 0px)) / 2); transform: translateX(-50%); z-index: 6; display: flex; gap: 4px;
  align-items: center; padding: 5px; border-radius: 12px; background: rgba(10,12,16,.78); backdrop-filter: blur(8px);
  border: 1px solid rgba(255,255,255,.08); box-shadow: 0 6px 20px rgba(0,0,0,.4); }
.ds-cv-bar button, .ds-cv-bar a { all: unset; cursor: pointer; color: #e5e7eb; font-size: 13px; padding: 5px 10px; border-radius: 8px;
  line-height: 1.2; white-space: nowrap; }
.ds-cv-bar button:hover, .ds-cv-bar a:hover { background: rgba(255,255,255,.1); }
.ds-cv-bar button.on { background: linear-gradient(90deg, #7c3aed, #6366f1); color: #fff; }
.ds-cv-bar .ds-cv-zoom { min-width: 52px; text-align: center; font-variant-numeric: tabular-nums; }
.ds-cv-bar .sep { width: 1px; height: 20px; background: rgba(255,255,255,.12); margin: 0 3px; }
.ds-cv-empty { position: absolute; inset: 0; padding-left: var(--ds-inset, 0px); padding-right: var(--ds-inset-r, 0px); display: flex; align-items: center; justify-content: center; z-index: 2; }
.ds-cv-card { display: flex; flex-direction: column; align-items: center; gap: 8px; padding: 26px 34px; border-radius: 16px;
  background: rgba(21,25,30,.9); border: 1px dashed rgba(255,255,255,.16); color: #e5e7eb; text-align: center; max-width: 520px;
  cursor: pointer; transition: border-color .15s, transform .15s; }
.ds-cv-card:hover { border-color: #8b5cf6; transform: translateY(-1px); }
.ds-cv-card .t { font-size: 1.15rem; font-weight: 600; } .ds-cv-card .d { font-size: .86rem; opacity: .7; }
.ds-cv-card .k { font-size: .75rem; opacity: .5; margin-top: 4px; }
.ds-cv-batch { cursor: default; border-style: solid; white-space: pre-line; }
.ds-cv-lens { position: absolute; width: 110px; height: 110px; border: 2px solid rgba(255,255,255,.95); border-radius: 6px;
  box-shadow: 0 0 0 9999px rgba(0,0,0,.28); pointer-events: none; z-index: 4; display: none; }
.ds-cv-lenspanel { position: absolute; display: none; gap: 6px; pointer-events: none; z-index: 5; }
.ds-cv-lensbox { position: relative; width: 230px; height: 230px; border-radius: 10px; background-color: #000; background-repeat: no-repeat;
  border: 2px solid rgba(255,255,255,.9); box-shadow: 0 8px 22px rgba(0,0,0,.5); }
.ds-cv-lensbox.orig { image-rendering: pixelated; }
.ds-cv-lensbox span { position: absolute; bottom: 5px; left: 5px; font-size: 11px; background: rgba(0,0,0,.65); color: #fff;
  padding: 1px 6px; border-radius: 4px; }
.ds-cv-over { position: absolute; inset: 0; z-index: 20; display: none; align-items: center; justify-content: center;
  background: rgba(10,12,16,.45); backdrop-filter: blur(10px); }
.ds-cv-over.show { display: flex; }
.ds-cv-ocard { display: flex; flex-direction: column; align-items: center; gap: 8px; padding: 24px 30px; min-width: 300px;
  border-radius: 16px; background: rgba(25,30,36,.82); border: 1px solid rgba(255,255,255,.08); color: #e5e7eb; }
.ds-cv-spin { width: 54px; height: 54px; border-radius: 14px; animation: ds-cv-spin 1.4s ease-in-out infinite; }
@keyframes ds-cv-spin { 0% { transform: rotate(0) scale(1); } 50% { transform: rotate(180deg) scale(.88); } 100% { transform: rotate(360deg) scale(1); } }
.ds-cv-ocard .b { font-weight: 700; } .ds-cv-ocard .m { font-weight: 600; font-size: .92rem; text-align: center; max-width: 420px; }
.ds-cv-ocard .s { font-size: .74rem; opacity: .55; animation: ds-cv-pulse 1.6s ease-in-out infinite; }
@keyframes ds-cv-pulse { 50% { opacity: .2; } }
.ds-cv-track { width: 260px; height: 6px; border-radius: 99px; background: rgba(255,255,255,.1); overflow: hidden; }
.ds-cv-fill { height: 100%; width: 0; background: linear-gradient(90deg, #7c3aed, #6366f1); transition: width .2s; }
.ds-cv-track.indet .ds-cv-fill { width: 35%; animation: ds-cv-indet 1.2s ease-in-out infinite; }
@keyframes ds-cv-indet { 0% { transform: translateX(-100%); } 100% { transform: translateX(300%); } }
.ds-cv-stop { all: unset; cursor: pointer; margin-top: 6px; padding: 6px 22px; border-radius: 8px; font-weight: 600; font-size: .85rem;
  border: 1px solid rgba(255,255,255,.35); color: #fff; }
.ds-cv-stop:hover { background: rgba(239,68,68,.25); border-color: #ef4444; }
.ds-cv-drop { position: absolute; inset: 8px; left: calc(var(--ds-inset, 0px) + 8px); right: calc(var(--ds-inset-r, 0px) + 8px); z-index: 25; display: none; align-items: center; justify-content: center;
  border: 2px dashed #8b5cf6; border-radius: 12px; background: rgba(139,92,246,.12); color: #fff; font-weight: 600; font-size: 1.1rem;
  pointer-events: none; }
.ds-cv.ds-dragover .ds-cv-drop { display: flex; }
"""

JS_ON_LOAD = r"""
const root = element.querySelector('.ds-cv');
var SEL_READY = false;   // set once the selection tools have initialised
const LOGO = '__LOGO__';
const VALID = ['png', 'jpg', 'jpeg', 'jfif', 'webp', 'bmp', 'tif', 'tiff'];
const st = { s: 1, tx: 0, ty: 0, w: 0, h: 0, pos: 0.5, fitted: false, nonce: null, key: '', focus: null };
let drag = null, cursor = null;

root.innerHTML = `
  <div class="ds-cv-stage"></div>
  <div class="ds-cv-divider" style="display:none"><span class="ds-cv-knob">&#x2194;</span></div>
  <span class="ds-cv-tag" data-side="l" style="display:none">Original</span>
  <span class="ds-cv-tag" data-side="r" style="display:none">Upscaled</span>
  <div class="ds-cv-lens"></div>
  <div class="ds-cv-lenspanel"><div class="ds-cv-lensbox orig"><span>Original</span></div><div class="ds-cv-lensbox up"><span>Upscaled</span></div></div>
  <div class="ds-cv-empty"></div>
  <div class="ds-cv-hud"></div>
  <div class="ds-cv-bar" style="display:none"></div>
  <div class="ds-cv-over"><div class="ds-cv-ocard">
     <img class="ds-cv-spin" src="${LOGO}"/><div class="b"></div><div class="m"></div>
     <div class="ds-cv-track"><div class="ds-cv-fill"></div></div>
     <div class="s">Working its magic...</div><button class="ds-cv-stop">STOP</button></div></div>
  <div class="ds-cv-drop">Drop image to open</div>
  <input type="file" accept="image/*" style="display:none"/>`;
const q = (sel) => root.querySelector(sel);
const stage = q('.ds-cv-stage'), divider = q('.ds-cv-divider'), lens = q('.ds-cv-lens'), lensPanel = q('.ds-cv-lenspanel');
const empty = q('.ds-cv-empty'), hud = q('.ds-cv-hud'), bar = q('.ds-cv-bar'), over = q('.ds-cv-over'), fileInput = q('input[type=file]');
const V = () => props.value || {};
const esc = (t) => String(t ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const vw = () => root.clientWidth, vh = () => root.clientHeight;
// The step panel floats over the canvas' left edge; treat it as an inset so fitting,
// centring and the HUD/toolbar use the visible area to its right.
const panelEl = () => document.querySelector('#ds-sidebar');
function insetL() {
  const p = panelEl();
  if (!p || !p.offsetParent || getComputedStyle(p).position !== 'absolute') return 0;
  const pr = p.getBoundingClientRect(), r = root.getBoundingClientRect();
  return Math.max(0, Math.min(r.width * 0.6, pr.right - r.left + 12));
}
// The file chips dock right under the floating panel, at its width; with the panel
// collapsed (or no room left below it) they sit at the canvas' bottom-left corner.
function dockHud() {
  const p = panelEl();
  const on = p && p.offsetParent && getComputedStyle(p).position === 'absolute';
  let docked = false;
  if (on) {
    const pr = p.getBoundingClientRect(), r = root.getBoundingClientRect(), y = pr.bottom - r.top + 8;
    if (y + hud.offsetHeight <= r.height - 8) {
      root.style.setProperty('--ds-hud-x', (pr.left - r.left) + 'px');
      root.style.setProperty('--ds-hud-y', y + 'px');
      root.style.setProperty('--ds-hud-w', pr.width + 'px');
      docked = true;
    }
  }
  root.classList.toggle('ds-hud-docked', docked);
}
function layoutInset() {
  dockHud();
  const l = insetL();
  root.style.setProperty('--ds-inset', l + 'px');
  root.style.setProperty('--ds-inset-r', (typeof insetR === 'function' ? insetR() : 0) + 'px');
  return l;
}
// watch() only fires for Python-side updates, so local changes re-render explicitly.
const setValue = (v) => { props.value = v; render(); };

// ---------- view transform ----------
function apply() {
  stage.style.width = st.w + 'px'; stage.style.height = st.h + 'px';
  stage.style.transform = `translate(${st.tx}px, ${st.ty}px) scale(${st.s})`;
  stage.classList.toggle('ds-px', st.s >= 3);
  const z = q('.ds-cv-zoom'); if (z) z.textContent = Math.round(st.s * 100) + '%';
  if (SEL_READY) selOnView();
  const v = V();
  const slider = v.mode === 'compare' && v.view !== 'lens';
  divider.style.display = slider ? '' : 'none';
  const clipEl = stage.querySelector('.ds-cv-clip');
  if (clipEl && !slider) clipEl.style.clipPath = 'none';   // lens: the original fills the canvas, as in Upscayl
  root.querySelectorAll('.ds-cv-tag').forEach((t) => (t.style.display = slider ? '' : 'none'));
  if (slider && st.focus && st.w) {
    // Divider just left of the selection, kept off the very edge so it stays grabbable.
    const fx = st.tx + st.focus[0] * st.w * st.s - 12;
    st.pos = Math.max(insetL() + 14, Math.min(vw() - insetR() - 14, fx)) / Math.max(1, vw());
    st.focus = null;
  }
  if (slider) {
    const x = st.pos * vw();
    divider.style.left = x + 'px';
    q('.ds-cv-tag[data-side=l]').style.left = (insetL() + 10) + 'px';
    q('.ds-cv-tag[data-side=r]').style.right = (insetR() + 10) + 'px';
    const clip = stage.querySelector('.ds-cv-clip');
    if (clip) {
      const sx = Math.max(0, Math.min(st.w, (x - st.tx) / st.s));
      clip.style.clipPath = `inset(0 ${st.w - sx}px 0 0)`;
    }
  }
}
function fit() {
  if (!st.w || !st.h) return;
  const l = layoutInset(), aw = vw() - l - insetR();
  const s = Math.min((aw - 40) / st.w, (vh() - 90) / st.h);
  st.s = Math.max(0.02, Math.min(s, 64));
  st.tx = l + (aw - st.w * st.s) / 2; st.ty = (vh() - 50 - st.h * st.s) / 2 + 4;
  st.fitted = true; apply();
}
function zoomAt(f, cx, cy) {
  const ns = Math.max(0.02, Math.min(64, st.s * f));
  st.tx = cx - (cx - st.tx) * (ns / st.s); st.ty = cy - (cy - st.ty) * (ns / st.s);
  st.s = ns; st.fitted = false; apply();
}
const midX = () => insetL() + (vw() - insetL() - insetR()) / 2;
function actual(cx, cy) { zoomAt(1 / st.s, cx ?? midX(), cy ?? vh() / 2); }

// ---------- rendering from props.value ----------
function render() {
  const v = V();
  const key = [v.mode, v.src, v.before, v.after].join('|');
  const changed = key !== st.key;
  if (changed) st.focus = v.mode === 'compare' && v.focus ? v.focus : null;
  st.key = key;
  root.classList.toggle('ds-pan', (v.mode === 'image' || v.mode === 'compare') && (tb.tool === 'move' || tb.tool === 'hand'));

  // empty / batch card
  if (v.mode === 'empty' || v.mode === 'batch' || !v.mode) {
    stage.innerHTML = ''; st.w = st.h = 0;
    empty.style.display = 'flex';
    empty.innerHTML = v.mode === 'batch'
      ? `<div class="ds-cv-card ds-cv-batch"><div class="t">${esc(v.text ? '' : 'Select a Folder')}</div><div class="d">${esc(v.text || "Make sure that the folder doesn't contain anything except PNG, JPG, JPEG & WEBP images.")}</div></div>`
      : v.batch
        ? `<div class="ds-cv-card ds-cv-batch"><div class="t">Select a Folder</div><div class="d">Use Step 1 in the left panel. Make sure that the folder doesn't contain anything except PNG, JPG, JPEG &amp; WEBP images.</div></div>`
        : `<div class="ds-cv-card" data-open="1"><div class="t">Select an Image</div><div class="d">Click, double-click or drag and drop a PNG, JPG, JPEG or WEBP image.</div><div class="d">Hit Ctrl+V or &#8984;+V to paste an image from the clipboard.</div><div class="k">Scroll to zoom · drag to pan · 0 fit · 1 = 100%</div></div>`;
    bar.style.display = 'none';
  } else {
    empty.style.display = 'none';
    if (changed) {
      if (v.mode === 'image') {
        stage.innerHTML = `<img class="ds-cv-single" src="${v.src}" draggable="false"/>`;
      } else {
        stage.innerHTML = `<img class="ds-cv-after" src="${v.after}" draggable="false"/><div class="ds-cv-clip"><img class="ds-cv-before" src="${v.before}" draggable="false"/></div>`;
      }
      const ref = stage.querySelector(v.mode === 'image' ? '.ds-cv-single' : '.ds-cv-after');
      const sameSize = v.mode === 'compare' && st.w > 0;
      const onload = () => {
        const prevW = st.w;
        st.w = ref.naturalWidth; st.h = ref.naturalHeight;
        if (sameSize && prevW && !st.fitted) {
          // Same picture, new resolution (result arrived): keep what's on screen.
          const k = prevW / st.w; st.s *= k;
          apply();
        } else { fit(); }
      };
      if (ref.complete && ref.naturalWidth) onload(); else ref.addEventListener('load', onload, { once: true });
    }
    renderBar(v);
  }
  if (v.nonce && v.nonce !== st.nonce) { st.nonce = v.nonce; st.pos = midX() / Math.max(1, vw()); if (st.w) fit(); }

  // HUD
  const chips = [];
  if (v.name) chips.push(`<span class="ds-cv-chip" title="${esc(v.name)}"><b>${esc(v.name)}</b></span>`);
  if (v.dims) chips.push(`<span class="ds-cv-chip">${esc(v.dims)}${v.out_dims ? ' &rarr; <b>' + esc(v.out_dims) + '</b>' : ''}</span>`);
  hud.innerHTML = chips.join('');
  dockHud();

  // progress overlay
  const p = v.progress;
  over.classList.toggle('show', !!p);
  if (p) {
    q('.ds-cv-ocard .b').textContent = p.batch || '';
    q('.ds-cv-ocard .m').textContent = p.msg || 'Hold on...';
    const track = q('.ds-cv-track'), fill = q('.ds-cv-fill');
    const pct = typeof p.pct === 'number' ? Math.max(0, Math.min(1, p.pct)) : null;
    track.classList.toggle('indet', pct === null);
    fill.style.width = pct === null ? '' : (pct * 100).toFixed(1) + '%';
  }
  apply();
  selOnRender(v, changed);
}

function renderBar(v) {
  const cmp = v.mode === 'compare';
  bar.style.display = 'flex';
  bar.innerHTML = `
    <button data-act="open" title="Open image (or drop / paste)">&#128194; Open</button><span class="sep"></span>
    <button data-act="out" title="Zoom out (-)">&minus;</button><button class="ds-cv-zoom" data-act="100" title="Actual size (1)">100%</button>
    <button data-act="in" title="Zoom in (+)">+</button><button data-act="fit" title="Fit to view (0)">Fit</button>
    ${cmp ? `<span class="sep"></span>
      <button data-act="slider" class="${v.view !== 'lens' ? 'on' : ''}">Slider</button>
      <button data-act="lens" class="${v.view === 'lens' ? 'on' : ''}">Lens</button><span class="sep"></span>
      <a data-act="dl" href="${v.after}" download="${esc(v.download_name || 'upscaled.png')}" title="Download result">&#11015; Download</a>
      <button data-act="folder" title="Open output folder">&#128193; Folder</button>` : ''}`;
}

// ---------- loading images (drop / paste / open) ----------
async function loadFile(file) {
  if (!file) return;
  const ext = (file.name.split('.').pop() || '').toLowerCase();
  if (!(file.type || '').startsWith('image') && !VALID.includes(ext)) { flash('Invalid image - use PNG, JPG, JPEG, JFIF or WEBP'); return; }
  setValue({ ...V(), progress: { msg: 'Loading image…', pct: null, batch: '' } });
  try {
    const name = file.name && file.name !== 'image.png' ? file.name : `pasted-${Date.now()}.${ext || 'png'}`;
    const f = file.name === name ? file : new File([file], name, { type: file.type || 'image/png' });
    const r = await upload(f);
    trigger('image_upload', { path: r.path, name });
  } catch (e) {
    console.error(e); setValue({ ...V(), progress: null }); flash('Upload failed');
  }
}
function flash(msg) {
  const c = document.createElement('div'); c.className = 'ds-cv-chip';
  c.style.cssText = 'position:absolute;top:50px;left:50%;transform:translateX(-50%);z-index:30;max-width:80%';
  c.textContent = msg; root.appendChild(c); setTimeout(() => c.remove(), 2600);
}
const openPicker = () => fileInput.click();
fileInput.addEventListener('change', () => { loadFile(fileInput.files[0]); fileInput.value = ''; });
root.addEventListener('dragover', (e) => { e.preventDefault(); root.classList.add('ds-dragover'); });
root.addEventListener('dragleave', (e) => { if (!root.contains(e.relatedTarget)) root.classList.remove('ds-dragover'); });
root.addEventListener('drop', (e) => { e.preventDefault(); root.classList.remove('ds-dragover'); loadFile(e.dataTransfer.files[0]); });
window.addEventListener('paste', (e) => {
  if (!root.offsetParent || V().progress) return;                      // tab hidden or busy
  const t = document.activeElement;
  if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable) && !root.contains(t)) return;
  const item = [...(e.clipboardData?.items || [])].find((i) => i.kind === 'file' && i.type.startsWith('image'));
  if (item) { e.preventDefault(); loadFile(item.getAsFile()); }
});

// ---------- toolbar / overlay buttons ----------
root.addEventListener('click', (e) => {
  const a = e.target.closest('[data-act]');
  if (e.target.closest('[data-open]')) { openPicker(); return; }
  if (e.target.closest('.ds-cv-stop')) { trigger('stop'); q('.ds-cv-ocard .m').textContent = 'Stopping…'; return; }
  if (!a) return;
  const act = a.dataset.act;
  if (act === 'open') openPicker();
  else if (act === 'fit') fit();
  else if (act === '100') actual();
  else if (act === 'in') zoomAt(1.25, midX(), vh() / 2);
  else if (act === 'out') zoomAt(0.8, midX(), vh() / 2);
  else if (act === 'slider' || act === 'lens') { setValue({ ...V(), view: act }); trigger('view', { view: act }); }
  else if (act === 'folder') trigger('open_folder');
});

// __TOOLBAR_JS__

// ---------- zoom / pan / divider / lens ----------
root.addEventListener('wheel', (e) => {
  if (!st.w || V().progress) return;
  e.preventDefault();
  const r = root.getBoundingClientRect();
  zoomAt(Math.pow(1.0015, -e.deltaY), e.clientX - r.left, e.clientY - r.top);
}, { passive: false });
root.addEventListener('pointerdown', (e) => {
  if (!st.w || e.button === 2 || e.target.closest('.ds-cv-bar, .ds-cv-over, .pc-tb, .pc-fly, .pc-opts')) return;
  const r = root.getBoundingClientRect();
  const x = e.clientX - r.left;
  const v = V();
  const onDivider = v.mode === 'compare' && v.view !== 'lens' && Math.abs(x - st.pos * vw()) < 14;
  if (!onDivider && e.button === 0 && toolPointerDown(e, x, e.clientY - r.top)) { drag = { kind: 'tool' }; root.setPointerCapture(e.pointerId); return; }
  drag = { kind: onDivider ? 'divider' : 'pan', x: e.clientX, y: e.clientY, tx: st.tx, ty: st.ty };
  root.setPointerCapture(e.pointerId);
  if (!onDivider) root.classList.add('ds-panning');
});
root.addEventListener('pointermove', (e) => {
  const r = root.getBoundingClientRect();
  cursor = { x: e.clientX - r.left, y: e.clientY - r.top };
  if (drag && drag.kind === 'tool') { toolPointerMove(e, cursor.x, cursor.y); return; }
  if (!drag) toolHover(cursor.x, cursor.y);
  if (drag) {
    if (drag.kind === 'divider') { st.pos = Math.max(0, Math.min(1, cursor.x / vw())); apply(); }
    else { st.tx = drag.tx + (e.clientX - drag.x); st.ty = drag.ty + (e.clientY - drag.y); st.fitted = false; apply(); }
  }
  updateLens();
});
const endDrag = () => { drag = null; root.classList.remove('ds-panning'); };
root.addEventListener('pointerup', (e) => {
  if (drag && drag.kind === 'tool') { const r = root.getBoundingClientRect(); drag = null; toolPointerUp(e, e.clientX - r.left, e.clientY - r.top); return; }
  endDrag();
});
root.addEventListener('pointercancel', endDrag);
root.addEventListener('pointerleave', () => { cursor = null; updateLens(); toolLeave(); });
root.addEventListener('dblclick', (e) => {
  if (e.target.closest('.ds-cv-bar, .ds-cv-over, .pc-tb, .pc-fly, .pc-opts')) return;
  if (!st.w) { openPicker(); return; }
  if (toolDblClick()) return;
  const r = root.getBoundingClientRect();
  if (st.fitted) actual(e.clientX - r.left, e.clientY - r.top); else fit();
});
root.addEventListener('keydown', (e) => {
  if (!st.w) return;
  if (toolKey(e) || toolShortcut(e)) { e.preventDefault(); e.stopPropagation(); return; }
  if (e.key === '0') fit();
  else if (e.key === '1') actual();
  else if (e.key === '+' || e.key === '=') zoomAt(1.25, midX(), vh() / 2);
  else if (e.key === '-') zoomAt(0.8, midX(), vh() / 2);
  else return;
  e.preventDefault();
});
root.addEventListener('pointerenter', () => root.focus({ preventScroll: true }));

function updateLens() {
  const v = V();
  const on = v.mode === 'compare' && v.view === 'lens' && cursor && !drag && st.w;
  if (!on) { lens.style.display = 'none'; lensPanel.style.display = 'none'; return; }
  const ix = (cursor.x - st.tx) / st.s, iy = (cursor.y - st.ty) / st.s;   // image coords (result pixels)
  if (ix < 0 || iy < 0 || ix > st.w || iy > st.h) { lens.style.display = 'none'; lensPanel.style.display = 'none'; return; }
  const L = 110, B = 230;
  lens.style.display = 'block'; lens.style.left = (cursor.x - L / 2) + 'px'; lens.style.top = (cursor.y - L / 2) + 'px';
  lensPanel.style.display = 'flex';
  let px = cursor.x - B - 3, py = cursor.y + L / 2 + 12;
  if (py + B > vh() - 56) py = cursor.y - L / 2 - B - 12;
  px = Math.max(insetL() + 6, Math.min(px, vw() - insetR() - 2 * B - 12));
  lensPanel.style.left = px + 'px'; lensPanel.style.top = Math.max(6, py) + 'px';
  const m = (B / L) * st.s;                     // screen px per image px inside the boxes
  const bw = st.w * m, bh = st.h * m, bx = -(ix * m - B / 2), by = -(iy * m - B / 2);
  for (const [sel, url] of [['.ds-cv-lensbox.orig', v.before], ['.ds-cv-lensbox.up', v.after]]) {
    const box = q(sel);
    box.style.backgroundImage = `url("${url}")`;
    box.style.backgroundSize = `${bw}px ${bh}px`;
    box.style.backgroundPosition = `${bx}px ${by}px`;
  }
}

const ro = new ResizeObserver(() => { layoutInset(); if (st.fitted) fit(); else apply(); });
ro.observe(root);
// Gradio removes a hidden column from the DOM and recreates it when shown, so watch
// for the panel coming and going, and re-attach the resize observer to it.
let watchedPanel = null, relayoutQueued = false;
const relayout = () => {
  if (relayoutQueued) return;
  relayoutQueued = true;
  setTimeout(() => {            // timer, not rAF: rAF is paused in background tabs
    relayoutQueued = false;
    const p = panelEl();
    if (p !== watchedPanel) { if (watchedPanel) ro.unobserve(watchedPanel); if (p) ro.observe(p); watchedPanel = p; }
    layoutInset(); if (st.fitted) fit(); else apply();
  }, 30);
};
// Observe the whole document: the row seen during the first render can be replaced
// by Gradio right after, which would leave a narrower observer watching a dead node.
new MutationObserver(relayout).observe(document.body, { childList: true, subtree: true });
relayout();
watch('value', render);
render();
"""


def _selection_js() -> str:
    """The selection tools with the PhotoCraft algorithm port embedded (also used by their worker)."""
    algo = (JS_DIR / "selection_algo.js").read_text(encoding="utf-8")
    tools = (JS_DIR / "selection_tools.js").read_text(encoding="utf-8")
    return tools.replace("__ALGO_SRC__", json.dumps(algo))


def component(logo_url: str, **kwargs) -> gr.HTML:
    return gr.HTML(
        value=value(),
        html_template=HTML_TEMPLATE,
        css_template=CSS_TEMPLATE + toolbar.CSS,
        js_on_load=JS_ON_LOAD.replace("__LOGO__", logo_url).replace(
            "// __TOOLBAR_JS__", toolbar.JS.replace("__PC_DATA__", toolbar.data_json()) + "\n" + _selection_js()),
        show_label=False,
        padding=False,
        elem_id="ds-canvas",
        **kwargs,
    )
