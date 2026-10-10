"""Before/after viewer - a port of Upscayl's SliderView (react-compare-slider
with hover zoom) and LensViewer, as a Gradio 6 HTML component."""

from __future__ import annotations

import time
import urllib.parse

import gradio as gr


def file_url(path: str) -> str:
    # Cache-bust so a re-run with the same output name refreshes.
    return f"/gradio_api/file={urllib.parse.quote(path)}?t={int(time.time() * 1000)}"


def value(before: str = "", after: str = "", mode: str = "slider", zoom: int = 100) -> dict:
    return {
        "before": file_url(before) if before else "",
        "after": file_url(after) if after else "",
        "mode": mode,
        "zoom": int(zoom),
    }


HTML_TEMPLATE = """
<div class="ds-compare" data-mode="${value.mode || 'slider'}" data-zoom="${value.zoom || 100}">
  ${value.after ? `
  <div class="ds-stage">
    <img class="ds-after" src="${value.after}" draggable="false" alt="Upscaled"/>
    <div class="ds-before-wrap"><img class="ds-before" src="${value.before}" draggable="false" alt="Original"/></div>
    <div class="ds-divider"><span class="ds-knob">&#x2194;</span></div>
    <span class="ds-tag ds-left">Original</span>
    <span class="ds-tag ds-right">Upscaled</span>
    <div class="ds-lens-cursor"></div>
    <div class="ds-lens-panel">
      <div class="ds-lens-box ds-lens-orig"><span>Original</span></div>
      <div class="ds-lens-box ds-lens-up"><span>Upscaled</span></div>
    </div>
  </div>` : ``}
</div>
"""

CSS_TEMPLATE = """
.ds-compare { width: 100%; user-select: none; }
.ds-stage { position: relative; width: 100%; height: 72vh; overflow: hidden; border-radius: 10px;
  background: linear-gradient(135deg, var(--background-fill-secondary), var(--background-fill-primary));
  --pos: 50%; touch-action: none; }
.ds-stage img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain;
  transition: transform 0.08s ease-out; will-change: transform; }
.ds-before-wrap { position: absolute; inset: 0; clip-path: inset(0 calc(100% - var(--pos)) 0 0); }
.ds-divider { position: absolute; top: 0; bottom: 0; left: var(--pos); width: 2px; margin-left: -1px;
  background: rgba(255,255,255,.85); box-shadow: 0 0 6px rgba(0,0,0,.5); cursor: ew-resize; }
.ds-knob { position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%); width: 34px; height: 34px;
  border-radius: 50%; background: rgba(255,255,255,.92); color: #333; display: flex; align-items: center;
  justify-content: center; font-size: 18px; box-shadow: 0 2px 8px rgba(0,0,0,.4); }
.ds-tag { position: absolute; bottom: 6px; padding: 2px 8px; border-radius: 6px; background: rgba(0,0,0,.55);
  color: #fff; font-size: 12px; font-weight: 500; opacity: .75; pointer-events: none; }
.ds-left { left: 6px; } .ds-right { right: 6px; }
.ds-lens-cursor, .ds-lens-panel { display: none; position: absolute; pointer-events: none; z-index: 5; }
.ds-lens-cursor { width: 100px; height: 100px; border: 2px solid rgba(255,255,255,.9);
  box-shadow: 0 0 0 9999px rgba(0,0,0,.25); border-radius: 4px; }
.ds-lens-panel { gap: 6px; }
.ds-lens-box { position: relative; width: 220px; height: 220px; border-radius: 8px; background-repeat: no-repeat;
  background-color: #000; border: 2px solid rgba(255,255,255,.85); box-shadow: 0 6px 18px rgba(0,0,0,.45);
  image-rendering: pixelated; }
.ds-lens-up { image-rendering: auto; }
.ds-lens-box span { position: absolute; bottom: 4px; left: 4px; font-size: 11px; background: rgba(0,0,0,.6);
  color: #fff; padding: 1px 6px; border-radius: 4px; }
.ds-compare[data-mode="lens"] .ds-before-wrap { clip-path: none; }
.ds-compare[data-mode="lens"] .ds-after, .ds-compare[data-mode="lens"] .ds-divider,
.ds-compare[data-mode="lens"] .ds-right { display: none; }
.ds-compare[data-mode="lens"] .ds-stage { cursor: crosshair; }
.ds-compare[data-mode="lens"] .ds-stage.ds-hover .ds-lens-cursor { display: block; }
.ds-compare[data-mode="lens"] .ds-stage.ds-hover .ds-lens-panel { display: flex; }
"""

JS_ON_LOAD = """
let pos = 50, dragging = false;
const LENS = 100, BOX = 220;
const q = (s) => element.querySelector(s);
function rectOf(img, stage) {
  // Visible image rect inside an object-fit: contain box.
  const r = stage.getBoundingClientRect();
  const nw = img.naturalWidth || r.width, nh = img.naturalHeight || r.height;
  const s = Math.min(r.width / nw, r.height / nh);
  const w = nw * s, h = nh * s;
  return { left: r.left + (r.width - w) / 2, top: r.top + (r.height - h) / 2, width: w, height: h, stage: r };
}
function setPos(e) {
  const stage = q('.ds-stage'); if (!stage) return;
  const r = stage.getBoundingClientRect();
  pos = Math.max(0, Math.min(100, (e.clientX - r.left) / r.width * 100));
  stage.style.setProperty('--pos', pos + '%');
}
function zoomAt(e) {
  const c = q('.ds-compare'), stage = q('.ds-stage'); if (!c || !stage) return;
  const z = (+c.dataset.zoom || 100) / 100;
  const imgs = stage.querySelectorAll('img');
  if (z <= 1) { imgs.forEach(i => { i.style.transform = ''; }); return; }
  const r = stage.getBoundingClientRect();
  const x = (e.clientX - r.left) / r.width * 100, y = (e.clientY - r.top) / r.height * 100;
  imgs.forEach(i => { i.style.transformOrigin = x + '% ' + y + '%'; i.style.transform = 'scale(' + z + ')'; });
}
function lens(e) {
  const stage = q('.ds-stage'), before = q('.ds-before'), after = q('.ds-after');
  if (!stage || !before) return;
  const ir = rectOf(before, stage);
  const inside = e.clientX >= ir.left && e.clientX <= ir.left + ir.width && e.clientY >= ir.top && e.clientY <= ir.top + ir.height;
  stage.classList.toggle('ds-hover', inside);
  if (!inside) return;
  const cx = e.clientX - ir.stage.left, cy = e.clientY - ir.stage.top;
  const cur = q('.ds-lens-cursor');
  cur.style.left = (cx - LENS / 2) + 'px'; cur.style.top = (cy - LENS / 2) + 'px';
  const panel = q('.ds-lens-panel');
  let px = cx - BOX - 3, py = cy + LENS / 2 + 12;
  if (py + BOX > ir.stage.height) py = cy - LENS / 2 - BOX - 12;
  px = Math.max(4, Math.min(px, ir.stage.width - 2 * BOX - 10));
  panel.style.left = px + 'px'; panel.style.top = Math.max(4, py) + 'px';
  const m = BOX / LENS;                       // magnification of the lens square
  const relX = e.clientX - ir.left, relY = e.clientY - ir.top;
  const bw = ir.width * m, bh = ir.height * m;
  const bx = -(relX * m - BOX / 2), by = -(relY * m - BOX / 2);
  for (const [sel, img] of [['.ds-lens-orig', before], ['.ds-lens-up', after]]) {
    const box = q(sel);
    box.style.backgroundImage = 'url("' + img.src + '")';
    box.style.backgroundSize = bw + 'px ' + bh + 'px';
    box.style.backgroundPosition = bx + 'px ' + by + 'px';
  }
}
element.addEventListener('pointerdown', (e) => {
  const c = q('.ds-compare'); if (!c || c.dataset.mode !== 'slider' || !q('.ds-stage')) return;
  dragging = true; setPos(e); e.preventDefault();
});
window.addEventListener('pointerup', () => { dragging = false; });
element.addEventListener('pointermove', (e) => {
  const c = q('.ds-compare'); if (!c) return;
  if (c.dataset.mode === 'lens') { lens(e); return; }
  if (dragging) setPos(e);
  zoomAt(e);
});
element.addEventListener('pointerleave', () => {
  const stage = q('.ds-stage'); if (!stage) return;
  stage.classList.remove('ds-hover');
  stage.querySelectorAll('img').forEach(i => { i.style.transform = ''; });
});
watch('value', () => { const s = q('.ds-stage'); if (s) s.style.setProperty('--pos', pos + '%'); });
"""


def component(**kwargs) -> gr.HTML:
    return gr.HTML(
        value=value(),
        html_template=HTML_TEMPLATE,
        css_template=CSS_TEMPLATE,
        js_on_load=JS_ON_LOAD,
        show_label=False,
        padding=False,
        **kwargs,
    )
