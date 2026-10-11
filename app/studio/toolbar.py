"""Floating tool palette on the canvas: a 1:1 copy of PhotoCraft's toolbar.

Layout, slots and flyouts follow PhotoCraft's `TOOL_SECTIONS`
(crates/ui-egui/src/panels.rs), laid out as its one-column view; sizes and
colours are its Pro theme tokens (crates/ui-egui/src/theme.rs): chrome #323232,
hover #424242, selected #4e4e4e with a #378ef0 border, icons #c8c8c8,
separators #1e1e1e, 30 px buttons with 20 px glyphs, 3 px radius. Icons are
PhotoCraft's SVGs (assets/photocraft-icons, see its ATTRIBUTION.md).

Only PhotoCraft's selection tools are kept (Move, the marquees, the lassos, Object /
Quick Selection, Magic Wand, Crop and Eyedropper). They select the part of the image
the model is applied to; the selection logic is studio/js/selection_tools.js on top of
the PhotoCraft algorithm port in studio/js/selection_algo.js.

The JS here runs inside the canvas component's closure (studio/canvas.py) so it
can use the canvas' view state and helpers.
"""

from __future__ import annotations

import json

from .paths import ASSETS_DIR

ICON_DIR = ASSETS_DIR / "photocraft-icons"

# id: (label, shortcut, icon, works in Deblur Studio)
TOOLS: dict[str, tuple[str, str, str, bool]] = {
    "move": ("Move Tool", "V", "move", True),
    "rect-marquee": ("Rectangular Marquee Tool", "M", "square-dashed", True),
    "ellipse-marquee": ("Elliptical Marquee Tool", "M", "circle-dashed", True),
    "row-marquee": ("Single Row Marquee Tool", "", "marquee-row", True),
    "column-marquee": ("Single Column Marquee Tool", "", "marquee-column", True),
    "lasso": ("Lasso Tool", "L", "lasso", True),
    "polygon-lasso": ("Polygonal Lasso Tool", "L", "pentagon", True),
    "magnetic-lasso": ("Magnetic Lasso Tool", "L", "lasso-magnetic", True),
    "object-selection": ("Object Selection Tool", "W", "square-dashed-mouse-pointer", True),
    "quick-selection": ("Quick Selection Tool", "W", "circle-dashed", True),
    "magic-wand": ("Magic Wand Tool", "W", "wand-sparkles", True),
    "crop": ("Crop Tool", "C", "crop", True),
    "eyedropper": ("Eyedropper Tool", "I", "pipette", True),
}

# Sections of slots; a slot is [shown tool, ...flyout tools]. A slot with more than one
# tool gets PhotoCraft's flyout triangle. Rendered one slot per row.
SECTIONS: list[list[list[str]]] = [
    [["move"]],
    [
        ["rect-marquee", "ellipse-marquee", "row-marquee", "column-marquee"],
        ["lasso", "polygon-lasso", "magnetic-lasso"],
        ["object-selection", "quick-selection", "magic-wand"],
        ["crop"],
        ["eyedropper"],
    ],
]

EXTRA_ICONS = ["check", "x"]


def _svg(name: str) -> str:
    text = (ICON_DIR / f"{name}.svg").read_text(encoding="utf-8")
    # Strip the fixed size so CSS controls it; keep viewBox and currentColor stroke.
    return " ".join(text.replace('width="24"', "").replace('height="24"', "").split())


def data_json() -> str:
    icons = sorted({t[2] for t in TOOLS.values()} | set(EXTRA_ICONS))
    return json.dumps({
        "tools": {k: {"label": v[0], "key": v[1], "icon": v[2], "ok": v[3]} for k, v in TOOLS.items()},
        "sections": SECTIONS,
        "icons": {name: _svg(name) for name in icons if (ICON_DIR / f"{name}.svg").exists()},
    })


CSS = """
.pc-tb { position: absolute; top: 16px; right: 16px; z-index: 28; box-sizing: border-box;
  display: flex; flex-direction: column; align-items: stretch;
  padding: 8px 5px; background: #323232; color: #c8c8c8; border: 1px solid #1e1e1e; border-radius: 8px;
  box-shadow: 0 18px 44px rgba(0,0,0,.55), 0 3px 10px rgba(0,0,0,.35); font-family: Inter, system-ui, sans-serif;
  max-height: calc(100% - 32px); user-select: none; touch-action: none; cursor: default; }
.pc-tb.pc-one { width: 40px; } .pc-tb.pc-two { width: 72px; }
.pc-tb svg { width: 100%; height: 100%; display: block; }
.pc-tb-head { all: unset; height: 14px; width: 30px; margin: 0 0 4px; color: #808080; cursor: pointer; display: flex; justify-content: center; }
.pc-tb-head:hover { color: #dedede; }
.pc-tb-head span { width: 11px; height: 11px; }
.pc-tb-scroll { overflow-y: auto; overflow-x: hidden; scrollbar-width: none; min-height: 0; flex: 1 1 auto; }
.pc-tb-scroll::-webkit-scrollbar { display: none; }
.pc-tb-row { display: flex; gap: 2px; margin-bottom: 3px; }
.pc-tb-sep { height: 1px; margin: 4px 8px 7px; background: #1e1e1e; }
.ds-cv.pc-sel { cursor: crosshair; }
.ds-cv-ants { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; z-index: 3; }
.pc-opts { position: absolute; top: 14px; z-index: 27; box-sizing: border-box; display: none; align-items: center; gap: 10px;
  left: calc(var(--ds-inset, 0px) + (100% - var(--ds-inset, 0px) - var(--ds-inset-r, 0px)) / 2); transform: translateX(-50%);
  max-width: calc(100% - var(--ds-inset, 0px) - var(--ds-inset-r, 0px) - 24px); overflow-x: auto; scrollbar-width: none;
  height: 34px; padding: 0 10px; background: #323232; color: #dedede; border: 1px solid #1e1e1e; border-radius: 8px;
  box-shadow: 0 12px 30px rgba(0,0,0,.5); font: 12px/1 Inter, system-ui, sans-serif; white-space: nowrap; user-select: none; }
.pc-opts svg { width: 16px; height: 16px; display: block; }
.pc-o-modes { display: flex; gap: 2px; }
.pc-o-b { all: unset; box-sizing: border-box; width: 24px; height: 24px; border-radius: 3px; color: #c8c8c8; cursor: pointer;
  display: flex; align-items: center; justify-content: center; }
.pc-o-b:hover { background: #424242; color: #fff; }
.pc-o-b.on { background: #4e4e4e; box-shadow: inset 0 0 0 1px #378ef0; color: #fff; }
.pc-o-b.pc-o-ok { color: #6fd18a; }
.pc-o-f { display: flex; align-items: center; gap: 5px; color: #b2b2b2; }
.pc-o-f input { all: unset; box-sizing: border-box; width: 52px; height: 22px; padding: 0 6px; border-radius: 3px; background: #242424;
  border: 1px solid #4a4a4a; color: #dedede; text-align: right; font: inherit; }
.pc-o-f input:focus { border-color: #378ef0; }
.pc-o-c { display: flex; align-items: center; gap: 5px; color: #dedede; cursor: pointer; }
.pc-o-c input { accent-color: #378ef0; margin: 0; }
.pc-o-t { all: unset; padding: 4px 8px; border-radius: 3px; color: #dedede; cursor: pointer; }
.pc-o-t:hover { background: #424242; } .pc-o-t:disabled { color: #6e6e6e; cursor: default; background: none; }
.pc-o-sep { width: 1px; height: 18px; background: #1e1e1e; box-shadow: 1px 0 0 #474747; }
.pc-o-hint, .pc-o-status { color: #9a9a9a; }
.pc-o-status { color: #8fc3ff; }
.pc-o-busy { display: flex; align-items: center; gap: 6px; color: #8fc3ff; }
.pc-o-busy i { width: 10px; height: 10px; border-radius: 50%; border: 2px solid #378ef0; border-right-color: transparent; animation: pc-spin .8s linear infinite; }
@keyframes pc-spin { to { transform: rotate(360deg); } }
.pc-tb.pc-two .pc-tb-sep { width: calc(100% - 16px); }
.pc-btn { all: unset; position: relative; box-sizing: border-box; width: 30px; height: 30px; border-radius: 3px; color: #c8c8c8;
  display: flex; align-items: center; justify-content: center; cursor: pointer; flex: none; }
.pc-btn .pc-ico { width: 20px; height: 20px; }
.pc-btn:hover { background: #424242; color: #dedede; }
.pc-btn.pc-on { background: #4e4e4e; box-shadow: inset 0 0 0 1px #378ef0; color: #fff; }
.pc-btn:focus-visible { outline: 1px solid #378ef0; }
.pc-btn.pc-has-fly::after { content: ""; position: absolute; right: 2px; bottom: 2px; width: 0; height: 0;
  border-left: 4px solid transparent; border-bottom: 4px solid #808080; }
.pc-fly { position: absolute; z-index: 40; min-width: 180px; padding: 4px; background: #323232; border: 1px solid #1e1e1e; border-radius: 6px;
  box-shadow: 0 14px 34px rgba(0,0,0,.55); color: #dedede; font-size: 12.5px; }
.pc-fly-item { all: unset; position: relative; box-sizing: border-box; display: flex; align-items: center; width: 100%; height: 26px;
  padding: 0 8px 0 42px; border-radius: 3px; cursor: pointer; white-space: nowrap; }
.pc-fly-item:hover { background: #424242; }
.pc-fly-item .pc-ico { position: absolute; left: 17px; top: 4px; width: 18px; height: 18px; color: #c8c8c8; }
.pc-fly-item .pc-ico svg { width: 14px; height: 14px; margin: 2px; }
.pc-fly-item .k { margin-left: auto; padding-left: 24px; color: #b2b2b2; font-size: 12px; }
.pc-fly-item.cur::before { content: ""; position: absolute; left: 6px; top: 11px; width: 4px; height: 4px; background: #dedede; }
"""

JS = r"""
// ===================== PhotoCraft tool palette (studio/toolbar.py) =====================
const PC = __PC_DATA__;
const tb = (() => {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem('ds-pc-toolbar') || '{}'); } catch (e) { saved = {}; }
  return {
    tool: PC.tools[saved.tool] ? saved.tool : 'move',
    slotTool: saved.slotTool || {},          // slot index -> last tool shown in that slot
    double: false,                           // one column (the reference also has a two-column view)
  };
})();
const tbSave = () => { try { localStorage.setItem('ds-pc-toolbar', JSON.stringify({ tool: tb.tool, slotTool: tb.slotTool, double: tb.double })); } catch (e) {} };
const icon = (name, cls = 'pc-ico') => `<span class="${cls}">${PC.icons[name] || ''}</span>`;
const tipOf = (id) => { const t = PC.tools[id]; return t.key ? `${t.label}  (${t.key})` : t.label; };
const slots = [];   // flat list: {index, tools}
PC.sections.forEach((sec) => sec.forEach((tools) => slots.push({ index: slots.length, tools })));
const slotOf = (id) => slots.find((x) => x.tools.includes(id));
const slotShown = (s) => (s.tools.includes(tb.tool) ? tb.tool : (s.tools.includes(tb.slotTool[s.index]) ? tb.slotTool[s.index] : s.tools[0]));

const pcRoot = document.createElement('div');
pcRoot.className = 'pc-tb';
root.appendChild(pcRoot);
let flyout = null, pressTimer = null, longPressed = false;
let onToolChange = () => {};   // set by the selection tools

function renderToolbar() {
  pcRoot.className = 'pc-tb ' + (tb.double ? 'pc-two' : 'pc-one');
  let html = `<div class="pc-tb-scroll">`;
  let i = 0;
  PC.sections.forEach((sec, si) => {
    if (si > 0) html += '<div class="pc-tb-sep"></div>';
    const per = tb.double ? 2 : 1;
    for (let r = 0; r < sec.length; r += per) {
      html += '<div class="pc-tb-row">';
      for (let c = r; c < Math.min(r + per, sec.length); c++, i++) {
        const s = slots[i], shown = slotShown(s);
        const on = s.tools.includes(tb.tool);
        html += `<button class="pc-btn${on ? ' pc-on' : ''}${s.tools.length > 1 ? ' pc-has-fly' : ''}" data-slot="${s.index}" title="${tipOf(shown)}">${icon(PC.tools[shown].icon)}</button>`;
      }
      html += '</div>';
    }
  });
  pcRoot.innerHTML = html + '</div>';
  root.classList.toggle('ds-pan', tb.tool === 'move' && !!st.w);
  root.classList.toggle('pc-sel', tb.tool !== 'move');
}

function selectTool(id) {
  if (!PC.tools[id] || id === tb.tool) { renderToolbar(); return; }
  const prev = tb.tool;
  tb.tool = id;
  const s = slotOf(id);
  if (s) tb.slotTool[s.index] = id;
  tbSave(); renderToolbar(); onToolChange(id, prev);
}

function openFlyout(s, btn) {
  closeFlyout();
  const shown = slotShown(s);
  flyout = document.createElement('div');
  flyout.className = 'pc-fly';
  flyout.innerHTML = s.tools.map((id) => {
    const t = PC.tools[id];
    return `<button class="pc-fly-item${id === shown ? ' cur' : ''}" data-tool="${id}">${icon(t.icon)}${t.label}${t.key ? `<span class="k">${t.key}</span>` : ''}</button>`;
  }).join('');
  root.appendChild(flyout);
  // Beside the tool, towards the canvas (the palette sits on the right, so open to its left).
  const br = btn.getBoundingClientRect(), rr = root.getBoundingClientRect();
  flyout.style.top = Math.max(6, Math.min(br.top - rr.top, rr.height - flyout.offsetHeight - 6)) + 'px';
  flyout.style.left = (br.left - rr.left - flyout.offsetWidth - 6) + 'px';
}
function closeFlyout() { if (flyout) { flyout.remove(); flyout = null; } }

pcRoot.addEventListener('pointerdown', (e) => {
  e.stopPropagation();
  const b = e.target.closest('[data-slot]');
  longPressed = false;
  if (b) {
    const s = slots[+b.dataset.slot];
    if (s.tools.length > 1) pressTimer = setTimeout(() => { longPressed = true; openFlyout(s, b); }, 350);  // long-press flyout
  }
});
const clearPress = () => { clearTimeout(pressTimer); pressTimer = null; };
pcRoot.addEventListener('pointerup', clearPress);
pcRoot.addEventListener('pointerleave', clearPress);
pcRoot.addEventListener('wheel', (e) => e.stopPropagation());
pcRoot.addEventListener('dblclick', (e) => e.stopPropagation());
pcRoot.addEventListener('contextmenu', (e) => {
  const b = e.target.closest('[data-slot]');
  e.preventDefault(); e.stopPropagation();
  if (b && slots[+b.dataset.slot].tools.length > 1) openFlyout(slots[+b.dataset.slot], b);
});
pcRoot.addEventListener('click', (e) => {
  e.stopPropagation();
  if (longPressed) { longPressed = false; return; }
  const b = e.target.closest('[data-slot], [data-pc]');
  if (!b) return;
  if (b.dataset.slot !== undefined) { selectTool(slotShown(slots[+b.dataset.slot])); return; }
  if (b.dataset.pc === 'cols') { tb.double = !tb.double; tbSave(); renderToolbar(); relayout(); }
});
root.addEventListener('click', (e) => {
  const item = e.target.closest('.pc-fly-item');
  if (item) { e.stopPropagation(); selectTool(item.dataset.tool); closeFlyout(); return; }
  if (flyout && !e.target.closest('.pc-fly')) closeFlyout();
}, true);
document.addEventListener('pointerdown', (e) => { if (flyout && !flyout.contains(e.target) && !pcRoot.contains(e.target)) closeFlyout(); });

// Tool shortcuts (canvas focused), Photoshop style: the key picks its group's current tool,
// Shift+key cycles through the group.
function toolShortcut(e) {
  if (e.ctrlKey || e.metaKey || e.altKey) return false;
  const k = e.key.length === 1 ? e.key.toUpperCase() : '';
  const ids = Object.keys(PC.tools).filter((id) => PC.tools[id].key === k);
  if (!ids.length) return false;
  const s = slotOf(ids[0]);
  if (e.shiftKey && s && s.tools.includes(tb.tool)) {
    const cyc = s.tools.filter((id) => PC.tools[id].key === k);
    selectTool(cyc[(cyc.indexOf(tb.tool) + 1) % cyc.length]);
  } else selectTool(s ? slotShown(s) : ids[0]);
  return true;
}
function insetR() {
  if (!pcRoot.offsetParent) return 0;
  return pcRoot.offsetWidth + 16 + 12;
}
renderToolbar();
// ===================== end tool palette =====================
"""
