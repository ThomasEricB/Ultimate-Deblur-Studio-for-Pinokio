// ===================== Selection tools (studio/js/selection_tools.js) =====================
// Runs inside the canvas component's closure (studio/canvas.py), after the tool palette.
// The selection is a coverage mask the size of the original image; the model is applied to
// the selected pixels only (Python composites the result over the untouched original).
// Behaviour follows PhotoCraft's tools (marquee, lasso_ui, magnetic_lasso_ui, smartselect_cmds);
// the algorithms are the PhotoCraft port in selection_algo.js.

const ALGO_SRC = __ALGO_SRC__;
const Algo = new Function(ALGO_SRC + '\n;return Algo;')();

const SEL = {
  src: null, srcKey: '', W: 0, H: 0,
  mask: null,                 // Uint8Array W*H, or null (no selection = whole image)
  crop: null,                 // committed crop {x0,y0,x1,y1} in image px
  draft: null,                // crop being edited
  g: null,                    // gesture in progress
  poly: null,                 // polygonal lasso vertices
  mag: null,                  // magnetic lasso state
  busy: 0,
  space: false,
};
const OPT_DEFAULTS = { mode: 'replace', feather: 0, antiAlias: true, tolerance: 32, contiguous: true, quickSize: 30,
  magWidth: 10, magContrast: 10, magFreq: 57, fuzziness: 40 };
const opt = (() => { try { return { ...OPT_DEFAULTS, ...JSON.parse(localStorage.getItem('ds-sel-options') || '{}') }; } catch (e) { return { ...OPT_DEFAULTS }; } })();
const optSave = () => { try { localStorage.setItem('ds-sel-options', JSON.stringify(opt)); } catch (e) {} };

// ---------- worker for the min-cut tools (keeps the page responsive) ----------
let worker = null, workerSrcKey = '', jobId = 0;
const pending = new Map();
try {
  const tail = `\nlet SRC = null;\nonmessage = (e) => { const m = e.data; try {\n  if (m.op === 'source') { SRC = m.src; return; }\n  let r = null;\n  if (m.op === 'object') r = Algo.objectSelect(SRC, m.rect);\n  else if (m.op === 'quick') r = Algo.quickSelect(SRC, m.points, m.size);\n  else if (m.op === 'wand') r = Algo.wandRegion(SRC, m.seed, m.tolerance, m.contiguous, m.antiAlias);\n  else if (m.op === 'range') r = Algo.colorRange(SRC, m.color, m.fuzziness);\n  postMessage({ id: m.id, r }, r ? [r.mask.buffer] : []);\n} catch (err) { postMessage({ id: m.id, error: String(err && err.message || err) }); } };`;
  worker = new Worker(URL.createObjectURL(new Blob([ALGO_SRC, tail], { type: 'text/javascript' })));
  worker.onmessage = (e) => { const p = pending.get(e.data.id); if (!p) return; pending.delete(e.data.id); e.data.error ? p.reject(new Error(e.data.error)) : p.resolve(e.data.r); };
} catch (e) { worker = null; }
function runAlgo(op, args) {
  if (!worker) {   // no worker: run on the page
    const s = SEL.src;
    if (op === 'object') return Promise.resolve(Algo.objectSelect(s, args.rect));
    if (op === 'quick') return Promise.resolve(Algo.quickSelect(s, args.points, args.size));
    if (op === 'wand') return Promise.resolve(Algo.wandRegion(s, args.seed, args.tolerance, args.contiguous, args.antiAlias));
    return Promise.resolve(Algo.colorRange(s, args.color, args.fuzziness));
  }
  if (workerSrcKey !== SEL.srcKey) {
    worker.postMessage({ op: 'source', src: { w: SEL.src.w, h: SEL.src.h, data: SEL.src.data } });
    workerSrcKey = SEL.srcKey;
  }
  const id = ++jobId;
  return new Promise((resolve, reject) => { pending.set(id, { resolve, reject }); worker.postMessage({ op, id, ...args }); });
}

// ---------- the original image (what the selection tools analyse) ----------
const baseUrl = (u) => (u || '').split('?')[0];
function sourceImageEl() {
  const v = V();
  return stage.querySelector(v.mode === 'compare' ? '.ds-cv-before' : '.ds-cv-single');
}
function viewBox() {   // the part of the original the canvas shows (a crop after a cropped run)
  const v = V(), b = v.view_box;
  return b && b.length === 4 ? { x: b[0], y: b[1], w: b[2], h: b[3] } : { x: 0, y: 0, w: SEL.W || st.w, h: SEL.H || st.h };
}
function ensureSource() {
  const v = V(), url = v.orig || (sourceImageEl() && sourceImageEl().src);
  if (!url) return Promise.resolve(null);
  const key = baseUrl(url);
  if (SEL.src && SEL.srcKey === key) return Promise.resolve(SEL.src);
  return new Promise((resolve) => {
    const im = new Image();
    im.onload = () => {
      const c = document.createElement('canvas');
      c.width = im.naturalWidth; c.height = im.naturalHeight;
      const ctx = c.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(im, 0, 0);
      const d = ctx.getImageData(0, 0, c.width, c.height);
      if (SEL.W && (SEL.W !== c.width || SEL.H !== c.height)) resetSelection(false);
      SEL.src = { w: c.width, h: c.height, data: d.data };
      SEL.srcKey = key; SEL.W = c.width; SEL.H = c.height;
      resolve(SEL.src);
    };
    im.onerror = () => resolve(null);
    im.src = url;
  });
}

// ---------- coordinates ----------
// Screen (canvas CSS px) <-> original-image px. The stage shows `viewBox` of the original at the
// stage's own reference size st.w x st.h.
function scale() { const vb = viewBox(); return st.s * (st.w / Math.max(1, vb.w)); }   // screen px per image px
function toImg(cx, cy) { const vb = viewBox(), k = scale(); return [(cx - st.tx) / k + vb.x, (cy - st.ty) / k + vb.y]; }
function toScr(ix, iy) { const vb = viewBox(), k = scale(); return [st.tx + (ix - vb.x) * k, st.ty + (iy - vb.y) * k]; }
const clampI = (p) => [Math.min(SEL.W, Math.max(0, p[0])), Math.min(SEL.H, Math.max(0, p[1]))];

// ---------- overlay: marching ants, previews, crop ----------
const ants = document.createElement('canvas');
ants.className = 'ds-cv-ants';
root.insertBefore(ants, stage.nextSibling);
// The selection outline is traced once per mask change into vector contours (marching
// squares over pixel centres, so diagonals come out as clean 45-degree runs instead of a
// pixel staircase), then stroked every frame with real dashes at screen resolution.
let antsPath = null, antsPhase = 0, antsTimer = null;
function rebuildAntsMask() {
  antsPath = null;
  if (!SEL.mask) return;
  const W = SEL.W, H = SEL.H, m = SEL.mask, CW = W + 2;
  const on = (cx, cy) => (cx >= 1 && cy >= 1 && cx <= W && cy <= H && m[(cy - 1) * W + cx - 1] >= 128) ? 1 : 0;
  const nb = new Map();
  const link = (e1, e2) => {
    let a = nb.get(e1); if (!a) nb.set(e1, (a = [])); a.push(e2);
    let b = nb.get(e2); if (!b) nb.set(e2, (b = [])); b.push(e1);
  };
  // Edge ids: horizontal edge from corner (cx,cy) rightwards = idx*2, vertical downwards = idx*2+1.
  const hE = (cx, cy) => (cy * CW + cx) * 2, vE = (cx, cy) => (cy * CW + cx) * 2 + 1;
  // Only rows/columns near selected pixels can hold a boundary.
  let x0 = W, y0 = H, x1 = -1, y1 = -1;
  for (let y = 0; y < H; y++) for (let x = 0, o = y * W; x < W; x++) if (m[o + x] >= 128) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  if (x1 < 0) return;
  for (let cy = y0; cy <= y1 + 1; cy++) {
    for (let cx = x0; cx <= x1 + 1; cx++) {
      const a = on(cx, cy), b = on(cx + 1, cy), c = on(cx + 1, cy + 1), d = on(cx, cy + 1);
      const k = a * 8 + b * 4 + c * 2 + d;
      if (k === 0 || k === 15) continue;
      const T = hE(cx, cy), R = vE(cx + 1, cy), B = hE(cx, cy + 1), L = vE(cx, cy);
      switch (k) {
        case 1: case 14: link(L, B); break;
        case 2: case 13: link(B, R); break;
        case 3: case 12: link(L, R); break;
        case 4: case 11: link(T, R); break;
        case 6: case 9: link(T, B); break;
        case 7: case 8: link(T, L); break;
        case 5: link(T, R); link(L, B); break;     // saddles: keep the diagonal pixels apart
        case 10: link(T, L); link(B, R); break;
      }
    }
  }
  const pt = (e) => { const i = e >> 1, cx = i % CW, cy = (i - cx) / CW; return (e & 1) ? [cx - 0.5, cy] : [cx, cy - 0.5]; };
  const path = new Path2D(), seen = new Set();
  for (const start of nb.keys()) {
    if (seen.has(start)) continue;
    let prev = -1, cur = start, pts = [];
    while (cur !== undefined && !seen.has(cur)) {
      seen.add(cur);
      const p = pt(cur), n = pts.length;
      // merge collinear runs so long straight edges are single segments
      if (n >= 2 && (pts[n - 1][0] - pts[n - 2][0]) * (p[1] - pts[n - 1][1]) === (pts[n - 1][1] - pts[n - 2][1]) * (p[0] - pts[n - 1][0])) pts[n - 1] = p;
      else pts.push(p);
      const ns = nb.get(cur), next = ns[0] !== prev ? ns[0] : ns[1];
      prev = cur; cur = next;
    }
    if (pts.length < 2) continue;
    path.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) path.lineTo(pts[i][0], pts[i][1]);
    path.closePath();
  }
  antsPath = path;
}
function drawOverlay() {
  const w = root.clientWidth, h = root.clientHeight, dpr = window.devicePixelRatio || 1;
  if (ants.width !== Math.round(w * dpr) || ants.height !== Math.round(h * dpr)) { ants.width = Math.round(w * dpr); ants.height = Math.round(h * dpr); }
  const ctx = ants.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, ants.width, ants.height);
  if (!SEL.W || !st.w) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  // committed crop: dim what will be cut off
  const showCrop = tb.tool === 'crop' ? SEL.draft : SEL.crop;
  if (showCrop) {
    const [x0, y0] = toScr(showCrop.x0, showCrop.y0), [x1, y1] = toScr(showCrop.x1, showCrop.y1);
    ctx.fillStyle = tb.tool === 'crop' ? 'rgba(0,0,0,.55)' : 'rgba(0,0,0,.4)';
    ctx.beginPath(); ctx.rect(0, 0, w, h); ctx.rect(x0, y0, x1 - x0, y1 - y0); ctx.fill('evenodd');
    ctx.strokeStyle = 'rgba(255,255,255,.9)'; ctx.lineWidth = 1; ctx.strokeRect(x0 + 0.5, y0 + 0.5, x1 - x0 - 1, y1 - y0 - 1);
    if (tb.tool === 'crop') {
      ctx.strokeStyle = 'rgba(255,255,255,.35)';
      for (let k = 1; k < 3; k++) {   // rule of thirds
        const gx = x0 + ((x1 - x0) * k) / 3, gy = y0 + ((y1 - y0) * k) / 3;
        ctx.beginPath(); ctx.moveTo(gx, y0); ctx.lineTo(gx, y1); ctx.moveTo(x0, gy); ctx.lineTo(x1, gy); ctx.stroke();
      }
      ctx.fillStyle = '#fff';
      for (const [hx, hy] of cropHandles(showCrop)) ctx.fillRect(hx - 4, hy - 4, 8, 8);
    }
  }
  // marching ants: white line under black dashes, marching with antsPhase
  if (SEL.mask && antsPath) {
    const vb = viewBox(), k = scale();
    ctx.save();
    ctx.setTransform(dpr * k, 0, 0, dpr * k, dpr * (st.tx - vb.x * k), dpr * (st.ty - vb.y * k));
    ctx.lineWidth = 1 / k; ctx.lineJoin = 'miter';
    ctx.strokeStyle = '#fff'; ctx.setLineDash([]); ctx.stroke(antsPath);
    ctx.strokeStyle = '#000'; ctx.setLineDash([4 / k, 4 / k]); ctx.lineDashOffset = -antsPhase / k; ctx.stroke(antsPath);
    ctx.restore();
  }
  // gesture previews
  const dash = (draw) => {
    ctx.save(); ctx.lineWidth = 1;
    ctx.strokeStyle = '#000'; ctx.setLineDash([4, 4]); ctx.lineDashOffset = -antsPhase; draw(); ctx.stroke();
    ctx.strokeStyle = '#fff'; ctx.lineDashOffset = -antsPhase + 4; draw(); ctx.stroke(); ctx.restore();
  };
  const g = SEL.g;
  if (g && (g.kind === 'marquee' || g.kind === 'object') && g.rect) {
    const r = g.rect, [x0, y0] = toScr(r.x0, r.y0), [x1, y1] = toScr(r.x1, r.y1);
    dash(() => { ctx.beginPath(); if (g.ellipse) ctx.ellipse((x0 + x1) / 2, (y0 + y1) / 2, Math.abs(x1 - x0) / 2, Math.abs(y1 - y0) / 2, 0, 0, Math.PI * 2); else ctx.rect(x0 + 0.5, y0 + 0.5, x1 - x0, y1 - y0); });
  }
  const polyline = (pts, close) => { ctx.beginPath(); pts.forEach((p, i) => { const [x, y] = toScr(p[0], p[1]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); if (close) ctx.closePath(); };
  if (g && g.kind === 'lasso' && g.pts.length > 1) dash(() => polyline(g.pts, false));
  if (SEL.poly) dash(() => polyline(SEL.poly.cursor ? [...SEL.poly.pts, SEL.poly.cursor] : SEL.poly.pts, false));
  if (SEL.mag) {
    const m = SEL.mag;
    dash(() => polyline([...m.path, ...m.live.slice(1)], false));
    ctx.fillStyle = '#fff'; ctx.strokeStyle = '#000';
    for (const i of m.anchors) { const [x, y] = toScr(m.path[i][0], m.path[i][1]); ctx.fillRect(x - 2.5, y - 2.5, 5, 5); ctx.strokeRect(x - 2.5, y - 2.5, 5, 5); }
  }
  if (g && g.kind === 'quick') {
    ctx.save(); ctx.strokeStyle = g.subtract ? 'rgba(255,90,90,.9)' : 'rgba(120,180,255,.9)'; ctx.lineWidth = Math.max(1, opt.quickSize * scale()); ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.globalAlpha = 0.35;
    polyline(g.pts, false); ctx.stroke(); ctx.restore();
  }
  // Quick Selection brush cursor
  if (tb.tool === 'quick-selection' && cursorPos && !SEL.busy) {
    ctx.save(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.beginPath();
    ctx.arc(cursorPos[0], cursorPos[1], Math.max(2, (opt.quickSize / 2) * scale()), 0, Math.PI * 2); ctx.stroke();
    ctx.strokeStyle = '#000'; ctx.beginPath(); ctx.arc(cursorPos[0], cursorPos[1], Math.max(2, (opt.quickSize / 2) * scale()) + 1, 0, Math.PI * 2); ctx.stroke(); ctx.restore();
  }
}
function animate() {
  const want = !!(SEL.mask || SEL.g || SEL.poly || SEL.mag);
  if (want && !antsTimer) antsTimer = setInterval(() => { antsPhase = (antsPhase + 1) % 8; drawOverlay(); }, 110);
  if (!want && antsTimer) { clearInterval(antsTimer); antsTimer = null; }
  drawOverlay();
}

// ---------- applying a new region ----------
function modeFor(e, fallback) {
  if (e.shiftKey && e.altKey) return 'intersect';
  if (e.shiftKey) return 'add';
  if (e.altKey) return 'subtract';
  return fallback || opt.mode;
}
function applyRegion(region, mode, label) {
  if (!SEL.W) return;
  if (region && opt.feather > 0 && label !== 'wand' && label !== 'quick' && label !== 'object') region = Algo.feather(region, opt.feather, SEL.W, SEL.H);
  if (!region && mode === 'replace') { setMask(null); return; }
  setMask(Algo.combine(SEL.mask, region, mode, SEL.W, SEL.H));
}
function setMask(m) {
  SEL.mask = m;
  rebuildAntsMask();
  renderOptions(); animate(); scheduleSync();
}
function resetSelection(sync = true) {
  SEL.mask = null; SEL.crop = null; SEL.draft = null; SEL.g = null; SEL.poly = null; SEL.mag = null;
  if (!SEL.mask) antsPath = null;
  if (sync) { scheduleSync(); trigger('crop', { crop: null }); }
  renderOptions(); animate();
}
function selectAll() { if (!SEL.W) return; setMask(new Uint8Array(SEL.W * SEL.H).fill(255)); }
function invertSel() {
  if (!SEL.W) return;
  const n = SEL.W * SEL.H, out = new Uint8Array(n);
  if (!SEL.mask) out.fill(255); else for (let i = 0; i < n; i++) out[i] = 255 - SEL.mask[i];
  setMask(out.some((v) => v) ? out : null);
}

// ---------- backend sync (mask PNG -> Python) ----------
let syncTimer = null, syncSeq = 0;
function scheduleSync() { clearTimeout(syncTimer); syncTimer = setTimeout(syncNow, 350); }
// The mask PNG is encoded from the mask bytes directly, never through a canvas: browsers with
// canvas fingerprint protection add random 1/255 noise on readback, which turned unselected
// pixels into "selected" ones and blew the selection's bounding box up to the whole image.
const CRC_T = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
function crc32(b) { let c = 0xffffffff; for (let i = 0; i < b.length; i++) c = CRC_T[(c ^ b[i]) & 255] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; }
function pngChunk(type, data) {
  const out = new Uint8Array(12 + data.length), dv = new DataView(out.buffer);
  dv.setUint32(0, data.length); for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8); dv.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}
async function maskPng(m, W, H) {
  const raw = new Uint8Array((W + 1) * H);  // 8-bit greyscale, filter type 0 per row
  for (let y = 0; y < H; y++) raw.set(m.subarray(y * W, y * W + W), y * (W + 1) + 1);
  const z = new Uint8Array(await new Response(new Blob([raw]).stream().pipeThrough(new CompressionStream('deflate'))).arrayBuffer());
  const ihdr = new Uint8Array(13), dv = new DataView(ihdr.buffer);
  dv.setUint32(0, W); dv.setUint32(4, H); ihdr[8] = 8; ihdr[9] = 0;
  const sig = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
  return new Blob([sig, pngChunk('IHDR', ihdr), pngChunk('IDAT', z), pngChunk('IEND', new Uint8Array(0))], { type: 'image/png' });
}

async function syncNow() {
  const seq = ++syncSeq;
  if (!SEL.mask) { trigger('selection', { path: null }); renderOptions(); return; }
  let x0 = SEL.W, y0 = SEL.H, x1 = 0, y1 = 0;
  const m = SEL.mask, W = SEL.W;
  for (let i = 0; i < m.length; i++) if (m[i]) { const x = i % W, y = (i / W) | 0; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  const blob = await maskPng(m, SEL.W, SEL.H);
  if (seq !== syncSeq || !blob) return;
  try {
    const r = await upload(new File([blob], 'selection.png', { type: 'image/png' }));
    if (seq !== syncSeq) return;
    trigger('selection', { path: r.path, w: SEL.W, h: SEL.H, bbox: [x0, y0, x1 + 1, y1 + 1] });
  } catch (e) { flash('Could not send the selection to the server'); }
}

// ---------- crop ----------
function cropHandles(r) {
  const [x0, y0] = toScr(r.x0, r.y0), [x1, y1] = toScr(r.x1, r.y1), mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
  return [[x0, y0], [mx, y0], [x1, y0], [x1, my], [x1, y1], [mx, y1], [x0, y1], [x0, my]];
}
function cropHit(r, cx, cy) {
  const hs = cropHandles(r);
  for (let i = 0; i < hs.length; i++) if (Math.abs(hs[i][0] - cx) <= 7 && Math.abs(hs[i][1] - cy) <= 7) return i;
  const [x0, y0] = toScr(r.x0, r.y0), [x1, y1] = toScr(r.x1, r.y1);
  return cx > x0 && cx < x1 && cy > y0 && cy < y1 ? 'move' : null;
}
const normRect = (r) => ({ x0: Math.min(r.x0, r.x1), y0: Math.min(r.y0, r.y1), x1: Math.max(r.x0, r.x1), y1: Math.max(r.y0, r.y1) });
function snapRect(r) {   // marquees and crops snap to whole pixels inside the image
  const n = normRect(r);
  return { x0: Math.max(0, Math.round(n.x0)), y0: Math.max(0, Math.round(n.y0)), x1: Math.min(SEL.W, Math.round(n.x1)), y1: Math.min(SEL.H, Math.round(n.y1)) };
}
function commitCrop() {
  if (!SEL.draft) return;
  const r = snapRect(SEL.draft);
  if (r.x1 - r.x0 < 2 || r.y1 - r.y0 < 2) { SEL.draft = null; renderOptions(); animate(); return; }
  const full = r.x0 === 0 && r.y0 === 0 && r.x1 === SEL.W && r.y1 === SEL.H;
  SEL.crop = full ? null : r; SEL.draft = SEL.crop ? { ...SEL.crop } : null;
  trigger('crop', { crop: SEL.crop ? [r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0] : null });
  flash(SEL.crop ? `Crop ${r.x1 - r.x0}×${r.y1 - r.y0} - only this area is output` : 'Crop cleared');
  renderOptions(); animate();
}
function clearCrop() { SEL.crop = null; SEL.draft = null; trigger('crop', { crop: null }); renderOptions(); animate(); }

// ---------- magnetic lasso (magnetic_lasso_ui.rs) ----------
const magSettings = () => Algo.Tracer.settings(opt.magWidth, opt.magContrast / 100);
const magSpacing = () => (8 + (100 - Math.min(100, Math.max(0, opt.magFreq))) * 0.9) / Math.max(0.01, scale());
function magFollow(q, fasten) {
  const m = SEL.mag, s = magSettings();
  if (!m) return;
  const lastT = m.trail[m.trail.length - 1];
  if (!lastT || Algo.pdist(lastT, q) > 0.5) m.trail.push(q);
  const target = m.tracer.snap(q, s), every = magSpacing();
  for (let k = 0; k < 64; k++) {
    const from = m.path[m.path.length - 1];
    const live = m.tracer.trace(from, target, m.trail, s);
    const c = fasten ? Algo.cutPath(live, every) : null;
    if (!c) { m.live = live; return; }
    const [head, point] = c;
    m.path.push(...head.slice(1)); m.anchors.push(m.path.length - 1);
    let near = 0, best = Infinity;
    m.trail.forEach((p, i) => { const d = Algo.pdist(p, point); if (d < best) { best = d; near = i; } });
    m.trail = [point, ...m.trail.slice(near + 1)];
  }
}
function magFastenLive() {
  const m = SEL.mag;
  if (!m || m.live.length < 2) return;
  m.path.push(...m.live.slice(1)); m.anchors.push(m.path.length - 1);
  m.trail = [m.path[m.path.length - 1]]; m.live = [m.path[m.path.length - 1]];
}
function magClose() {
  const m = SEL.mag;
  if (!m) return;
  magFastenLive();
  const s = magSettings(), last = m.path[m.path.length - 1], first = m.path[0];
  const back = m.tracer.trace(last, first, [last, first], s);
  const pts = [...m.path, ...back.slice(1, -1)];
  SEL.mag = null;
  finishPolygon(pts, m.mode);
}
function finishPolygon(pts, mode) {
  if (pts.length < 3) { animate(); return; }
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of pts) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
  const area = Algo.rIntersect(Algo.rect(Math.floor(x0), Math.floor(y0), Math.ceil(x1) + 1, Math.ceil(y1) + 1), Algo.rect(0, 0, SEL.W, SEL.H));
  if (area.x1 <= area.x0 || area.y1 <= area.y0) { animate(); return; }
  applyRegion(Algo.polygon(pts, area, opt.antiAlias), mode, 'lasso');
}

// ---------- busy work ----------
async function busy(label, fn) {
  SEL.busy++; renderOptions(label);
  try { return await fn(); } catch (e) { console.error(e); flash(`${label} failed: ${e.message || e}`); return undefined; }
  finally { SEL.busy--; renderOptions(); animate(); }
}

// ---------- canvas hooks (called from canvas.py) ----------
let cursorPos = null;
function isSelTool() { return tb.tool !== 'move'; }
function toolPointerDown(e, cx, cy) {
  if (!isSelTool() || SEL.space || !st.w) return false;
  if (SEL.busy) { flash('Still working on the last selection…'); return true; }
  if (!SEL.src) { ensureSource().then(() => { renderOptions(); animate(); }); flash('Reading the image… try again in a moment'); return true; }
  const p = toImg(cx, cy), t = tb.tool;
  root.focus({ preventScroll: true });
  if (t === 'rect-marquee' || t === 'ellipse-marquee') {
    SEL.g = { kind: 'marquee', ellipse: t === 'ellipse-marquee', start: p, mode: modeFor(e), shift0: e.shiftKey, alt0: e.altKey, rect: null };
  } else if (t === 'row-marquee' || t === 'column-marquee') {
    const y = Math.floor(p[1]), x = Math.floor(p[0]);
    const r = t === 'row-marquee' ? Algo.rect(0, y, SEL.W, y + 1) : Algo.rect(x, 0, x + 1, SEL.H);
    applyRegion(Algo.box(Algo.rIntersect(r, Algo.rect(0, 0, SEL.W, SEL.H))), modeFor(e), 'row');
  } else if (t === 'lasso') {
    SEL.g = { kind: 'lasso', pts: [clampI(p)], mode: modeFor(e) };
  } else if (t === 'polygon-lasso') {
    const q = clampI(p);
    if (!SEL.poly) SEL.poly = { pts: [q], mode: modeFor(e), cursor: null };
    else {
      const [fx, fy] = toScr(SEL.poly.pts[0][0], SEL.poly.pts[0][1]);
      if (SEL.poly.pts.length >= 3 && Math.hypot(fx - cx, fy - cy) <= 8) { const pp = SEL.poly; SEL.poly = null; finishPolygon(pp.pts, pp.mode); return true; }
      SEL.poly.pts.push(q);
    }
  } else if (t === 'magnetic-lasso') {
    const q = clampI(p);
    if (!SEL.mag) {
      const tracer = new Algo.Tracer(SEL.src), s0 = tracer.snap(q, magSettings());
      SEL.mag = { tracer, path: [s0], anchors: [0], live: [s0], trail: [s0], mode: modeFor(e) };
    } else {
      const m = SEL.mag, [fx, fy] = toScr(m.path[0][0], m.path[0][1]);
      if (m.anchors.length >= 2 && Math.hypot(fx - cx, fy - cy) <= 8) { magClose(); return true; }
      magFollow(q, false); magFastenLive();
    }
  } else if (t === 'object-selection') {
    SEL.g = { kind: 'object', start: p, mode: modeFor(e), rect: null };
  } else if (t === 'quick-selection') {
    const subtract = e.altKey || opt.mode === 'subtract';
    SEL.g = { kind: 'quick', pts: [clampI(p)], subtract, mode: subtract ? 'subtract' : (opt.mode === 'add' || SEL.mask ? 'add' : 'replace') };
  } else if (t === 'magic-wand') {
    const seed = [Math.floor(p[0]), Math.floor(p[1])], mode = modeFor(e);
    if (seed[0] < 0 || seed[1] < 0 || seed[0] >= SEL.W || seed[1] >= SEL.H) return true;
    busy('Magic Wand', async () => applyRegion(await runAlgo('wand', { seed, tolerance: opt.tolerance, contiguous: opt.contiguous, antiAlias: opt.antiAlias }), mode, 'wand'));
  } else if (t === 'eyedropper') {
    const x = Math.floor(p[0]), y = Math.floor(p[1]), mode = modeFor(e);
    if (x < 0 || y < 0 || x >= SEL.W || y >= SEL.H) return true;
    const o = (y * SEL.W + x) * 4, d = SEL.src.data, color = [d[o], d[o + 1], d[o + 2]];
    const hex = '#' + color.map((v) => v.toString(16).padStart(2, '0')).join('').toUpperCase();
    busy('Color Range', async () => { applyRegion(await runAlgo('range', { color, fuzziness: opt.fuzziness }), mode, 'range'); flash(`Selected colours near ${hex}`); });
  } else if (t === 'crop') {
    const hit = SEL.draft ? cropHit(SEL.draft, cx, cy) : null;
    SEL.g = { kind: 'crop', hit, start: p, orig: SEL.draft ? { ...SEL.draft } : null };
    if (hit === null) SEL.draft = { x0: p[0], y0: p[1], x1: p[0], y1: p[1] };
  }
  animate();
  return true;
}
function toolPointerMove(e, cx, cy) {
  const g = SEL.g;
  if (!g) return;
  const p = toImg(cx, cy);
  if (g.kind === 'marquee' || g.kind === 'object') {
    let [sx, sy] = g.start, dx = p[0] - sx, dy = p[1] - sy;
    if (g.kind === 'marquee' && e.shiftKey && !g.shift0) { const m = Math.max(Math.abs(dx), Math.abs(dy)); dx = Math.sign(dx || 1) * m; dy = Math.sign(dy || 1) * m; }
    const fromCentre = g.kind === 'marquee' && e.altKey && !g.alt0;
    g.rect = fromCentre ? { x0: sx - dx, y0: sy - dy, x1: sx + dx, y1: sy + dy } : { x0: sx, y0: sy, x1: sx + dx, y1: sy + dy };
    g.rect = normRect(g.rect);
  } else if (g.kind === 'lasso' || g.kind === 'quick') {
    const q = clampI(p), last = g.pts[g.pts.length - 1];
    if (Algo.pdist(q, last) * scale() >= 1.5) g.pts.push(q);
  } else if (g.kind === 'crop') {
    const o = g.orig, dx = p[0] - g.start[0], dy = p[1] - g.start[1];
    if (g.hit === null) SEL.draft = { x0: g.start[0], y0: g.start[1], x1: p[0], y1: p[1] };
    else if (g.hit === 'move') {
      const w = o.x1 - o.x0, h = o.y1 - o.y0, x0 = Math.min(SEL.W - w, Math.max(0, o.x0 + dx)), y0 = Math.min(SEL.H - h, Math.max(0, o.y0 + dy));
      SEL.draft = { x0, y0, x1: x0 + w, y1: y0 + h };
    } else {
      const r = { ...o }, i = g.hit;
      if (i === 0 || i === 6 || i === 7) r.x0 = o.x0 + dx;
      if (i === 2 || i === 3 || i === 4) r.x1 = o.x1 + dx;
      if (i === 0 || i === 1 || i === 2) r.y0 = o.y0 + dy;
      if (i === 4 || i === 5 || i === 6) r.y1 = o.y1 + dy;
      SEL.draft = r;
    }
  }
  drawOverlay();
}
function toolPointerUp(e, cx, cy) {
  const g = SEL.g;
  SEL.g = null;
  if (!g) return;
  if (g.kind === 'marquee') {
    const r = g.rect ? snapRect(g.rect) : null;
    if (!r || r.x1 - r.x0 < 1 || r.y1 - r.y0 < 1) {   // a click without a drag deselects (Photoshop)
      if (g.mode === 'replace') setMask(null);
    } else applyRegion(g.ellipse ? Algo.ellipse(r, opt.antiAlias) : Algo.box(r), g.mode, 'marquee');
  } else if (g.kind === 'lasso') {
    if (g.pts.length < 3) { if (g.mode === 'replace') setMask(null); } else finishPolygon(g.pts, g.mode);
  } else if (g.kind === 'object') {
    const r = g.rect ? snapRect(g.rect) : null;
    if (!r || r.x1 - r.x0 < 4 || r.y1 - r.y0 < 4) { flash('Drag a rectangle around the object'); }
    else busy('Object Selection', async () => {
      const region = await runAlgo('object', { rect: Algo.rect(r.x0, r.y0, r.x1, r.y1) });
      if (!region) flash('No object found in that rectangle');
      applyRegion(region, g.mode, 'object');
    });
  } else if (g.kind === 'quick') {
    const pts = g.pts.map((q) => [q[0], q[1]]);
    busy('Quick Selection', async () => applyRegion(await runAlgo('quick', { points: pts, size: opt.quickSize }), g.mode, 'quick'));
  } else if (g.kind === 'crop') {
    if (SEL.draft) { SEL.draft = snapRect(SEL.draft); if (SEL.draft.x1 - SEL.draft.x0 < 2 || SEL.draft.y1 - SEL.draft.y0 < 2) SEL.draft = SEL.crop ? { ...SEL.crop } : null; }
    renderOptions();
  }
  animate();
}
function toolHover(cx, cy) {
  cursorPos = [cx, cy];
  if (SEL.poly) { SEL.poly.cursor = clampI(toImg(cx, cy)); drawOverlay(); }
  else if (SEL.mag && !SEL.g) { magFollow(clampI(toImg(cx, cy)), true); drawOverlay(); }
  else if (tb.tool === 'quick-selection') drawOverlay();
}
function toolLeave() { cursorPos = null; if (SEL.poly) SEL.poly.cursor = null; drawOverlay(); }
function toolDblClick() {
  if (SEL.poly) { const pp = SEL.poly; SEL.poly = null; finishPolygon(pp.pts, pp.mode); return true; }
  if (SEL.mag) { magClose(); return true; }
  return isSelTool();   // other selection tools: don't zoom on double-click
}
function toolKey(e) {
  const mod = e.ctrlKey || e.metaKey, k = e.key.toLowerCase();
  if (mod && k === 'd') { setMask(null); return true; }
  if (mod && k === 'a') { selectAll(); return true; }
  if (mod && e.shiftKey && k === 'i') { invertSel(); return true; }
  if (e.key === 'Escape') {
    if (SEL.poly || SEL.mag || SEL.g) { SEL.poly = null; SEL.mag = null; SEL.g = null; animate(); return true; }
    if (tb.tool === 'crop' && SEL.draft) { SEL.draft = SEL.crop ? { ...SEL.crop } : null; renderOptions(); animate(); return true; }
    return false;
  }
  if (e.key === 'Enter') {
    if (SEL.poly) { const pp = SEL.poly; SEL.poly = null; finishPolygon(pp.pts, pp.mode); return true; }
    if (SEL.mag) { magClose(); return true; }
    if (tb.tool === 'crop' && SEL.draft) { commitCrop(); return true; }
    return false;
  }
  if (e.key === 'Backspace' || e.key === 'Delete') {
    if (SEL.poly) { SEL.poly.pts.pop(); if (!SEL.poly.pts.length) SEL.poly = null; drawOverlay(); return true; }
    if (SEL.mag) {
      const m = SEL.mag;
      if (m.anchors.length <= 1) { SEL.mag = null; animate(); return true; }
      m.anchors.pop();
      m.path = m.path.slice(0, m.anchors[m.anchors.length - 1] + 1);
      m.trail = [m.path[m.path.length - 1]]; m.live = [m.path[m.path.length - 1]];
      drawOverlay(); return true;
    }
    return false;
  }
  if (e.key === '[' || e.key === ']') {
    const d = e.key === ']' ? 1 : -1;
    if (tb.tool === 'quick-selection') { opt.quickSize = Math.max(1, Math.min(2000, Math.round(opt.quickSize * (d > 0 ? 1.25 : 0.8)))); optSave(); renderOptions(); drawOverlay(); return true; }
    if (tb.tool === 'magnetic-lasso') { opt.magWidth = Math.max(1, Math.min(256, opt.magWidth + d)); optSave(); renderOptions(); return true; }
  }
  return false;
}
window.addEventListener('keydown', (e) => { if (e.code === 'Space' && root.offsetParent && document.activeElement === root) { SEL.space = true; root.classList.add('ds-pan'); } });
window.addEventListener('keyup', (e) => { if (e.code === 'Space') { SEL.space = false; renderToolbar(); } });

// Called by render() when the canvas content changes.
function selOnRender(v, changed) {
  if (v.mode === 'empty' || v.mode === 'batch' || (v.mode === 'image' && changed)) {
    if (SEL.mask || SEL.crop || SEL.src) { SEL.src = null; SEL.srcKey = ''; SEL.W = SEL.H = 0; resetSelection(true); }
  }
  if (v.mode === 'image' || v.mode === 'compare') ensureSource().then(() => { renderOptions(); animate(); });
  else renderOptions();
  if (!SEL.mask) antsPath = null;
  animate();
}
function selOnView() { drawOverlay(); }

// ---------- options bar ----------
const opts = document.createElement('div');
opts.className = 'pc-opts';
root.appendChild(opts);
const MODE_ICONS = {
  replace: '<svg viewBox="0 0 16 16"><rect x="3" y="3" width="10" height="10" fill="currentColor"/></svg>',
  add: '<svg viewBox="0 0 16 16"><rect x="1.5" y="1.5" width="8" height="8" fill="currentColor"/><rect x="6.5" y="6.5" width="8" height="8" fill="currentColor"/></svg>',
  subtract: '<svg viewBox="0 0 16 16"><rect x="1.5" y="1.5" width="8" height="8" fill="currentColor"/><rect x="6.5" y="6.5" width="8" height="8" fill="#323232" stroke="currentColor"/></svg>',
  intersect: '<svg viewBox="0 0 16 16"><rect x="1.5" y="1.5" width="8" height="8" fill="none" stroke="currentColor"/><rect x="6.5" y="6.5" width="8" height="8" fill="none" stroke="currentColor"/><rect x="6.5" y="6.5" width="3" height="3" fill="currentColor"/></svg>',
};
const num = (key, label, min, max, unit = '') => `<label class="pc-o-f">${label}<input type="number" data-opt="${key}" min="${min}" max="${max}" value="${opt[key]}">${unit}</label>`;
const chk = (key, label) => `<label class="pc-o-c"><input type="checkbox" data-opt="${key}" ${opt[key] ? 'checked' : ''}>${label}</label>`;
function renderOptions(busyLabel) {
  const t = tb.tool, parts = [];
  if (t !== 'move' && t !== 'crop') {
    parts.push('<span class="pc-o-modes">' + ['replace', 'add', 'subtract', 'intersect'].map((m) =>
      `<button class="pc-o-b${opt.mode === m ? ' on' : ''}" data-mode="${m}" title="${{ replace: 'New selection', add: 'Add to selection (Shift)', subtract: 'Subtract from selection (Alt)', intersect: 'Intersect with selection (Shift+Alt)' }[m]}">${MODE_ICONS[m]}</button>`).join('') + '</span>');
  }
  if (['rect-marquee', 'ellipse-marquee', 'lasso', 'polygon-lasso', 'magnetic-lasso', 'eyedropper', 'row-marquee', 'column-marquee'].includes(t)) parts.push(num('feather', 'Feather:', 0, 250, 'px'));
  if (['ellipse-marquee', 'lasso', 'polygon-lasso', 'magnetic-lasso', 'magic-wand'].includes(t)) parts.push(chk('antiAlias', 'Anti-alias'));
  if (t === 'magnetic-lasso') parts.push(num('magWidth', 'Width:', 1, 256, 'px'), num('magContrast', 'Contrast:', 1, 100, '%'), num('magFreq', 'Frequency:', 0, 100));
  if (t === 'quick-selection') parts.push(num('quickSize', 'Size:', 1, 2000, 'px'));
  if (t === 'magic-wand') parts.push(num('tolerance', 'Tolerance:', 0, 255), chk('contiguous', 'Contiguous'));
  if (t === 'eyedropper') parts.push(num('fuzziness', 'Fuzziness:', 0, 200), '<span class="pc-o-hint">Click a colour to select every pixel like it</span>');
  if (t === 'object-selection') parts.push('<span class="pc-o-hint">Drag a rectangle around the object</span>');
  if (t === 'crop') {
    const r = SEL.draft ? snapRect(SEL.draft) : null;
    parts.push(r ? `<span class="pc-o-hint">${r.x1 - r.x0} × ${r.y1 - r.y0} px</span>` : '<span class="pc-o-hint">Drag to choose the area to output</span>');
    if (SEL.draft) parts.push(`<button class="pc-o-b pc-o-ok" data-act="crop-commit" title="Commit crop (Enter)">${PC.icons.check || '✓'}</button><button class="pc-o-b" data-act="crop-cancel" title="Cancel (Esc)">${PC.icons.x || '✕'}</button>`);
    if (SEL.crop) parts.push('<button class="pc-o-t" data-act="crop-clear">Clear crop</button>');
  }
  if (t !== 'crop' || SEL.mask) {
    parts.push('<span class="pc-o-sep"></span><button class="pc-o-t" data-act="all" title="Select All (Ctrl+A)">All</button>'
      + `<button class="pc-o-t" data-act="none" title="Deselect (Ctrl+D)" ${SEL.mask ? '' : 'disabled'}>Deselect</button>`
      + '<button class="pc-o-t" data-act="invert" title="Inverse (Ctrl+Shift+I)">Inverse</button>');
  }
  const label = busyLabel || (SEL.busy ? 'Working…' : '');
  let status = '';
  if (label) status = `<span class="pc-o-busy"><i></i>${label}…</span>`;
  else if (SEL.mask) status = '<span class="pc-o-status">Model applies to the selection only</span>';
  else if (SEL.crop) status = '<span class="pc-o-status">Output is cropped</span>';
  if (status) parts.push(status);
  opts.innerHTML = parts.join('');
  opts.style.display = st.w && (t !== 'move' || SEL.mask || SEL.crop) ? 'flex' : 'none';
}
opts.addEventListener('pointerdown', (e) => e.stopPropagation());
opts.addEventListener('dblclick', (e) => e.stopPropagation());
opts.addEventListener('wheel', (e) => e.stopPropagation());
opts.addEventListener('keydown', (e) => e.stopPropagation());
opts.addEventListener('click', (e) => {
  e.stopPropagation();
  const b = e.target.closest('[data-mode], [data-act]');
  if (!b) return;
  if (b.dataset.mode) { opt.mode = b.dataset.mode; optSave(); renderOptions(); return; }
  const a = b.dataset.act;
  if (a === 'all') selectAll();
  else if (a === 'none') setMask(null);
  else if (a === 'invert') invertSel();
  else if (a === 'crop-commit') commitCrop();
  else if (a === 'crop-cancel') { SEL.draft = SEL.crop ? { ...SEL.crop } : null; renderOptions(); animate(); }
  else if (a === 'crop-clear') clearCrop();
});
opts.addEventListener('change', (e) => {
  const el = e.target, key = el.dataset.opt;
  if (!key) return;
  if (el.type === 'checkbox') opt[key] = el.checked;
  else { const v = parseFloat(el.value); if (Number.isFinite(v)) opt[key] = Math.min(+el.max, Math.max(+el.min, v)); }
  optSave(); renderOptions(); drawOverlay();
});

onToolChange = (id, prev) => {
  // Switching tools drops an unfinished lasso; entering Crop edits the committed crop.
  if (prev === 'polygon-lasso') SEL.poly = null;
  if (prev === 'magnetic-lasso') SEL.mag = null;
  SEL.draft = id === 'crop' && SEL.crop ? { ...SEL.crop } : null;
  renderOptions(); animate();
};
renderOptions();
SEL_READY = true;
root.__dsSel = SEL;   // for diagnostics / automated checks
// ===================== end selection tools =====================
