// Selection algorithms, ported to JavaScript from PhotoCraft (MIT OR Apache-2.0,
// https://github.com/storytold/photocraft, commit 65e8d5f0):
//   crates/algo/src/selection.rs           combine, wand_region, polygon, color_range, antialias
//   crates/algo/src/selection/marquee.rs   elliptical marquee coverage (4x4 sub-samples)
//   crates/algo/src/selection/distance.rs  exact Euclidean distance transform (edt)
//   crates/algo/src/selection_blur.rs      feather (Gaussian, sigma = radius, truncated at 3 sigma)
//   crates/algo/src/segment/mod.rs         RgbImage, grid_cut, components, clean_mask, finish_region
//   crates/algo/src/segment/maxflow.rs     Boykov-Kolmogorov max-flow / min-cut
//   crates/algo/src/segment/gmm.rs         Gaussian mixtures (k-means++, EM, refit)
//   crates/algo/src/segment/grabcut.rs     GrabCut and rectangle Object Selection
//   crates/algo/src/segment/quick.rs       Quick Selection min-cut model (constants fitted to Photoshop)
//   crates/algo/src/magnetic.rs            Magnetic Lasso live wire (Mortensen & Barrett)
//
// Pure functions only (no DOM): this file runs both on the page and in a Web Worker.
// Images are {w, h, data: Uint8ClampedArray RGBA}; rects are {x0, y0, x1, y1} (x1/y1 exclusive);
// selection masks are Uint8Array 0..255 the size of the image; regions are {bbox, mask}.

'use strict';

const Algo = (() => {
  // ------------------------------------------------------------------ rects
  const rect = (x0, y0, x1, y1) => ({ x0, y0, x1, y1 });
  const rw = (r) => Math.max(0, r.x1 - r.x0), rh = (r) => Math.max(0, r.y1 - r.y0);
  const rEmpty = (r) => r.x1 <= r.x0 || r.y1 <= r.y0;
  const rIntersect = (a, b) => rect(Math.max(a.x0, b.x0), Math.max(a.y0, b.y0), Math.min(a.x1, b.x1), Math.min(a.y1, b.y1));
  const rInflate = (r, m) => rect(r.x0 - m, r.y0 - m, r.x1 + m, r.y1 + m);
  const rContains = (r, x, y) => x >= r.x0 && y >= r.y0 && x < r.x1 && y < r.y1;

  // ------------------------------------------------------------------ RGB images (samples 0..1)
  // RGB over r from straight RGBA8, transparent pixels composited over mid gray (segment/mod.rs).
  function rgb(src, r) {
    const w = rw(r), h = rh(r), px = new Float32Array(w * h * 3), d = src.data, sw = src.w;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const sx = r.x0 + x, sy = r.y0 + y, o = (y * w + x) * 3;
        if (sx < 0 || sy < 0 || sx >= src.w || sy >= src.h) { px[o] = px[o + 1] = px[o + 2] = 0.5; continue; }
        const i = (sy * sw + sx) * 4, a = d[i + 3] / 255;
        px[o] = (d[i] / 255) * a + 0.5 * (1 - a);
        px[o + 1] = (d[i + 1] / 255) * a + 0.5 * (1 - a);
        px[o + 2] = (d[i + 2] / 255) * a + 0.5 * (1 - a);
      }
    }
    return { w, h, px };
  }
  // Box-average downsampling by an integer step (output ceil(w/step) x ceil(h/step)).
  function downsample(img, step) {
    if (step <= 1) return img;
    const ow = Math.ceil(img.w / step), oh = Math.ceil(img.h / step), px = new Float32Array(ow * oh * 3);
    for (let oy = 0; oy < oh; oy++) {
      for (let ox = 0; ox < ow; ox++) {
        let a0 = 0, a1 = 0, a2 = 0, n = 0;
        for (let y = oy * step; y < Math.min((oy + 1) * step, img.h); y++) {
          for (let x = ox * step; x < Math.min((ox + 1) * step, img.w); x++) {
            const i = (y * img.w + x) * 3; a0 += img.px[i]; a1 += img.px[i + 1]; a2 += img.px[i + 2]; n++;
          }
        }
        const o = (oy * ow + ox) * 3; px[o] = a0 / n; px[o + 1] = a1 / n; px[o + 2] = a2 / n;
      }
    }
    return { w: ow, h: oh, px };
  }
  const rgbScaled = (src, r, step) => downsample(rgb(src, r), step);
  // Integer working-resolution step so that the area holds at most about maxPx pixels.
  const scaleFor = (r, maxPx) => Math.max(1, Math.ceil(Math.sqrt((rw(r) * rh(r)) / Math.max(1, maxPx))));
  const d2 = (p, i, j) => { const a = p[i] - p[j], b = p[i + 1] - p[j + 1], c = p[i + 2] - p[j + 2]; return a * a + b * b + c * c; };

  // ------------------------------------------------------------------ distance transform (distance.rs)
  const EMPTY_DISTANCE = 1e10;
  function edt(inside, w, h) {
    const n = w * h, out = new Float32Array(n);
    let any = false, all = true;
    for (let i = 0; i < n; i++) { if (inside[i]) any = true; else all = false; }
    if (!any) return out.fill(EMPTY_DISTANCE);
    if (all) return out;
    const NO = 0xffffffff, vertical = new Uint32Array(n).fill(NO);   // column-major
    for (let x = 0; x < w; x++) {
      const col = x * h;
      let nearest = -1;
      for (let y = 0; y < h; y++) { if (inside[y * w + x]) nearest = y; if (nearest >= 0) vertical[col + y] = y - nearest; }
      nearest = -1;
      for (let y = h - 1; y >= 0; y--) { if (vertical[col + y] === 0) nearest = y; if (nearest >= 0) vertical[col + y] = Math.min(vertical[col + y], nearest - y); }
    }
    const costs = new Uint32Array(w), sites = new Int32Array(w), starts = new Float64Array(w);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) costs[x] = vertical[x * h + y];
      let count = 0;
      for (let q = 0; q < w; q++) {
        const dq = costs[q];
        if (dq === NO) continue;
        const fq = dq * dq;
        let start = -Infinity;
        while (count > 0) {
          const p = sites[count - 1], dp = costs[p];
          start = (fq - dp * dp) / (2 * (q - p)) + (q + p) * 0.5;
          if (start > starts[count - 1]) break;
          count--;
        }
        if (count === 0) start = -Infinity;
        sites[count] = q; starts[count] = start; count++;
      }
      const row = y * w;
      if (count === 0) { out.fill(EMPTY_DISTANCE, row, row + w); continue; }
      let k = 0;
      for (let x = 0; x < w; x++) {
        while (k + 1 < count && starts[k + 1] < x) k++;
        const site = sites[k], v = costs[site], dx = x - site;
        out[row + x] = Math.sqrt(dx * dx + v * v);
      }
    }
    return out;
  }

  // ------------------------------------------------------------------ max-flow (maxflow.rs, BK)
  const NONE = -1, TERMINAL = -2, ORPHAN = -3, EPS = 1e-5;
  class Graph {
    constructor(nodes, edges) {
      this.n = nodes;
      this.first = new Int32Array(nodes).fill(NONE);
      this.trCap = new Float64Array(nodes);
      this.parent = new Int32Array(nodes);
      this.isSink = new Uint8Array(nodes);
      this.ts = new Int32Array(nodes);
      this.dist = new Int32Array(nodes);
      this.active = new Uint8Array(nodes);
      const cap = Math.max(16, edges * 2);
      this.head = new Int32Array(cap); this.next = new Int32Array(cap); this.rCap = new Float64Array(cap);
      this.arcs = 0; this.flow = 0;
    }
    grow() {
      const c = this.head.length * 2;
      const h = new Int32Array(c); h.set(this.head); this.head = h;
      const nx = new Int32Array(c); nx.set(this.next); this.next = nx;
      const r = new Float64Array(c); r.set(this.rCap); this.rCap = r;
    }
    addTweights(i, capSource, capSink) {
      const delta = this.trCap[i];
      let s = capSource, t = capSink;
      if (delta > 0) s += delta; else t -= delta;
      this.flow += Math.min(s, t);
      this.trCap[i] = s - t;
    }
    addEdge(i, j, cap, revCap) {
      if (this.arcs + 2 > this.head.length) this.grow();
      const a = this.arcs;
      this.head[a] = j; this.next[a] = this.first[i]; this.rCap[a] = cap; this.first[i] = a;
      this.head[a + 1] = i; this.next[a + 1] = this.first[j]; this.rCap[a + 1] = revCap; this.first[j] = a + 1;
      this.arcs += 2;
    }
    inSource(i) { return this.parent[i] !== NONE && !this.isSink[i]; }
    maxflow() {
      const n = this.n, { parent, isSink, ts, dist, active, first, head, next } = this;
      const rCap = this.rCap, trCap = this.trCap;
      // Deque of active nodes: each node is queued at most once (active flag), so n + 1 slots do.
      const qcap = n + 1, q = new Int32Array(qcap);
      let qh = 0, qt = 0, qlen = 0;
      const pushBack = (v) => { q[qt] = v; qt = (qt + 1) % qcap; qlen++; };
      const pushFront = (v) => { qh = (qh - 1 + qcap) % qcap; q[qh] = v; qlen++; };
      const popFront = () => { const v = q[qh]; qh = (qh + 1) % qcap; qlen--; return v; };
      let orphans = [], oh = 0;
      for (let i = 0; i < n; i++) {
        active[i] = 0; ts[i] = 0;
        if (Math.abs(trCap[i]) > EPS) {
          isSink[i] = trCap[i] < 0 ? 1 : 0; parent[i] = TERMINAL; dist[i] = 1; active[i] = 1; pushBack(i);
        } else parent[i] = NONE;
      }
      let time = 0;
      const augment = (m) => {
        const u = head[m ^ 1], v = head[m];
        let b = rCap[m], x = u;
        for (;;) { const a = parent[x]; if (a === TERMINAL) break; b = Math.min(b, rCap[a ^ 1]); x = head[a]; }
        b = Math.min(b, trCap[x]);
        x = v;
        for (;;) { const a = parent[x]; if (a === TERMINAL) break; b = Math.min(b, rCap[a]); x = head[a]; }
        b = Math.min(b, -trCap[x]);
        rCap[m ^ 1] += b; rCap[m] -= b;
        x = u;
        for (;;) {
          const a = parent[x]; if (a === TERMINAL) break;
          rCap[a] += b; rCap[a ^ 1] -= b;
          const up = head[a];
          if (rCap[a ^ 1] <= EPS) { parent[x] = ORPHAN; orphans.push(x); }
          x = up;
        }
        trCap[x] -= b;
        if (trCap[x] <= EPS) { parent[x] = ORPHAN; orphans.push(x); }
        x = v;
        for (;;) {
          const a = parent[x]; if (a === TERMINAL) break;
          rCap[a ^ 1] += b; rCap[a] -= b;
          const up = head[a];
          if (rCap[a] <= EPS) { parent[x] = ORPHAN; orphans.push(x); }
          x = up;
        }
        trCap[x] += b;
        if (trCap[x] >= -EPS) { parent[x] = ORPHAN; orphans.push(x); }
        this.flow += b;
      };
      const adopt = (i) => {
        const sink = isSink[i];
        let best = NONE, dMin = Infinity;
        for (let a = first[i]; a !== NONE; a = next[a]) {
          const cap = sink ? rCap[a] : rCap[a ^ 1], j = head[a];
          if (cap > EPS && isSink[j] === sink && parent[j] !== NONE) {
            let d = 0, k = j;
            for (;;) {
              if (ts[k] === time) { d += dist[k]; break; }
              const pa = parent[k];
              d += 1;
              if (pa === TERMINAL) { ts[k] = time; dist[k] = 1; break; }
              if (pa === ORPHAN) { d = Infinity; break; }
              k = head[pa];
            }
            if (d < Infinity) {
              if (d < dMin) { best = a; dMin = d; }
              let kk = j, dd = d;
              while (ts[kk] !== time) { ts[kk] = time; dist[kk] = dd; dd--; kk = head[parent[kk]]; }
            }
          }
        }
        if (best !== NONE) { parent[i] = best; ts[i] = time; dist[i] = dMin + 1; return; }
        parent[i] = NONE;
        for (let a = first[i]; a !== NONE; a = next[a]) {
          const j = head[a], pj = parent[j];
          if (isSink[j] === sink && pj !== NONE) {
            const cap = sink ? rCap[a] : rCap[a ^ 1];
            if (cap > EPS && !active[j]) { active[j] = 1; pushBack(j); }
            if (pj !== TERMINAL && pj !== ORPHAN && head[pj] === i) { parent[j] = ORPHAN; orphans.push(j); }
          }
        }
      };
      while (qlen > 0) {
        const i = popFront();
        active[i] = 0;
        if (parent[i] === NONE) continue;
        let contact = NONE;
        const sink = isSink[i];
        for (let a = first[i]; a !== NONE; a = next[a]) {
          const j = head[a], cap = sink ? rCap[a ^ 1] : rCap[a];
          if (cap > EPS) {
            if (parent[j] === NONE) {
              isSink[j] = sink; parent[j] = a ^ 1; ts[j] = ts[i]; dist[j] = dist[i] + 1;
              if (!active[j]) { active[j] = 1; pushBack(j); }
            } else if (isSink[j] !== sink) {
              contact = sink ? a ^ 1 : a; break;
            } else if (ts[j] <= ts[i] && dist[j] > dist[i]) {
              parent[j] = a ^ 1; ts[j] = ts[i]; dist[j] = dist[i] + 1;
            }
          }
        }
        if (contact === NONE) continue;
        if (!active[i]) { active[i] = 1; pushFront(i); }
        time++;
        orphans = []; oh = 0;
        augment(contact);
        while (oh < orphans.length) adopt(orphans[oh++]);
      }
      return this.flow;
    }
  }

  // ------------------------------------------------------------------ grid cut (segment/mod.rs)
  const FREE = 0, HARD_FG = 1, HARD_BG = 2;
  const NB = [[1, 0, 1], [0, 1, 1], [1, 1, Math.SQRT2], [-1, 1, Math.SQRT2]];
  function contrastBeta(img) {
    const { w, h, px } = img;
    let sum = 0, n = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const a = (y * w + x) * 3;
      for (const [dx, dy] of NB) {
        const nx = x + dx, ny = y + dy;
        if (nx >= 0 && nx < w && ny < h) { sum += d2(px, a, (ny * w + nx) * 3); n++; }
      }
    }
    return n === 0 || sum <= 1e-12 ? 0 : n / (2 * sum);
  }
  // Binary labelling by one min cut: data costs + contrast-sensitive smoothness over the
  // 8-connected grid; only free pixels become graph nodes. Returns Uint8Array (1 = foreground).
  function gridCutWith(img, costFg, costBg, fixed, weight) {
    const { w, h, px } = img, n = w * h;
    const idx = new Int32Array(n).fill(-1);
    let count = 0;
    for (let i = 0; i < n; i++) if (fixed[i] === FREE) idx[i] = count++;
    const out = new Uint8Array(n);
    for (let i = 0; i < n; i++) out[i] = fixed[i] === HARD_FG ? 1 : 0;
    if (count === 0) return out;
    const g = new Graph(count, count * 4);
    for (let i = 0; i < n; i++) if (idx[i] >= 0) g.addTweights(idx[i], costBg[i], costFg[i]);
    const fixedLink = (node, f, wt) => { if (f === HARD_FG) g.addTweights(node, wt, 0); else g.addTweights(node, 0, wt); };
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x, a = i * 3;
      for (const [dx, dy, len] of NB) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || nx >= w || ny >= h) continue;
        const j = ny * w + nx;
        const fi = idx[i] >= 0, fj = idx[j] >= 0;
        if (!fi && !fj) continue;
        const wt = weight(d2(px, a, j * 3)) / len;
        if (fi && fj) g.addEdge(idx[i], idx[j], wt, wt);
        else if (fi) fixedLink(idx[i], fixed[j], wt);
        else fixedLink(idx[j], fixed[i], wt);
      }
    }
    g.maxflow();
    for (let i = 0; i < n; i++) if (idx[i] >= 0) out[i] = g.inSource(idx[i]) ? 1 : 0;
    return out;
  }
  const gridCut = (img, cf, cb, fixed, gamma, beta) => gridCutWith(img, cf, cb, fixed, (dd) => gamma * Math.exp(-beta * dd));

  // ------------------------------------------------------------------ components / masks
  function components(mask, w, h, eight) {
    const lab = new Int32Array(w * h), sizes = [0], stack = [];
    for (let s = 0; s < w * h; s++) {
      if (!mask[s] || lab[s] !== 0) continue;
      const l = sizes.length; sizes.push(0); lab[s] = l; stack.push(s);
      while (stack.length) {
        const i = stack.pop(); sizes[l]++;
        const x = i % w, y = (i / w) | 0;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          if (!eight && dx !== 0 && dy !== 0) continue;
          if (dx === 0 && dy === 0) continue;
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          const j = ny * w + nx;
          if (mask[j] && lab[j] === 0) { lab[j] = l; stack.push(j); }
        }
      }
    }
    return [lab, sizes];
  }
  // Keeps components at least `frac` of the largest; fills holes not touching the border smaller
  // than `holeFrac` of the kept area.
  function cleanMask(mask, w, h, frac, holeFrac) {
    const [lab, sizes] = components(mask, w, h, true);
    let largest = 0;
    for (let k = 1; k < sizes.length; k++) largest = Math.max(largest, sizes[k]);
    const out = new Uint8Array(w * h);
    if (largest === 0) return out;
    const min = Math.ceil(largest * frac);
    let area = 0;
    for (let i = 0; i < w * h; i++) { out[i] = lab[i] !== 0 && sizes[lab[i]] >= min ? 1 : 0; area += out[i]; }
    const maxHole = Math.floor(area * holeFrac);
    const inv = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) inv[i] = out[i] ? 0 : 1;
    const [hl, hs] = components(inv, w, h, false);
    const touches = new Uint8Array(hs.length);
    for (let x = 0; x < w; x++) { touches[hl[x]] = 1; touches[hl[(h - 1) * w + x]] = 1; }
    for (let y = 0; y < h; y++) { touches[hl[y * w]] = 1; touches[hl[y * w + w - 1]] = 1; }
    for (let i = 0; i < w * h; i++) if (hl[i] !== 0 && !touches[hl[i]] && hs[hl[i]] <= maxHole) out[i] = 1;
    return out;
  }
  // Edge pixels of a 0/255 mask blend with their 3x3 neighbourhood (the wand's anti-alias rule).
  function antialiasU8(mask, w, h) {
    const src = mask.slice();
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const v = src[y * w + x];
      let sum = 0, edge = false;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const nx = x + dx, ny = y + dy;
        const nv = nx < 0 || ny < 0 || nx >= w || ny >= h ? v : src[ny * w + nx];
        if (nv !== v) edge = true;
        sum += nv;
      }
      if (edge) { const avg = sum / (9 * 255); mask[y * w + x] = Math.round((v > 127 ? 0.5 + 0.5 * avg : 0.5 * avg) * 255); }
    }
  }
  function trimRegion(r) {
    const w = rw(r.bbox);
    if (w === 0) return null;
    const h = (r.mask.length / w) | 0;
    let x0 = Infinity, y0 = Infinity, x1 = 0, y1 = 0;
    for (let y = 0; y < h; y++) {
      let a = -1, b = -1;
      for (let x = 0; x < w; x++) if (r.mask[y * w + x]) { if (a < 0) a = x; b = x; }
      if (a >= 0) { x0 = Math.min(x0, a); x1 = Math.max(x1, b + 1); y0 = Math.min(y0, y); y1 = y + 1; }
    }
    if (x0 === Infinity) return null;
    if (x0 === 0 && y0 === 0 && x1 === w && y1 === h) return r;
    const nw = x1 - x0, mask = new Uint8Array(nw * (y1 - y0));
    for (let y = y0; y < y1; y++) mask.set(r.mask.subarray(y * w + x0, y * w + x1), (y - y0) * nw);
    const b = r.bbox;
    return { bbox: rect(b.x0 + x0, b.y0 + y0, b.x0 + x1, b.y0 + y1), mask };
  }

  // ------------------------------------------------------------------ GMM (gmm.rs)
  const DEFAULT_REG = 2e-5;
  class Rng {   // xorshift64, as PhotoCraft's (deterministic sampling)
    constructor(seed) { this.s = (BigInt.asUintN(64, BigInt(seed) * 0x9E3779B97F4A7C15n)) | 1n; }
    nextU64() { let x = this.s; x ^= BigInt.asUintN(64, x << 13n); x ^= x >> 7n; x ^= BigInt.asUintN(64, x << 17n); this.s = x; return x; }
    f32() { return Number(this.nextU64() >> 40n) / 16777216; }
  }
  function makeComponent(weight, mean, cov, reg) {
    const c = cov.map((r) => r.slice());
    for (let i = 0; i < 3; i++) c[i][i] += reg;
    const det = c[0][0] * (c[1][1] * c[2][2] - c[1][2] * c[2][1]) - c[0][1] * (c[1][0] * c[2][2] - c[1][2] * c[2][0])
      + c[0][2] * (c[1][0] * c[2][1] - c[1][1] * c[2][0]);
    if (!(det > 1e-30) || weight <= 0) return null;
    const inv = [
      (c[1][1] * c[2][2] - c[1][2] * c[2][1]) / det, (c[0][2] * c[2][1] - c[0][1] * c[2][2]) / det, (c[0][1] * c[1][2] - c[0][2] * c[1][1]) / det,
      (c[1][2] * c[2][0] - c[1][0] * c[2][2]) / det, (c[0][0] * c[2][2] - c[0][2] * c[2][0]) / det, (c[0][2] * c[1][0] - c[0][0] * c[1][2]) / det,
      (c[1][0] * c[2][1] - c[1][1] * c[2][0]) / det, (c[0][1] * c[2][0] - c[0][0] * c[2][1]) / det, (c[0][0] * c[1][1] - c[0][1] * c[1][0]) / det,
    ];
    const logNorm = Math.log(weight) - 0.5 * Math.log(det) - 1.5 * Math.log(2 * Math.PI);
    return { weight, mean, inv, logNorm };
  }
  function logWeighted(c, r, g, b) {
    const d0 = r - c.mean[0], d1 = g - c.mean[1], d2v = b - c.mean[2], m = c.inv;
    const q = d0 * (m[0] * d0 + m[1] * d1 + m[2] * d2v) + d1 * (m[3] * d0 + m[4] * d1 + m[5] * d2v) + d2v * (m[6] * d0 + m[7] * d1 + m[8] * d2v);
    return c.logNorm - 0.5 * q;
  }
  class Acc {
    constructor() { this.n = 0; this.s = [0, 0, 0]; this.ss = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]; }
    add(p, o, w) {
      const z = [p[o], p[o + 1], p[o + 2]];
      this.n += w;
      for (let i = 0; i < 3; i++) { this.s[i] += w * z[i]; for (let j = 0; j < 3; j++) this.ss[i][j] += w * z[i] * z[j]; }
    }
    component(total, reg) {
      if (this.n <= 0) return null;
      const m = this.s.map((v) => v / this.n), c = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) c[i][j] = this.ss[i][j] / this.n - m[i] * m[j];
      return makeComponent(this.n / total, m, c, reg);
    }
  }
  class Gmm {
    constructor(comps, reg) { this.comps = comps; this.reg = reg; }
    static fromLabels(samples, labels, k, reg) {   // samples: Float32Array of RGB triples
      const acc = Array.from({ length: k }, () => new Acc()), n = samples.length / 3;
      for (let i = 0; i < n; i++) acc[labels[i]].add(samples, i * 3, 1);
      const comps = acc.map((a) => a.component(n, reg)).filter(Boolean);
      return comps.length ? new Gmm(comps, reg) : null;
    }
    static fit(samples, k, reg, emIters = 3) {
      const n = samples.length / 3;
      if (n === 0) return null;
      k = Math.max(1, Math.min(k, n));
      const centers = kmeans(samples, k, 10), labels = new Int32Array(n);
      for (let i = 0; i < n; i++) labels[i] = nearest(centers, samples, i * 3);
      let g = Gmm.fromLabels(samples, labels, centers.length, reg);
      if (!g) return null;
      for (let it = 0; it < emIters; it++) g = g.emStep(samples) || g;
      return g;
    }
    emStep(samples) {
      const k = this.comps.length, acc = Array.from({ length: k }, () => new Acc()), lw = new Float64Array(k), n = samples.length / 3;
      for (let i = 0; i < n; i++) {
        const o = i * 3;
        let mx = -Infinity;
        for (let c = 0; c < k; c++) { lw[c] = logWeighted(this.comps[c], samples[o], samples[o + 1], samples[o + 2]); mx = Math.max(mx, lw[c]); }
        let sum = 0;
        for (let c = 0; c < k; c++) { lw[c] = Math.exp(lw[c] - mx); sum += lw[c]; }
        for (let c = 0; c < k; c++) { const r = lw[c] / sum; if (r > 1e-6) acc[c].add(samples, o, r); }
      }
      const comps = acc.map((a) => a.component(n, this.reg)).filter(Boolean);
      return comps.length ? new Gmm(comps, this.reg) : null;
    }
    componentOf(r, g, b) {
      let best = 0, bv = -Infinity;
      for (let i = 0; i < this.comps.length; i++) { const v = logWeighted(this.comps[i], r, g, b); if (v > bv) { bv = v; best = i; } }
      return best;
    }
    refit(samples) {
      const n = samples.length / 3, labels = new Int32Array(n);
      for (let i = 0; i < n; i++) labels[i] = this.componentOf(samples[i * 3], samples[i * 3 + 1], samples[i * 3 + 2]);
      return Gmm.fromLabels(samples, labels, this.comps.length, this.reg);
    }
    negLog(r, g, b) {
      let mx = -Infinity;
      const k = this.comps.length, vals = new Float64Array(k);
      for (let i = 0; i < k; i++) { vals[i] = logWeighted(this.comps[i], r, g, b); mx = Math.max(mx, vals[i]); }
      let s = 0;
      for (let i = 0; i < k; i++) s += Math.exp(vals[i] - mx);
      return Math.min(200, Math.max(-50, -(mx + Math.log(s))));
    }
  }
  function nearest(centers, p, o) {
    let best = 0, bd = Infinity;
    for (let i = 0; i < centers.length; i++) {
      const c = centers[i], a = c[0] - p[o], b = c[1] - p[o + 1], d = c[2] - p[o + 2], dd = a * a + b * b + d * d;
      if (dd < bd) { bd = dd; best = i; }
    }
    return best;
  }
  function kmeans(samples, k, iters) {
    const n = samples.length / 3, rng = new Rng(0x5EED + n);
    const at = (i) => [samples[i * 3], samples[i * 3 + 1], samples[i * 3 + 2]];
    let centers = [at(Number(rng.nextU64() % BigInt(n)))];
    const dist = new Float64Array(n);
    const dd = (i, c) => { const a = samples[i * 3] - c[0], b = samples[i * 3 + 1] - c[1], d = samples[i * 3 + 2] - c[2]; return a * a + b * b + d * d; };
    for (let i = 0; i < n; i++) dist[i] = dd(i, centers[0]);
    while (centers.length < k) {
      let total = 0;
      for (let i = 0; i < n; i++) total += dist[i];
      if (total <= 1e-12) break;
      let t = rng.f32() * total, pick = n - 1;
      for (let i = 0; i < n; i++) { t -= dist[i]; if (t <= 0) { pick = i; break; } }
      const c = at(pick);
      centers.push(c);
      for (let i = 0; i < n; i++) dist[i] = Math.min(dist[i], dd(i, c));
    }
    for (let it = 0; it < iters; it++) {
      const sums = centers.map(() => [0, 0, 0, 0]);
      for (let i = 0; i < n; i++) { const c = nearest(centers, samples, i * 3), s = sums[c]; s[0] += samples[i * 3]; s[1] += samples[i * 3 + 1]; s[2] += samples[i * 3 + 2]; s[3]++; }
      const nc = sums.filter((s) => s[3] > 0).map((s) => [s[0] / s[3], s[1] / s[3], s[2] / s[3]]);
      const done = nc.length === centers.length && nc.every((c, i) => c[0] === centers[i][0] && c[1] === centers[i][1] && c[2] === centers[i][2]);
      centers = nc;
      if (done) break;
    }
    return centers;
  }
  // Every k-th sample so that at most `max` remain (samples: Float32Array of triples).
  function subsample(v, max) {
    const n = v.length / 3;
    if (n <= max) return v;
    const k = Math.ceil(n / max), out = new Float32Array(Math.ceil(n / k) * 3);
    let o = 0;
    for (let i = 0; i < n; i += k) { out[o++] = v[i * 3]; out[o++] = v[i * 3 + 1]; out[o++] = v[i * 3 + 2]; }
    return out.subarray(0, o);
  }
  function dataCosts(img, fg, bg) {
    const n = img.w * img.h, cf = new Float32Array(n), cb = new Float32Array(n), p = img.px;
    for (let i = 0; i < n; i++) { const o = i * 3; cf[i] = fg.negLog(p[o], p[o + 1], p[o + 2]); cb[i] = bg.negLog(p[o], p[o + 1], p[o + 2]); }
    return [cf, cb];
  }

  // ------------------------------------------------------------------ GrabCut / Object Selection (grabcut.rs)
  const BG = 0, FG = 1, PR_BG = 2, PR_FG = 3, GAMMA = 50, K = 5, MAX_SAMPLES = 40000;
  const isFg = (t) => t === FG || t === PR_FG;
  function grabcut(img, trimap, iters) {
    const beta = contrastBeta(img), n = img.w * img.h;
    const fixed = new Uint8Array(n);
    for (let i = 0; i < n; i++) fixed[i] = trimap[i] === FG ? HARD_FG : trimap[i] === BG ? HARD_BG : FREE;
    let models = null;
    for (let it = 0; it < Math.max(1, iters); it++) {
      let nf = 0;
      for (let i = 0; i < n; i++) if (isFg(trimap[i])) nf++;
      const fs = new Float32Array(nf * 3), bs = new Float32Array((n - nf) * 3);
      let fo = 0, bo = 0;
      for (let i = 0; i < n; i++) {
        const o = i * 3, dst = isFg(trimap[i]) ? fs : bs;
        const k = dst === fs ? fo : bo;
        dst[k] = img.px[o]; dst[k + 1] = img.px[o + 1]; dst[k + 2] = img.px[o + 2];
        if (dst === fs) fo += 3; else bo += 3;
      }
      const f2 = subsample(fs, MAX_SAMPLES), b2 = subsample(bs, MAX_SAMPLES);
      let fg, bg;
      if (!models) {
        fg = Gmm.fit(f2, K, DEFAULT_REG); bg = Gmm.fit(b2, K, DEFAULT_REG);
        if (!fg || !bg) return null;
      } else {
        fg = models[0].refit(f2) || models[0]; bg = models[1].refit(b2) || models[1];
      }
      const [cf, cb] = dataCosts(img, fg, bg);
      const cut = gridCut(img, cf, cb, fixed, GAMMA, beta);
      let changed = 0;
      for (let i = 0; i < n; i++) {
        if (trimap[i] === PR_BG || trimap[i] === PR_FG) {
          const nt = cut[i] ? PR_FG : PR_BG;
          if (nt !== trimap[i]) changed++;
          trimap[i] = nt;
        }
      }
      models = [fg, bg];
      if (changed * 2000 < n) break;
    }
    return models;
  }
  // Working-resolution labelling -> full-resolution region: the boundary is re-cut at full
  // resolution in a band around it with the colour models, then anti-aliased by one pixel.
  function finishRegion(src, window, step, low, lw, lh, models) {
    const w = rw(window), h = rh(window), mask = new Uint8Array(w * h);
    if (step <= 1) {
      for (let i = 0; i < w * h; i++) mask[i] = low[i] ? 255 : 0;
    } else {
      const band = new Uint8Array(lw * lh);
      for (let y = 0; y < lh; y++) for (let x = 0; x < lw; x++) {
        const v = low[y * lw + x];
        outer: for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx >= 0 && ny >= 0 && nx < lw && ny < lh && low[ny * lw + nx] !== v) { band[y * lw + x] = 1; break outer; }
        }
      }
      for (let y = 0; y < h; y++) {
        const ly = Math.min((y / step) | 0, lh - 1);
        for (let x = 0; x < w; x++) mask[y * w + x] = low[ly * lw + Math.min((x / step) | 0, lw - 1)] ? 255 : 0;
      }
      if (models) refineBand(src, window, step, low, band, lw, lh, models[0], models[1], mask);
    }
    antialiasU8(mask, w, h);
    return trimRegion({ bbox: window, mask });
  }
  function refineBand(src, window, step, low, band, lw, lh, fg, bg, mask) {
    const T = 192, HALO = 12, w = rw(window);
    const block = (x, y) => Math.min(((y - window.y0) / step) | 0, lh - 1) * lw + Math.min(((x - window.x0) / step) | 0, lw - 1);
    for (let ty = window.y0; ty < window.y1; ty += T) {
      for (let tx = window.x0; tx < window.x1; tx += T) {
        const t = rect(tx, ty, Math.min(tx + T, window.x1), Math.min(ty + T, window.y1));
        const bx0 = ((t.x0 - window.x0) / step) | 0, by0 = ((t.y0 - window.y0) / step) | 0;
        const bx1 = Math.min(((t.x1 - 1 - window.x0) / step) | 0, lw - 1), by1 = Math.min(((t.y1 - 1 - window.y0) / step) | 0, lh - 1);
        let any = false;
        for (let by = by0; by <= by1 && !any; by++) for (let bx = bx0; bx <= bx1; bx++) if (band[by * lw + bx]) { any = true; break; }
        if (!any) continue;
        const r = rIntersect(rInflate(t, HALO), window), img = rgb(src, r), rww = rw(r), rhh = rh(r);
        const fixed = new Uint8Array(rww * rhh), cfg = new Float32Array(rww * rhh), cbg = new Float32Array(rww * rhh);
        for (let yy = 0; yy < rhh; yy++) for (let xx = 0; xx < rww; xx++) {
          const i = yy * rww + xx, b = block(r.x0 + xx, r.y0 + yy), lab = low[b];
          if (!band[b]) { fixed[i] = lab ? HARD_FG : HARD_BG; continue; }
          const o = i * 3, p = img.px;
          cfg[i] = fg.negLog(p[o], p[o + 1], p[o + 2]) + (lab ? 0 : 0.4);
          cbg[i] = bg.negLog(p[o], p[o + 1], p[o + 2]) + (lab ? 0.4 : 0);
        }
        const cut = gridCut(img, cfg, cbg, fixed, 50, contrastBeta(img));
        for (let yy = 0; yy < rh(t); yy++) {
          const sy = t.y0 - r.y0 + yy, sx = t.x0 - r.x0, o = (t.y0 - window.y0 + yy) * w + (t.x0 - window.x0);
          for (let xx = 0; xx < rw(t); xx++) mask[o + xx] = cut[sy * rww + sx + xx] ? 255 : 0;
        }
      }
    }
  }
  // Object Selection (rectangle mode): GrabCut from `r` (outside = background, inside = probable
  // foreground) on a working copy of about workPx pixels, re-cut at full resolution.
  function objectSelect(src, r, workPx = 160000) {
    const canvas = rect(0, 0, src.w, src.h);
    r = rIntersect(r, canvas);
    if (rw(r) < 2 || rh(r) < 2) return null;
    const margin = Math.max(8, Math.ceil(Math.max(rw(r), rh(r)) * 0.15));
    const window = rIntersect(rInflate(r, margin), canvas), step = scaleFor(window, workPx);
    const img = rgbScaled(src, window, step), lw = img.w, lh = img.h;
    const trimap = new Uint8Array(lw * lh).fill(BG);
    let anyBg = false;
    for (let y = 0; y < lh; y++) for (let x = 0; x < lw; x++) {
      const cx = window.x0 + x * step + ((step / 2) | 0), cy = window.y0 + y * step + ((step / 2) | 0);
      if (rContains(r, cx, cy)) trimap[y * lw + x] = PR_FG; else anyBg = true;
    }
    if (!anyBg) {
      const ring = Math.max(1, (Math.min(lw, lh) / 50) | 0);
      for (let y = 0; y < lh; y++) for (let x = 0; x < lw; x++) if (x < ring || y < ring || x + ring >= lw || y + ring >= lh) trimap[y * lw + x] = BG;
    }
    const models = grabcut(img, trimap, 8);
    if (!models) return null;
    let low = new Uint8Array(lw * lh);
    for (let i = 0; i < lw * lh; i++) low[i] = isFg(trimap[i]) ? 1 : 0;
    low = cleanMask(low, lw, lh, 0.1, 0.002);
    return finishRegion(src, window, step, low, lw, lh, models);
  }

  // ------------------------------------------------------------------ Quick Selection (quick.rs)
  const TUNING = { area: 0.004079, dist: 0.002651, pow: 0.98885, floor: 0.018847, sigma: 0.048450, k: 1.98549, grow: 1.69371, growFrac: 0.103003, directPx: 300000 };
  const workingStep = (w, h) => (w * h <= 512 * 512 ? 1 : 2);
  function strokeMask(pts, radius, w, h) {
    const m = new Uint8Array(w * h);
    const segs = pts.length === 1 ? [[pts[0], pts[0]]] : pts.slice(1).map((p, i) => [pts[i], p]);
    for (const [a, b] of segs) {
      const x0 = Math.max(0, Math.floor(Math.min(a[0], b[0]) - radius)), y0 = Math.max(0, Math.floor(Math.min(a[1], b[1]) - radius));
      const x1 = Math.min(w, Math.ceil(Math.max(a[0], b[0]) + radius)), y1 = Math.min(h, Math.ceil(Math.max(a[1], b[1]) + radius));
      const dx = b[0] - a[0], dy = b[1] - a[1], len2 = dx * dx + dy * dy;
      for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
        const px = x + 0.5, py = y + 0.5;
        const t = len2 > 0 ? Math.min(1, Math.max(0, ((px - a[0]) * dx + (py - a[1]) * dy) / len2)) : 0;
        const qx = a[0] + t * dx - px, qy = a[1] + t * dy - py;
        if (qx * qx + qy * qy <= radius * radius) m[y * w + x] = 1;
      }
    }
    for (const [x, y] of pts) if (x >= 0 && y >= 0 && x < w && y < h) m[(y | 0) * w + (x | 0)] = 1;
    return m;
  }
  function cutLevels(img, fixed, costFg, weight, scale, directPx) {
    const { w, h } = img, zeros = new Float32Array(w * h);
    if (w * h <= directPx || w < 8 || h < 8) return gridCutWith(img, costFg, zeros, fixed, (dd) => scale * weight(dd));
    const cw = Math.ceil(w / 2), ch = Math.ceil(h / 2), cimg = downsample(img, 2);
    const cfix = new Uint8Array(cw * ch), ccost = new Float32Array(cw * ch);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const c = ((y / 2) | 0) * cw + ((x / 2) | 0);
      ccost[c] += costFg[y * w + x];
      if (fixed[y * w + x] === HARD_FG) cfix[c] = HARD_FG;
    }
    const coarse = cutLevels(cimg, cfix, ccost, weight, scale * 2, directPx);
    const proj = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) proj[i] = coarse[(((i / w) | 0) >> 1) * cw + ((i % w) >> 1)];
    const edge = new Uint8Array(w * h);
    let anyEdge = false;
    for (let i = 0; i < w * h; i++) {
      const x = i % w, y = (i / w) | 0, v = proj[i];
      if ((x > 0 && proj[i - 1] !== v) || (x + 1 < w && proj[i + 1] !== v) || (y > 0 && proj[i - w] !== v) || (y + 1 < h && proj[i + w] !== v)) { edge[i] = 1; anyEdge = true; }
    }
    if (!anyEdge) { for (let i = 0; i < w * h; i++) proj[i] = proj[i] || fixed[i] === HARD_FG ? 1 : 0; return proj; }
    const near = edt(edge, w, h), banded = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) banded[i] = fixed[i] === HARD_FG ? HARD_FG : near[i] <= 3 ? FREE : proj[i] ? HARD_FG : HARD_BG;
    return gridCutWith(img, costFg, zeros, banded, (dd) => scale * weight(dd));
  }
  function selectWorking(img, seeds, radius, t) {
    const { w, h } = img, d = edt(seeds, w, h), grow = Math.max(0, t.grow + t.growFrac * radius);
    const costFg = new Float32Array(w * h), fixed = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) {
      costFg[i] = t.area * (1 + t.dist * Math.pow(Math.max(0, d[i] - grow), t.pow));
      fixed[i] = d[i] <= grow ? HARD_FG : FREE;
    }
    const sigma = Math.max(1e-4, t.sigma);
    const weight = (dd) => t.floor + 1 / (1 + Math.pow(Math.sqrt(dd) / sigma, t.k));
    return cutLevels(img, fixed, costFg, weight, 1, Math.max(1024, t.directPx));
  }
  function upsample(low, lw, lh, step, w, h) {
    const out = new Uint8Array(w * h);
    if (step === 1) { for (let i = 0; i < w * h; i++) out[i] = low[i] ? 255 : 0; return out; }
    const at = (x, y) => (low[Math.min(lh - 1, Math.max(0, y)) * lw + Math.min(lw - 1, Math.max(0, x))] ? 1 : 0);
    for (let y = 0; y < h; y++) {
      const fy = (y + 0.5) / step - 0.5, iy = Math.floor(fy), ty = fy - iy;
      for (let x = 0; x < w; x++) {
        const fx = (x + 0.5) / step - 0.5, ix = Math.floor(fx), tx = fx - ix;
        const v = (at(ix, iy) * (1 - tx) + at(ix + 1, iy) * tx) * (1 - ty) + (at(ix, iy + 1) * (1 - tx) + at(ix + 1, iy + 1) * tx) * ty;
        out[y * w + x] = Math.round(v * 255);
      }
    }
    return out;
  }
  // Grows a selection from a brush stroke (points in image pixels, brush diameter `size`).
  function quickSelect(src, points, size, t = TUNING) {
    const canvas = rect(0, 0, src.w, src.h);
    if (!points.length) return null;
    const r = Math.max(1, size) / 2;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const [x, y] of points) { x0 = Math.min(x0, x - r); y0 = Math.min(y0, y - r); x1 = Math.max(x1, x + r); y1 = Math.max(y1, y + r); }
    const bbox = rIntersect(rect(Math.floor(x0), Math.floor(y0), Math.ceil(x1) + 1, Math.ceil(y1) + 1), canvas);
    if (rEmpty(bbox)) return null;
    const step = workingStep(src.w, src.h), rwork = r / step;
    let margin = Math.trunc(Math.max(48, 2 * rwork)) * step;
    for (let guard = 0; guard < 12; guard++) {
      const snap0 = (v, o) => o + Math.floor((v - o) / step) * step;
      const grown = rIntersect(rInflate(bbox, margin), canvas);
      const window = rect(snap0(grown.x0, 0), snap0(grown.y0, 0), grown.x1, grown.y1);
      const img = rgbScaled(src, window, step), lw = img.w, lh = img.h;
      const pts = points.map(([x, y]) => [(x - window.x0) / step, (y - window.y0) / step]);
      const seeds = strokeMask(pts, Math.max(0.5, rwork), lw, lh);
      if (!seeds.some((v) => v)) return null;
      const inner = [window.x0 > 0, window.y0 > 0, window.x1 < src.w, window.y1 < src.h];
      const low = selectWorking(img, seeds, rwork, t);
      const touches = (side) => {
        if (side === 0) { for (let y = 0; y < lh; y++) if (low[y * lw]) return true; return false; }
        if (side === 1) { for (let x = 0; x < lw; x++) if (low[x]) return true; return false; }
        if (side === 2) { for (let y = 0; y < lh; y++) if (low[y * lw + lw - 1]) return true; return false; }
        for (let x = 0; x < lw; x++) if (low[(lh - 1) * lw + x]) return true; return false;
      };
      if ([0, 1, 2, 3].some((k) => inner[k] && touches(k))) { margin *= 2; continue; }
      const mask = upsample(low, lw, lh, step, rw(window), rh(window));
      return trimRegion({ bbox: window, mask });
    }
    return null;
  }

  // ------------------------------------------------------------------ Magic Wand (selection.rs wand_region)
  function wandRegion(src, seed, tolerance, contiguous, antiAlias) {
    const w = src.w, h = src.h, img = src.data;
    if (seed[0] < 0 || seed[1] < 0 || seed[0] >= w || seed[1] >= h) return null;
    const si = (seed[1] * w + seed[0]) * 4, t0 = img[si], t1 = img[si + 1], t2 = img[si + 2], t3 = img[si + 3];
    const tol = Math.max(0, Math.floor(tolerance + 1e-3));
    const similar = (i) => {
      const o = i * 4;
      if (t3 === 0 && img[o + 3] === 0) return true;
      return Math.abs(img[o] - t0) <= tol && Math.abs(img[o + 1] - t1) <= tol && Math.abs(img[o + 2] - t2) <= tol && Math.abs(img[o + 3] - t3) <= tol;
    };
    const marks = new Uint8Array(w * h);
    let bx0 = Infinity, by0 = Infinity, bx1 = 0, by1 = 0;
    if (contiguous) {
      const stack = [[seed[0], seed[1]]];
      while (stack.length) {
        const [x, y] = stack.pop(), row = y * w;
        if (marks[row + x]) continue;
        let l = x, r = x;
        while (l > 0 && !marks[row + l - 1] && similar(row + l - 1)) l--;
        while (r + 1 < w && !marks[row + r + 1] && similar(row + r + 1)) r++;
        marks.fill(255, row + l, row + r + 1);
        bx0 = Math.min(bx0, l); bx1 = Math.max(bx1, r); by0 = Math.min(by0, y); by1 = Math.max(by1, y);
        for (const ny of [y - 1, y + 1]) {
          if (ny < 0 || ny >= h) continue;
          const nrow = ny * w;
          let inRun = false;
          for (let nx = l; nx <= r; nx++) {
            const ok = !marks[nrow + nx] && similar(nrow + nx);
            if (ok && !inRun) stack.push([nx, ny]);
            inRun = ok;
          }
        }
      }
    } else {
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        if (similar(y * w + x)) { marks[y * w + x] = 255; bx0 = Math.min(bx0, x); bx1 = Math.max(bx1, x); by0 = Math.min(by0, y); by1 = Math.max(by1, y); }
      }
    }
    if (bx0 === Infinity) return null;
    const pad = antiAlias ? 1 : 0;
    const x0 = Math.max(0, bx0 - pad), y0 = Math.max(0, by0 - pad), x1 = Math.min(w, bx1 + 1 + pad), y1 = Math.min(h, by1 + 1 + pad);
    const bw = x1 - x0, mask = new Uint8Array(bw * (y1 - y0));
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
      const v = marks[y * w + x];
      let o = v;
      if (antiAlias) {
        let sum = 0, edge = false;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx, ny = y + dy;
          const nv = nx < 0 || ny < 0 || nx >= w || ny >= h ? v : marks[ny * w + nx];
          if (nv !== v) edge = true;
          sum += nv;
        }
        if (edge) { const avg = sum / (9 * 255); o = Math.round((v > 127 ? 0.5 + 0.5 * avg : 0.5 * avg) * 255); }
      }
      mask[(y - y0) * bw + (x - x0)] = o;
    }
    return { bbox: rect(x0, y0, x1, y1), mask };
  }

  // ------------------------------------------------------------------ Color Range (selection.rs color_range)
  // Coverage falls off linearly with the per-channel colour distance; fuzziness in levels (0..200).
  function colorRange(src, color, fuzziness) {
    const n = src.w * src.h, d = src.data, f = Math.max(0, fuzziness) + 1, mask = new Uint8Array(n);
    let any = false;
    for (let i = 0; i < n; i++) {
      const o = i * 4;
      if (d[o + 3] === 0) continue;
      const dist = Math.max(Math.abs(d[o] - color[0]), Math.abs(d[o + 1] - color[1]), Math.abs(d[o + 2] - color[2]));
      const v = Math.min(1, Math.max(0, 1 - dist / f));
      if (v > 0) { mask[i] = Math.round(v * 255); any = true; }
    }
    return any ? { bbox: rect(0, 0, src.w, src.h), mask } : null;
  }

  // ------------------------------------------------------------------ shapes (selection.rs polygon, marquee.rs)
  // Anti-aliased even-odd polygon coverage over `area`: exact horizontal coverage, 4 sub-rows.
  function polygon(points, area, antiAlias) {
    const w = rw(area), h = rh(area), m = new Float32Array(w * h);
    if (points.length < 3) return null;
    const sub = antiAlias ? 4 : 1, xs = [];
    for (let py = 0; py < h; py++) {
      for (let s = 0; s < sub; s++) {
        const y = area.y0 + py + (s + 0.5) / sub;
        xs.length = 0;
        for (let i = 0; i < points.length; i++) {
          const a = points[i], b = points[(i + 1) % points.length];
          if ((a[1] <= y && b[1] > y) || (b[1] <= y && a[1] > y)) xs.push(a[0] + ((y - a[1]) / (b[1] - a[1])) * (b[0] - a[0]));
        }
        xs.sort((p, q) => p - q);
        for (let k = 0; k + 1 < xs.length; k += 2) {
          const x0 = Math.max(0, xs[k] - area.x0), x1 = Math.min(w, xs[k + 1] - area.x0);
          if (x1 <= x0) continue;
          const c0 = Math.floor(x0), c1 = Math.min(w, Math.ceil(x1));
          for (let cx = c0; cx < c1; cx++) {
            let cov;
            if (antiAlias) cov = Math.min(cx + 1, x1) - Math.max(cx, x0);
            else { const c = cx + 0.5; cov = c >= x0 && c < x1 ? 1 : 0; }
            m[py * w + cx] += Math.max(0, cov) / sub;
          }
        }
      }
    }
    const mask = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) mask[i] = Math.round(Math.min(1, m[i]) * 255);
    return { bbox: area, mask };
  }
  function ellipse(r, antiAlias) {
    const w = rw(r), h = rh(r);
    if (w < 1 || h < 1) return null;
    const cx = (r.x0 + r.x1) / 2, cy = (r.y0 + r.y1) / 2, rx = w / 2, ry = h / 2;
    const inside = (x, y) => { const dx = (x - cx) / rx, dy = (y - cy) / ry; return dx * dx + dy * dy <= 1; };
    const mask = new Uint8Array(w * h);
    for (let y = r.y0; y < r.y1; y++) for (let x = r.x0; x < r.x1; x++) {
      let cov;
      if (!antiAlias) cov = inside(x + 0.5, y + 0.5) ? 16 : 0;
      else if (inside(x, y) && inside(x + 1, y) && inside(x, y + 1) && inside(x + 1, y + 1)) cov = 16;
      else { cov = 0; for (let i = 0; i < 16; i++) if (inside(x + ((i % 4) + 0.5) / 4, y + (((i / 4) | 0) + 0.5) / 4)) cov++; }
      mask[(y - r.y0) * w + (x - r.x0)] = Math.round((cov / 16) * 255);
    }
    return { bbox: r, mask };
  }
  const box = (r) => { const mask = new Uint8Array(rw(r) * rh(r)).fill(255); return rEmpty(r) ? null : { bbox: r, mask }; };

  // Feather: Gaussian blur of a region's coverage, sigma = radius, kernel truncated at 3 sigma,
  // zero beyond the image (selection_blur.rs). Grows the region's box by 3 sigma.
  function feather(region, radius, W, H) {
    if (!region || !(radius >= 0.1)) return region;
    const sigma = Math.min(radius, 250), R = Math.ceil(sigma * 3);
    const b = rIntersect(rInflate(region.bbox, R), rect(0, 0, W, H)), w = rw(b), h = rh(b);
    const src = new Float32Array(w * h), ow = rw(region.bbox);
    for (let y = region.bbox.y0; y < region.bbox.y1; y++) for (let x = region.bbox.x0; x < region.bbox.x1; x++) {
      src[(y - b.y0) * w + (x - b.x0)] = region.mask[(y - region.bbox.y0) * ow + (x - region.bbox.x0)] / 255;
    }
    const k = new Float32Array(2 * R + 1);
    let s = 0;
    for (let i = -R; i <= R; i++) { k[i + R] = Math.exp(-(i * i) / (2 * sigma * sigma)); s += k[i + R]; }
    for (let i = 0; i < k.length; i++) k[i] /= s;
    const tmp = new Float32Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let a = 0;
      for (let i = -R; i <= R; i++) { const xx = x + i; if (xx >= 0 && xx < w) a += src[y * w + xx] * k[i + R]; }
      tmp[y * w + x] = a;
    }
    const mask = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let a = 0;
      for (let i = -R; i <= R; i++) { const yy = y + i; if (yy >= 0 && yy < h) a += tmp[yy * w + x] * k[i + R]; }
      mask[y * w + x] = Math.round(Math.min(1, a) * 255);
    }
    return { bbox: b, mask };
  }

  // ------------------------------------------------------------------ combining (selection.rs combine_region)
  // Combines a region with the current selection (Uint8Array W*H or null) by mode
  // replace|add|subtract|intersect. Returns the new selection or null when nothing is selected.
  function combine(old, region, mode, W, H) {
    const n = W * H;
    const put = (dst, fn) => {
      if (!region) return;
      const b = region.bbox, bw = rw(b);
      for (let y = Math.max(0, b.y0); y < Math.min(H, b.y1); y++) for (let x = Math.max(0, b.x0); x < Math.min(W, b.x1); x++) {
        const i = y * W + x;
        dst[i] = fn(dst[i], region.mask[(y - b.y0) * bw + (x - b.x0)]);
      }
    };
    let out;
    if (mode === 'replace' || (mode === 'add' && !old)) { out = new Uint8Array(n); put(out, (_, v) => v); }
    else if (!old) return null;
    else if (mode === 'add') { out = old.slice(); put(out, (p, v) => Math.max(p, v)); }
    else if (mode === 'subtract') { out = old.slice(); put(out, (p, v) => Math.max(0, p - v)); }
    else {   // intersect: outside the region the new mask is zero
      const nm = new Uint8Array(n); put(nm, (_, v) => v);
      out = new Uint8Array(n);
      for (let i = 0; i < n; i++) out[i] = Math.min(old[i], nm[i]);
    }
    for (let i = 0; i < n; i++) if (out[i]) return out;
    return null;
  }

  // ------------------------------------------------------------------ Magnetic Lasso (magnetic.rs)
  const PIECE = 256, MAX_GUIDE_LENGTH = 262144;
  const W_CENTRE = 0.43, W_STRENGTH = 0.43, W_DIRECTION = 0.14, W_LENGTH = 0.02, W_GUIDE = 0.4, NEUTRAL_DIRECTION = 2 / 3;
  const DIRS = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]];
  const pdist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  function dist2Segment(p, a, b) {
    const vx = b[0] - a[0], vy = b[1] - a[1], l2 = vx * vx + vy * vy;
    const t = l2 > 0 ? Math.min(1, Math.max(0, ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / l2)) : 0;
    const dx = p[0] - (a[0] + t * vx), dy = p[1] - (a[1] + t * vy);
    return dx * dx + dy * dy;
  }
  const plength = (p) => { let s = 0; for (let i = 1; i < p.length; i++) s += pdist(p[i - 1], p[i]); return s; };
  function simplify(p, tol) {   // Ramer-Douglas-Peucker
    const n = p.length;
    if (n < 3) return p.slice();
    const keep = new Uint8Array(n); keep[0] = keep[n - 1] = 1;
    const tol2 = tol * tol, stack = [[0, n - 1]];
    while (stack.length) {
      const [a, b] = stack.pop();
      if (b <= a + 1) continue;
      let far = 0, fk = a;
      for (let k = a + 1; k < b; k++) { const d = dist2Segment(p[k], p[a], p[b]); if (d > far) { far = d; fk = k; } }
      if (far > tol2) { keep[fk] = 1; stack.push([a, fk], [fk, b]); }
    }
    return p.filter((_, i) => keep[i]);
  }
  function split(g, piece) {
    if (!g.length) return [];
    const out = [];
    let cur = [g[0]], acc = 0;
    for (let i = 1; i < g.length; i++) {
      let a = g[i - 1];
      const b = g[i];
      let seg = pdist(a, b);
      while (acc + seg > piece && seg > 0) {
        const t = Math.min(1, Math.max(0, (piece - acc) / seg)), m = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
        cur.push(m); out.push(cur); cur = [m]; a = m; seg = pdist(a, b); acc = 0;
      }
      cur.push(b); acc += seg;
    }
    out.push(cur);
    return out;
  }
  class Tracer {
    constructor(src) { this.src = src; this.bounds = rect(0, 0, src.w, src.h); }
    static settings(width, contrast) {
      return { width: Number.isFinite(width) ? Math.min(256, Math.max(1, width)) : 10, contrast: Number.isFinite(contrast) ? Math.min(1, Math.max(0.01, contrast)) : 0.1 };
    }
    clampPoint(p) { return [Math.min(this.bounds.x1, Math.max(0, p[0])), Math.min(this.bounds.y1, Math.max(0, p[1]))]; }
    pixel(p) { return [Math.min(this.bounds.x1 - 1, Math.max(0, Math.floor(p[0]))), Math.min(this.bounds.y1 - 1, Math.max(0, Math.floor(p[1])))]; }
    field(area) {
      const ext = rIntersect(rInflate(area, 2), this.bounds), ew = rw(ext), eh = rh(ext), d = this.src.data, sw = this.src.w;
      // Premultiplied channels: an opaque shape on transparency has an edge.
      const ch = new Float32Array(ew * eh * 4);
      for (let y = 0; y < eh; y++) for (let x = 0; x < ew; x++) {
        const i = ((ext.y0 + y) * sw + ext.x0 + x) * 4, a = d[i + 3] / 255, o = (y * ew + x) * 4;
        ch[o] = (d[i] / 255) * a; ch[o + 1] = (d[i + 1] / 255) * a; ch[o + 2] = (d[i + 2] / 255) * a; ch[o + 3] = a;
      }
      const at = (x, y, c) => ch[(Math.min(eh - 1, Math.max(0, y - ext.y0)) * ew + Math.min(ew - 1, Math.max(0, x - ext.x0))) * 4 + c];
      const g1 = rIntersect(rInflate(area, 1), ext), gw = rw(g1), st = new Float32Array(gw * rh(g1)), gd = new Float32Array(gw * rh(g1) * 2);
      for (let y = g1.y0; y < g1.y1; y++) for (let x = g1.x0; x < g1.x1; x++) {
        let best = 0, bx = 0, by = 0;
        for (let c = 0; c < 4; c++) {
          const gx = (at(x + 1, y - 1, c) + 2 * at(x + 1, y, c) + at(x + 1, y + 1, c)) - (at(x - 1, y - 1, c) + 2 * at(x - 1, y, c) + at(x - 1, y + 1, c));
          const gy = (at(x - 1, y + 1, c) + 2 * at(x, y + 1, c) + at(x + 1, y + 1, c)) - (at(x - 1, y - 1, c) + 2 * at(x, y - 1, c) + at(x + 1, y - 1, c));
          const m2 = gx * gx + gy * gy;
          if (m2 > best) { best = m2; bx = gx; by = gy; }
        }
        const mag = Math.sqrt(best), i = (y - g1.y0) * gw + (x - g1.x0);
        st[i] = Math.min(1, mag / 4);
        if (mag > 1e-6) { gd[i * 2] = bx / mag; gd[i * 2 + 1] = by / mag; }
      }
      const sAt = (x, y) => (rContains(g1, x, y) ? st[(y - g1.y0) * gw + (x - g1.x0)] : 0);
      const w = rw(area), n = w * rh(area);
      const f = { area, w, strength: new Float32Array(n), grad: new Float32Array(n * 2), ridge: new Uint8Array(n), offset: new Float32Array(n * 2) };
      for (let y = area.y0; y < area.y1; y++) for (let x = area.x0; x < area.x1; x++) {
        const s0 = sAt(x, y), gi = (y - g1.y0) * gw + (x - g1.x0);
        const g = rContains(g1, x, y) ? [gd[gi * 2], gd[gi * 2 + 1]] : [0, 0];
        const i = (y - area.y0) * w + (x - area.x0);
        f.strength[i] = s0; f.grad[i * 2] = g[0]; f.grad[i * 2 + 1] = g[1];
        if (g[0] === 0 && g[1] === 0) continue;
        const qd = (v, o) => (Math.abs(v) >= 0.4142 * Math.abs(o) ? Math.sign(v) : 0);
        const dx = qd(g[0], g[1]), dy = qd(g[1], g[0]);
        const sm = sAt(x - dx, y - dy), sp = sAt(x + dx, y + dy);
        if (s0 > 0 && s0 >= sm && s0 >= sp) {
          f.ridge[i] = 1;
          const denom = sm - 2 * s0 + sp;
          const t = denom < -1e-6 ? Math.min(0.5, Math.max(-0.5, (0.5 * (sm - sp)) / denom)) : 0;
          f.offset[i * 2] = t * dx; f.offset[i * 2 + 1] = t * dy;
        }
      }
      return f;
    }
    static fIndex(f, x, y) { return rContains(f.area, x, y) ? (y - f.area.y0) * f.w + (x - f.area.x0) : -1; }
    static fPoint(f, i, contrast) {
      const x = (i % f.w) + f.area.x0, y = ((i / f.w) | 0) + f.area.y0;
      const onEdge = f.ridge[i] && f.strength[i] >= contrast;
      return [x + 0.5 + (onEdge ? f.offset[i * 2] : 0), y + 0.5 + (onEdge ? f.offset[i * 2 + 1] : 0)];
    }
    // Where a fastening point near p goes: the most prominent edge centre within the width, or p.
    snap(p, s) {
      p = this.clampPoint(p);
      const r = s.width, [px, py] = this.pixel(p), ri = Math.ceil(r) + 1;
      const area = rIntersect(rect(px - ri, py - ri, px + ri + 1, py + ri + 1), this.bounds), f = this.field(area);
      let best = null, bs = -Infinity;
      for (let y = area.y0; y < area.y1; y++) for (let x = area.x0; x < area.x1; x++) {
        const i = Tracer.fIndex(f, x, y);
        if (i < 0 || f.strength[i] < s.contrast || !f.ridge[i]) continue;
        const d = pdist([x + 0.5, y + 0.5], p);
        if (d > r) continue;
        const score = f.strength[i] * (1 - 0.25 * (d / r));
        if (score > bs) { bs = score; best = Tracer.fPoint(f, i, s.contrast); }
      }
      return best || p;
    }
    // The border from `from` to `to` along the edges near `guide` (the pointer's path).
    trace(from, to, guide, s) {
      from = this.clampPoint(from); to = this.clampPoint(to);
      let g = simplify([from, ...guide.map((q) => this.clampPoint(q)), to], 0.5);
      if (plength(g) > MAX_GUIDE_LENGTH) return g;
      const pieces = split(g, PIECE), out = [from];
      let cur = from;
      pieces.forEach((piece, k) => {
        const target = k === pieces.length - 1 ? to : this.snap(piece[piece.length - 1], s);
        out.push(...this.tracePiece(cur, target, piece, s).slice(1));
        cur = target;
      });
      return simplify(out, 0.5);
    }
    tracePiece(a, b, guide, s) {
      const rad = s.width + 1.5, [ax, ay] = this.pixel(a), [bx, by] = this.pixel(b);
      let x0 = Math.min(ax, bx), y0 = Math.min(ay, by), x1 = Math.max(ax, bx) + 1, y1 = Math.max(ay, by) + 1;
      for (const q of guide) { const [x, y] = this.pixel(q); x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x + 1); y1 = Math.max(y1, y + 1); }
      const area = rIntersect(rInflate(rect(x0, y0, x1, y1), Math.ceil(rad) + 1), this.bounds), f = this.field(area);
      const near = corridor(area, guide, rad), start = Tracer.fIndex(f, ax, ay), goal = Tracer.fIndex(f, bx, by);
      if (start < 0 || goal < 0) return [a, b];
      near[start] = 0; near[goal] = 0;
      const cells = search(f, near, rad * rad, start, goal, s.contrast);
      if (!cells || cells.length < 2) return [a, b];
      const pts = cells.map((i) => Tracer.fPoint(f, i, s.contrast));
      pts[0] = a; pts[pts.length - 1] = b;
      return pts;
    }
  }
  function corridor(area, guide, rad) {
    const w = rw(area), near = new Float32Array(w * rh(area)).fill(Infinity), r2 = rad * rad;
    const segs = guide.length === 1 ? [guide[0], guide[0]] : guide;
    for (let k = 1; k < segs.length; k++) {
      const a = segs[k - 1], b = segs[k];
      const r = rIntersect(rect(Math.floor(Math.min(a[0], b[0]) - rad), Math.floor(Math.min(a[1], b[1]) - rad),
        Math.ceil(Math.max(a[0], b[0]) + rad) + 1, Math.ceil(Math.max(a[1], b[1]) + rad) + 1), area);
      for (let y = r.y0; y < r.y1; y++) for (let x = r.x0; x < r.x1; x++) {
        const dd = dist2Segment([x + 0.5, y + 0.5], a, b), i = (y - area.y0) * w + (x - area.x0);
        if (dd <= r2 && dd < near[i]) near[i] = dd;
      }
    }
    return near;
  }
  function directionCost(tpx, tpy, tqx, tqy, dx, dy, len) {
    if ((tpx === 0 && tpy === 0) || (tqx === 0 && tqy === 0)) return NEUTRAL_DIRECTION;
    let lx = dx / len, ly = dy / len;
    if (tpx * lx + tpy * ly < 0) { lx = -lx; ly = -ly; }
    const dp = Math.min(1, Math.max(-1, tpx * lx + tpy * ly)), dq = Math.min(1, Math.max(-1, lx * tqx + ly * tqy));
    return (2 / (3 * Math.PI)) * (Math.acos(dp) + Math.acos(dq));
  }
  function search(f, near, rad2, start, goal, contrast) {
    const w = f.w, h = f.strength.length / w, n = w * h;
    const node = new Float32Array(n).fill(Infinity), tan = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      const dd = near[i];
      if (!Number.isFinite(dd)) continue;
      const s = f.strength[i], onEdge = s >= contrast && f.ridge[i];
      const good = Math.min(1, Math.max(0, (s - contrast) / Math.max(0.02, contrast))) * (0.6 + 0.4 * Math.min(1, s));
      const pull = Math.min(1, Math.sqrt(dd / Math.max(1, rad2)));
      node[i] = W_CENTRE * (onEdge ? 0 : 1) + W_STRENGTH * (1 - good) + W_LENGTH + W_GUIDE * pull;
      if (s >= 0.5 * contrast) { tan[i * 2] = f.grad[i * 2 + 1]; tan[i * 2 + 1] = -f.grad[i * 2]; }
    }
    // Float64 like the heap keys: a float32 copy can round below its own key and look stale.
    const dist = new Float64Array(n).fill(Infinity), from = new Uint8Array(n).fill(255);
    dist[start] = 0;
    // Binary min-heap of (cost, index).
    let hk = new Float64Array(1024), hv = new Int32Array(1024), hn = 0;
    const hpush = (k, v) => {
      if (hn === hk.length) { const a = new Float64Array(hn * 2); a.set(hk); hk = a; const b = new Int32Array(hn * 2); b.set(hv); hv = b; }
      let i = hn++;
      while (i > 0) { const p = (i - 1) >> 1; if (hk[p] <= k) break; hk[i] = hk[p]; hv[i] = hv[p]; i = p; }
      hk[i] = k; hv[i] = v;
    };
    const hpop = () => {
      const k = hk[0], v = hv[0], lk = hk[--hn], lv = hv[hn];
      let i = 0;
      for (;;) {
        let c = 2 * i + 1;
        if (c >= hn) break;
        if (c + 1 < hn && hk[c + 1] < hk[c]) c++;
        if (hk[c] >= lk) break;
        hk[i] = hk[c]; hv[i] = hv[c]; i = c;
      }
      hk[i] = lk; hv[i] = lv;
      return [k, v];
    };
    hpush(0, start);
    while (hn > 0) {
      const [d, i] = hpop();
      if (d > dist[i]) continue;
      if (i === goal) break;
      const x = i % w, y = (i / w) | 0, tpx = tan[i * 2], tpy = tan[i * 2 + 1];
      for (let k = 0; k < 8; k++) {
        const dx = DIRS[k][0], dy = DIRS[k][1], nx = x + dx, ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const j = ny * w + nx, cost = node[j];
        if (!Number.isFinite(cost)) continue;
        const len = dx !== 0 && dy !== 0 ? Math.SQRT2 : 1;
        const nd = d + cost * len + W_DIRECTION * directionCost(tpx, tpy, tan[j * 2], tan[j * 2 + 1], dx, dy, len);
        if (nd < dist[j]) { dist[j] = nd; from[j] = k; hpush(nd, j); }
      }
    }
    if (!Number.isFinite(dist[goal])) return null;
    const cells = [goal];
    let cur = goal;
    while (cur !== start && cells.length <= n) {
      const k = from[cur];
      if (k === 255) return null;
      const x = (cur % w) - DIRS[k][0], y = ((cur / w) | 0) - DIRS[k][1];
      if (x < 0 || y < 0) return null;
      cur = y * w + x;
      cells.push(cur);
    }
    return cells.reverse();
  }
  // The part of polyline p up to `len` along it and that point, when more than a pixel lies beyond
  // (magnetic_lasso_ui.rs `cut`).
  function cutPath(p, len) {
    if (plength(p) < len + 1) return null;
    const head = [];
    let acc = 0;
    for (let i = 1; i < p.length; i++) {
      const a = p[i - 1], b = p[i], d = pdist(a, b);
      head.push(a);
      if (acc + d >= len && d > 0) {
        const t = (len - acc) / d, point = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
        head.push(point);
        return [head, point];
      }
      acc += d;
    }
    return null;
  }

  return {
    rect, rw, rh, rIntersect, rInflate, edt, Graph, gridCut, gridCutWith, Gmm, grabcut, objectSelect,
    quickSelect, wandRegion, colorRange, polygon, ellipse, box, feather, combine, Tracer, simplify, plength, cutPath, pdist,
  };
})();
