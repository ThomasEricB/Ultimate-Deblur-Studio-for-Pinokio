# Selection tool palette: icons and algorithms

Icons copied unmodified from [PhotoCraft](https://github.com/storytold/photocraft) `assets/icons/` at
commit `65e8d5f02049c4a399ac49ea4b7f6faec85f3ebd` (2026-10-11). The floating palette on the canvas
(`studio/toolbar.py`) reproduces the selection part of PhotoCraft's toolbar layout
(`crates/ui-egui/src/panels.rs`, `TOOL_SECTIONS`) and its Pro theme colours (`crates/ui-egui/src/theme.rs`).

| Files | Source | License |
|---|---|---|
| `move`, `square-dashed`, `circle-dashed`, `lasso`, `pentagon`, `square-dashed-mouse-pointer`, `wand-sparkles`, `crop`, `pipette`, `check`, `x` | [Lucide](https://github.com/lucide-icons/lucide) icons (Lucide Icons and Contributors; some derived from Feather by Cole Bemis) | ISC / MIT, [`LICENSE-lucide.txt`](LICENSE-lucide.txt) |
| `marquee-row`, `marquee-column` | Original PhotoCraft glyphs (ArtCraft Team and the PhotoCraft contributors) | MIT (PhotoCraft is MIT OR Apache-2.0), [`LICENSE-photocraft-MIT.txt`](LICENSE-photocraft-MIT.txt) |
| `lasso-magnetic` | Adapted from "Magnetic Lasso Tool" by Lil' Seal (Noun Project), redrawn by the PhotoCraft contributors | CC BY 3.0, [`LICENSE-noun-magnetic-lasso.txt`](LICENSE-noun-magnetic-lasso.txt) |

`studio/js/selection_algo.js` is a JavaScript port of PhotoCraft's selection and segmentation
algorithms (same commit; files listed in its header), used under the MIT license
([`LICENSE-photocraft-MIT.txt`](LICENSE-photocraft-MIT.txt)).
