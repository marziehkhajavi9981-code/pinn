#!/usr/bin/env python3
"""Generate sparse observation masks and train/val/test splits.

The generated artifact is intentionally independent of the current training
loop. Future training code can consume the NPZ file when sparse/masked
supervision is desired, while the existing dense workflow remains unchanged.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
import sys
from typing import Dict, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.grids import crop_stacks  # noqa: E402


MaskDict = Dict[str, np.ndarray]


def _load_cropped_stacks(u_path: str, v_path: str, crop_h: int, crop_w: int) -> Tuple[np.ndarray, np.ndarray]:
    u_data = np.load(u_path)
    v_data = np.load(v_path)
    U = u_data["U"]
    V = v_data["V"]
    Uc, Vc = crop_stacks(U, V, crop_h, crop_w)
    return Uc, Vc


def _target_count(fraction: float, h: int, w: int) -> int:
    return max(1, min(h * w, int(round(float(fraction) * h * w))))


def random_pixels(shape: Tuple[int, int, int], obs_fraction: float, rng: np.random.Generator) -> np.ndarray:
    return rng.random(shape) < obs_fraction


def moving_particles(shape: Tuple[int, int, int], obs_fraction: float, rng: np.random.Generator) -> np.ndarray:
    N, h, w = shape
    n_obs = _target_count(obs_fraction, h, w)
    mask = np.zeros(shape, dtype=bool)
    for k in range(N):
        flat_idx = rng.choice(h * w, size=n_obs, replace=False)
        mask[k].reshape(-1)[flat_idx] = True
    return mask


def fixed_sensors(shape: Tuple[int, int, int], obs_fraction: float, rng: np.random.Generator) -> np.ndarray:
    N, h, w = shape
    n_obs = _target_count(obs_fraction, h, w)
    flat_idx = rng.choice(h * w, size=n_obs, replace=False)
    frame = np.zeros((h, w), dtype=bool)
    frame.reshape(-1)[flat_idx] = True
    return np.broadcast_to(frame, (N, h, w)).copy()


def regular_grid(shape: Tuple[int, int, int], obs_fraction: float, rng: np.random.Generator) -> np.ndarray:
    N, h, w = shape
    step = max(1, int(round(math.sqrt(1.0 / max(obs_fraction, 1e-12)))))
    y0 = int(rng.integers(0, min(step, h)))
    x0 = int(rng.integers(0, min(step, w)))
    frame = np.zeros((h, w), dtype=bool)
    frame[y0::step, x0::step] = True
    return np.broadcast_to(frame, (N, h, w)).copy()


def random_blocks(
    shape: Tuple[int, int, int],
    obs_fraction: float,
    rng: np.random.Generator,
    block_h: int,
    block_w: int,
) -> np.ndarray:
    N, h, w = shape
    mask = np.zeros(shape, dtype=bool)
    target = _target_count(obs_fraction, h, w)
    block_h = max(1, min(block_h, h))
    block_w = max(1, min(block_w, w))
    max_blocks = max(1, math.ceil((target * 3) / (block_h * block_w)))
    for k in range(N):
        attempts = 0
        while int(mask[k].sum()) < target and attempts < max_blocks * 20:
            y0 = int(rng.integers(0, h - block_h + 1))
            x0 = int(rng.integers(0, w - block_w + 1))
            mask[k, y0 : y0 + block_h, x0 : x0 + block_w] = True
            attempts += 1
        if int(mask[k].sum()) > target:
            observed = np.flatnonzero(mask[k].reshape(-1))
            drop = rng.choice(observed, size=int(mask[k].sum()) - target, replace=False)
            mask[k].reshape(-1)[drop] = False
    return mask


def center_block(shape: Tuple[int, int, int], obs_fraction: float) -> np.ndarray:
    N, h, w = shape
    side_scale = math.sqrt(max(0.0, min(obs_fraction, 1.0)))
    bh = max(1, int(round(h * side_scale)))
    bw = max(1, int(round(w * side_scale)))
    y0 = (h - bh) // 2
    x0 = (w - bw) // 2
    frame = np.zeros((h, w), dtype=bool)
    frame[y0 : y0 + bh, x0 : x0 + bw] = True
    return np.broadcast_to(frame, (N, h, w)).copy()


def center_hole(shape: Tuple[int, int, int], obs_fraction: float) -> np.ndarray:
    N, h, w = shape
    missing_fraction = max(0.0, min(1.0, 1.0 - obs_fraction))
    side_scale = math.sqrt(missing_fraction)
    bh = int(round(h * side_scale))
    bw = int(round(w * side_scale))
    frame = np.ones((h, w), dtype=bool)
    if bh > 0 and bw > 0:
        y0 = (h - bh) // 2
        x0 = (w - bw) // 2
        frame[y0 : y0 + bh, x0 : x0 + bw] = False
    return np.broadcast_to(frame, (N, h, w)).copy()


def temporal_stride(shape: Tuple[int, int, int], obs_fraction: float) -> np.ndarray:
    N, h, w = shape
    stride = max(1, int(round(1.0 / max(obs_fraction, 1e-12))))
    mask = np.zeros(shape, dtype=bool)
    mask[::stride, :, :] = True
    return mask


def temporal_windows(shape: Tuple[int, int, int], obs_fraction: float) -> np.ndarray:
    N, h, w = shape
    n_frames = max(1, min(N, int(round(obs_fraction * N))))
    start = max(0, (N - n_frames) // 2)
    mask = np.zeros(shape, dtype=bool)
    mask[start : start + n_frames, :, :] = True
    return mask


def make_observation_mask(args: argparse.Namespace, shape: Tuple[int, int, int]) -> np.ndarray:
    rng = np.random.default_rng(args.seed)
    model = args.mask_model
    if model == "random_pixels":
        mask = random_pixels(shape, args.obs_fraction, rng)
    elif model == "moving_particles":
        mask = moving_particles(shape, args.obs_fraction, rng)
    elif model == "fixed_sensors":
        mask = fixed_sensors(shape, args.obs_fraction, rng)
    elif model == "regular_grid":
        mask = regular_grid(shape, args.obs_fraction, rng)
    elif model == "random_blocks":
        mask = random_blocks(shape, args.obs_fraction, rng, args.block_h, args.block_w)
    elif model == "center_block":
        mask = center_block(shape, args.obs_fraction)
    elif model == "center_hole":
        mask = center_hole(shape, args.obs_fraction)
    elif model == "temporal_stride":
        mask = temporal_stride(shape, args.obs_fraction)
    elif model == "temporal_windows":
        mask = temporal_windows(shape, args.obs_fraction)
    else:
        raise ValueError(f"Unknown mask model: {model}")
    return mask.astype(bool)


def _split_counts(fractions: Tuple[float, float, float], n_items: int) -> Tuple[int, int]:
    train_fraction, val_fraction, _ = fractions
    train_end = int(round(train_fraction * n_items))
    val_end = train_end + int(round(val_fraction * n_items))
    train_end = min(max(train_end, 0), n_items)
    val_end = min(max(val_end, train_end), n_items)
    return train_end, val_end


def split_random_observed(obs_mask: np.ndarray, fractions: Tuple[float, float, float], rng: np.random.Generator) -> MaskDict:
    flat_obs = np.flatnonzero(obs_mask.reshape(-1))
    rng.shuffle(flat_obs)
    train_end, val_end = _split_counts(fractions, len(flat_obs))
    masks = {name: np.zeros_like(obs_mask, dtype=bool) for name in ("train_mask", "val_mask", "test_mask")}
    masks["train_mask"].reshape(-1)[flat_obs[:train_end]] = True
    masks["val_mask"].reshape(-1)[flat_obs[train_end:val_end]] = True
    masks["test_mask"].reshape(-1)[flat_obs[val_end:]] = True
    return masks


def split_time_block(obs_mask: np.ndarray, fractions: Tuple[float, float, float]) -> MaskDict:
    N = obs_mask.shape[0]
    train_end, val_end = _split_counts(fractions, N)
    train = np.zeros_like(obs_mask, dtype=bool)
    val = np.zeros_like(obs_mask, dtype=bool)
    test = np.zeros_like(obs_mask, dtype=bool)
    train[:train_end] = obs_mask[:train_end]
    val[train_end:val_end] = obs_mask[train_end:val_end]
    test[val_end:] = obs_mask[val_end:]
    return {"train_mask": train, "val_mask": val, "test_mask": test}


def split_spatial_block(obs_mask: np.ndarray, fractions: Tuple[float, float, float]) -> MaskDict:
    _, _, w = obs_mask.shape
    train_end, val_end = _split_counts(fractions, w)
    train = np.zeros_like(obs_mask, dtype=bool)
    val = np.zeros_like(obs_mask, dtype=bool)
    test = np.zeros_like(obs_mask, dtype=bool)
    train[:, :, :train_end] = obs_mask[:, :, :train_end]
    val[:, :, train_end:val_end] = obs_mask[:, :, train_end:val_end]
    test[:, :, val_end:] = obs_mask[:, :, val_end:]
    return {"train_mask": train, "val_mask": val, "test_mask": test}


def split_spatiotemporal_block(obs_mask: np.ndarray, fractions: Tuple[float, float, float]) -> MaskDict:
    N, _, w = obs_mask.shape
    t_train_end, t_val_end = _split_counts(fractions, N)
    x_train_end, x_val_end = _split_counts(fractions, w)
    train = np.zeros_like(obs_mask, dtype=bool)
    val = np.zeros_like(obs_mask, dtype=bool)
    test = np.zeros_like(obs_mask, dtype=bool)
    train[:t_train_end, :, :x_train_end] = obs_mask[:t_train_end, :, :x_train_end]
    val[t_train_end:t_val_end, :, x_train_end:x_val_end] = obs_mask[
        t_train_end:t_val_end, :, x_train_end:x_val_end
    ]
    test[t_val_end:, :, x_val_end:] = obs_mask[t_val_end:, :, x_val_end:]
    return {"train_mask": train, "val_mask": val, "test_mask": test}


def make_splits(args: argparse.Namespace, obs_mask: np.ndarray) -> MaskDict:
    fractions = (args.train_fraction, args.val_fraction, args.test_fraction)
    total = sum(fractions)
    if not np.isclose(total, 1.0):
        fractions = tuple(v / total for v in fractions)
    rng = np.random.default_rng(args.seed + 17)
    strategy = args.split_strategy
    if strategy == "random_observed":
        return split_random_observed(obs_mask, fractions, rng)
    if strategy == "time_block":
        return split_time_block(obs_mask, fractions)
    if strategy == "spatial_block":
        return split_spatial_block(obs_mask, fractions)
    if strategy == "spatiotemporal_block":
        return split_spatiotemporal_block(obs_mask, fractions)
    raise ValueError(f"Unknown split strategy: {strategy}")


def make_physics_mask(shape: Tuple[int, int, int], obs_mask: np.ndarray, mode: str) -> np.ndarray:
    physics = np.ones(shape, dtype=bool)
    if shape[1] > 2:
        physics[:, 0, :] = False
        physics[:, -1, :] = False
    if shape[2] > 2:
        physics[:, :, 0] = False
        physics[:, :, -1] = False
    if mode == "interior":
        return physics
    if mode == "observed_interior":
        return physics & obs_mask
    if mode == "none":
        return np.zeros(shape, dtype=bool)
    raise ValueError(f"Unknown physics mask mode: {mode}")


def summarize(mask: np.ndarray) -> Dict[str, float]:
    return {"count": int(mask.sum()), "fraction": float(mask.mean())}


def write_npz(args: argparse.Namespace, shape: Tuple[int, int, int], masks: MaskDict, meta: Dict[str, object]) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        obs_mask=masks["obs_mask"],
        train_mask=masks["train_mask"],
        val_mask=masks["val_mask"],
        test_mask=masks["test_mask"],
        physics_mask=masks["physics_mask"],
        shape=np.asarray(shape, dtype=np.int32),
        meta_json=np.asarray(json.dumps(meta, indent=2)),
    )


def _frame_indices(n_frames: int, max_frames: int) -> np.ndarray:
    if n_frames <= max_frames:
        return np.arange(n_frames, dtype=int)
    return np.unique(np.linspace(0, n_frames - 1, max_frames).round().astype(int))


def _quantize_field(field: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    denom = max(float(vmax) - float(vmin), 1e-12)
    scaled = np.clip((field.astype(np.float32) - float(vmin)) / denom, 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8)


def make_preview_payload(
    masks: MaskDict,
    U: np.ndarray,
    V: np.ndarray,
    max_frames: int,
    include_data: bool,
) -> Dict[str, object]:
    obs = masks["obs_mask"]
    train = masks["train_mask"]
    val = masks["val_mask"]
    test = masks["test_mask"]
    physics = masks["physics_mask"]
    N, h, w = obs.shape
    speed = np.sqrt(U.astype(np.float32) ** 2 + V.astype(np.float32) ** 2)
    ranges = {
        "u": [float(np.nanmin(U)), float(np.nanmax(U))],
        "v": [float(np.nanmin(V)), float(np.nanmax(V))],
        "speed": [float(np.nanmin(speed)), float(np.nanmax(speed))],
    }
    frames = []
    for idx in _frame_indices(N, max_frames):
        label = np.zeros((h, w), dtype=np.uint8)
        label[obs[idx]] = 4
        label[physics[idx] & ~obs[idx]] = 5
        label[train[idx]] = 1
        label[val[idx]] = 2
        label[test[idx]] = 3
        frames.append(
            {
                "index": int(idx),
                "rows": ["".join(str(int(v)) for v in row) for row in label],
                "counts": {
                    "missing": int((~obs[idx]).sum()),
                    "train": int(train[idx].sum()),
                    "val": int(val[idx].sum()),
                    "test": int(test[idx].sum()),
                    "observed": int(obs[idx].sum()),
                    "physics": int(physics[idx].sum()),
                },
            }
        )
        if include_data:
            u_q = _quantize_field(U[idx], ranges["u"][0], ranges["u"][1])
            v_q = _quantize_field(V[idx], ranges["v"][0], ranges["v"][1])
            speed_q = _quantize_field(speed[idx], ranges["speed"][0], ranges["speed"][1])
            frames[-1]["data"] = {
                "u": ["".join(f"{int(v):02x}" for v in row) for row in u_q],
                "v": ["".join(f"{int(v):02x}" for v in row) for row in v_q],
                "speed": ["".join(f"{int(v):02x}" for v in row) for row in speed_q],
            }
    return {
        "n": int(N),
        "h": int(h),
        "w": int(w),
        "includeData": bool(include_data),
        "ranges": ranges,
        "frames": frames,
    }


def write_html_viewer(
    args: argparse.Namespace,
    masks: MaskDict,
    U: np.ndarray,
    V: np.ndarray,
    meta: Dict[str, object],
) -> None:
    if not args.html:
        return
    html_path = Path(args.html)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    payload = make_preview_payload(masks, U, V, args.max_preview_frames, not args.no_data_preview)
    data_json = json.dumps({"payload": payload, "meta": meta})
    title = html.escape(f"Mask Split Viewer: {Path(args.out).name}")
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; background: #101418; color: #eef3f6; }}
    main {{ display: grid; grid-template-columns: 320px 1fr; min-height: 100vh; }}
    aside {{ padding: 18px; border-right: 1px solid #2d3740; background: #151b21; }}
    section {{ padding: 18px; overflow: auto; }}
    canvas {{ image-rendering: pixelated; max-width: 100%; border: 1px solid #34414c; background: #050708; }}
    label {{ display: block; margin: 14px 0 6px; color: #b8c7d1; font-size: 13px; }}
    input, select {{ width: 100%; }}
    pre {{ white-space: pre-wrap; color: #b8c7d1; font-size: 12px; }}
    .legend {{ display: grid; grid-template-columns: 16px 1fr; gap: 8px; align-items: center; margin-top: 16px; }}
    .swatch {{ width: 16px; height: 16px; border: 1px solid #5e6b75; }}
    .stats {{ display: grid; grid-template-columns: 1fr auto; gap: 6px 16px; margin-top: 16px; color: #d5e2ea; }}
    h1 {{ font-size: 18px; margin: 0 0 8px; }}
    h2 {{ font-size: 15px; margin-top: 20px; }}
  </style>
</head>
<body>
<main>
  <aside>
    <h1>{title}</h1>
    <div id="shape"></div>
    <label for="frame">Frame</label>
    <input id="frame" type="range" min="0" value="0">
    <div id="frameLabel"></div>
    <label for="scale">Cell scale</label>
    <input id="scale" type="range" min="2" max="16" value="8">
    <label for="view">View</label>
    <select id="view">
      <option value="mask">Mask / split</option>
      <option value="speed">Speed magnitude</option>
      <option value="u">U component</option>
      <option value="v">V component</option>
      <option value="speed-mask">Speed + mask overlay</option>
      <option value="u-mask">U + mask overlay</option>
      <option value="v-mask">V + mask overlay</option>
    </select>
    <div id="rangeLabel"></div>
    <div class="legend" id="legend"></div>
    <h2>Frame Counts</h2>
    <div class="stats" id="stats"></div>
    <h2>Metadata</h2>
    <pre id="meta"></pre>
  </aside>
  <section>
    <canvas id="canvas"></canvas>
  </section>
</main>
<script>
const DATA = {data_json};
const colors = {{
  0: ["#07090b", "missing"],
  1: ["#2fb344", "train"],
  2: ["#f2cc3d", "val"],
  3: ["#ef6351", "test"],
  4: ["#4f8cff", "observed unsplit"],
  5: ["#323b44", "physics only"]
}};
const payload = DATA.payload;
const frameSlider = document.getElementById("frame");
const scaleSlider = document.getElementById("scale");
const viewSelect = document.getElementById("view");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
frameSlider.max = payload.frames.length - 1;
document.getElementById("shape").textContent = `${{payload.n}} frames, ${{payload.h}} x ${{payload.w}} grid`;
document.getElementById("meta").textContent = JSON.stringify(DATA.meta, null, 2);
document.getElementById("legend").innerHTML = Object.entries(colors).map(([k, v]) =>
  `<div class="swatch" style="background:${{v[0]}}"></div><div>${{v[1]}}</div>`
).join("");
if (!payload.includeData) {{
  [...viewSelect.options].forEach(opt => {{
    if (opt.value !== "mask") opt.disabled = true;
  }});
}}
function hexByte(row, x) {{
  return parseInt(row.slice(x * 2, x * 2 + 2), 16);
}}
function lerp(a, b, t) {{
  return Math.round(a + (b - a) * t);
}}
function rgb(r, g, b) {{
  return `rgb(${{r}},${{g}},${{b}})`;
}}
function speedColor(q) {{
  const t = q / 255;
  return rgb(lerp(16, 244, t), lerp(26, 198, t), lerp(34, 73, t));
}}
function signedColor(q) {{
  const t = q / 255;
  if (t < 0.5) {{
    const s = t / 0.5;
    return rgb(lerp(49, 236, s), lerp(110, 236, s), lerp(180, 236, s));
  }}
  const s = (t - 0.5) / 0.5;
  return rgb(lerp(236, 190, s), lerp(236, 52, s), lerp(236, 55, s));
}}
function maskStroke(code) {{
  if (code === "1") return "#35d066";
  if (code === "2") return "#ffd84a";
  if (code === "3") return "#ff7566";
  if (code === "4") return "#6aa2ff";
  if (code === "5") return "#6f7a84";
  return null;
}}
function draw() {{
  const frame = payload.frames[Number(frameSlider.value)];
  const scale = Number(scaleSlider.value);
  const view = viewSelect.value;
  const baseView = view.replace("-mask", "");
  canvas.width = payload.w * scale;
  canvas.height = payload.h * scale;
  for (let y = 0; y < payload.h; y++) {{
    const maskRow = frame.rows[y];
    const dataRow = frame.data ? frame.data[baseView]?.[y] : null;
    for (let x = 0; x < payload.w; x++) {{
      if (view === "mask" || !dataRow) {{
        ctx.fillStyle = colors[maskRow[x]][0];
      }} else {{
        const q = hexByte(dataRow, x);
        ctx.fillStyle = baseView === "speed" ? speedColor(q) : signedColor(q);
      }}
      ctx.fillRect(x * scale, y * scale, scale, scale);
      if (view.endsWith("-mask")) {{
        const stroke = maskStroke(maskRow[x]);
        if (stroke) {{
          ctx.strokeStyle = stroke;
          ctx.lineWidth = Math.max(1, Math.floor(scale / 4));
          ctx.strokeRect(x * scale + 0.5, y * scale + 0.5, scale - 1, scale - 1);
        }}
      }}
    }}
  }}
  document.getElementById("frameLabel").textContent =
    `Preview frame ${{Number(frameSlider.value) + 1}}/${{payload.frames.length}}: source index ${{frame.index}}`;
  const range = payload.ranges[baseView];
  document.getElementById("rangeLabel").textContent = range
    ? `${{baseView}} range: ${{range[0].toExponential(3)}} to ${{range[1].toExponential(3)}}`
    : "mask labels";
  document.getElementById("stats").innerHTML = Object.entries(frame.counts)
    .map(([k, v]) => `<div>${{k}}</div><div>${{v}}</div>`).join("");
}}
frameSlider.addEventListener("input", draw);
scaleSlider.addEventListener("input", draw);
viewSelect.addEventListener("input", draw);
draw();
</script>
</body>
</html>
"""
    html_path.write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create sparse observation masks and train/val/test splits.")
    parser.add_argument("--u", default="u_stack.npz", help="Path to u_stack.npz with key 'U'.")
    parser.add_argument("--v", default="v_stack.npz", help="Path to v_stack.npz with key 'V'.")
    parser.add_argument("--crop_h", type=int, default=64)
    parser.add_argument("--crop_w", type=int, default=64)
    parser.add_argument("--out", default="masks/splits_random_pixels_005_seed0.npz")
    parser.add_argument("--html", default="", help="Optional HTML viewer path. Defaults to OUT with .html.")
    parser.add_argument("--max_preview_frames", type=int, default=80)
    parser.add_argument("--no_data_preview", action="store_true",
                        help="Do not embed quantized U/V/speed previews in the HTML viewer.")
    parser.add_argument("--mask_model", default="random_pixels",
                        choices=["random_pixels", "moving_particles", "fixed_sensors", "regular_grid",
                                 "random_blocks", "center_block", "center_hole", "temporal_stride",
                                 "temporal_windows"])
    parser.add_argument("--obs_fraction", type=float, default=0.05)
    parser.add_argument("--block_h", type=int, default=8)
    parser.add_argument("--block_w", type=int, default=8)
    parser.add_argument("--split_strategy", default="time_block",
                        choices=["time_block", "random_observed", "spatial_block", "spatiotemporal_block"])
    parser.add_argument("--train_fraction", type=float, default=0.70)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--physics_mask", default="interior", choices=["interior", "observed_interior", "none"])
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.html:
        args.html = str(Path(args.out).with_suffix(".html"))
    Uc, Vc = _load_cropped_stacks(args.u, args.v, args.crop_h, args.crop_w)
    shape = tuple(int(v) for v in Uc.shape)
    obs_mask = make_observation_mask(args, shape)
    split_masks = make_splits(args, obs_mask)
    physics_mask = make_physics_mask(shape, obs_mask, args.physics_mask)
    masks = {"obs_mask": obs_mask, "physics_mask": physics_mask, **split_masks}
    meta = {
        "u": args.u,
        "v": args.v,
        "crop_h": args.crop_h,
        "crop_w": args.crop_w,
        "shape": shape,
        "mask_model": args.mask_model,
        "obs_fraction": args.obs_fraction,
        "split_strategy": args.split_strategy,
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "physics_mask": args.physics_mask,
        "seed": args.seed,
        "summaries": {name: summarize(mask) for name, mask in masks.items()},
    }
    write_npz(args, shape, masks, meta)
    write_html_viewer(args, masks, Uc, Vc, meta)
    print(json.dumps({"out": args.out, "html": args.html, "summaries": meta["summaries"]}, indent=2))


if __name__ == "__main__":
    main()
