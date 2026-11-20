"""
Synthetic binary segmentation mask generator for radial spokes in Saturn's
B-ring, mimicking Cassini-style spoke ground-truth annotations.

Conventions (matching the project's demo masks in ./data/masks):
  - Horizontal axis (x) = azimuthal position (longitude)
  - Vertical axis (y)   = radial distance within the rings
  - Spokes extend predominantly vertically (radially), with azimuthal
    displacement, curvature, tapering, branching and fragmentation.

Output: single-channel PNG, strictly binary (0 / 255), no antialiasing.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

IMG_SIZE = 512


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def smooth_noise(n, rng, scale=9, passes=3):
    """Smoothed 1-D random noise of length n, roughly in [-1, 1]."""
    if n <= 1:
        return np.zeros(max(n, 0))
    x = rng.normal(size=n)
    for _ in range(passes):
        k = max(2, scale)
        kernel = np.hanning(k)
        kernel /= kernel.sum()
        x = np.convolve(np.pad(x, k, mode="edge"), kernel, mode="same")[k:-k]
    s = np.abs(x).max()
    return x / s if s > 1e-9 else x


def fill_stroke(mask, ys, xl, xr):
    """Rasterize a variable-width vertical stroke directly into a bool mask
    (hard pixel edges, no antialiasing)."""
    h, w = mask.shape
    for y, a, b in zip(ys, xl, xr):
        if y < 0 or y >= h:
            continue
        x0 = int(np.floor(a))
        x1 = int(np.ceil(b))
        if x1 < x0:
            continue
        x0 = max(x0, 0)
        x1 = min(x1, w - 1)
        if x1 >= x0:
            mask[y, x0:x1 + 1] = True


def scatter_fragments(mask, rng, ys, xc, half_w, n_frag, spread):
    """Small detached blobs / specks along a spoke -> fragmented, diffuse
    outline appearance while staying strictly binary."""
    h, w = mask.shape
    for _ in range(n_frag):
        idx = rng.integers(0, len(ys))
        y = int(ys[idx]) + rng.integers(-4, 5)
        x = int(xc[idx] + rng.uniform(-spread, spread) * (half_w[idx] + 6))
        r = rng.integers(1, 4)
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        blob = (xx ** 2 + yy ** 2) <= r ** 2 + rng.integers(-1, 3)
        y0, y1 = max(y - r, 0), min(y + r + 1, h)
        x0, x1 = max(x - r, 0), min(x + r + 1, w)
        by0, by1 = y0 - (y - r), blob.shape[0] - ((y + r + 1) - y1)
        bx0, bx1 = x0 - (x - r), blob.shape[1] - ((x + r + 1) - x1)
        if y1 > y0 and x1 > x0:
            mask[y0:y1, x0:x1] |= blob[by0:by1, bx0:bx1]


# ---------------------------------------------------------------------------
# spoke model
# ---------------------------------------------------------------------------

def draw_spoke(mask, rng, y_start, y_end, x_base, w_max, tilt, curve_amp,
               taper, branch_prob=0.35, fragment_prob=0.4, parent=False):
    """Draw one irregular, predominantly radial spoke.

    y_start / y_end may lie outside the image (spokes truncated by borders).
    The centerline x(y) combines a linear tilt, smooth curvature (random
    sinusoid mixture) and boundary noise. The half-width profile varies
    along the spoke with tapering ends and irregular modulation.
    """
    ys = np.arange(y_start, y_end + 1)
    n = len(ys)
    if n < 5:
        return
    t = (ys - y_start) / max(n - 1, 1)  # 0..1 along the spoke

    # --- centerline: tilt + curvature + wiggle -------------------------
    u = (ys - (y_start + y_end) / 2) / IMG_SIZE
    curve = curve_amp * (
        np.sin(2 * np.pi * rng.uniform(0.6, 1.8) * u + rng.uniform(0, 2 * np.pi))
        + 0.5 * np.sin(2 * np.pi * rng.uniform(2.0, 4.0) * u + rng.uniform(0, 2 * np.pi))
    )
    wiggle = rng.uniform(1.5, 6.0) * smooth_noise(n, rng, scale=max(5, n // 12))
    xc = x_base + tilt * (ys - ys.mean()) + curve + wiggle

    # --- half-width profile: taper + irregular modulation ---------------
    # taper: raised-cosine ends, optionally stronger on one side (asymmetry)
    end0 = np.clip(t / max(taper, 1e-3), 0, 1)
    end1 = np.clip((1 - t) / max(taper * rng.uniform(0.6, 1.6), 1e-3), 0, 1)
    shape = np.minimum(end0, end1) ** rng.uniform(0.4, 0.9)
    mod = 1.0 + 0.45 * smooth_noise(n, rng, scale=max(7, n // 6))
    half_w = np.maximum(w_max * shape * mod, 0.0)

    # --- fragmentation: drop whole segments of the spoke ----------------
    presence = np.ones(n, dtype=bool)
    if rng.random() < fragment_prob:
        n_gaps = rng.integers(1, 4)
        for _ in range(n_gaps):
            g = rng.integers(3, max(4, int(0.18 * n)))
            s = rng.integers(0, max(n - g, 1))
            presence[s:s + g] = False
    xl = xc - half_w
    xr = xc + half_w
    xl[~presence] = 1.0
    xr[~presence] = -1.0  # inverted interval -> nothing filled

    fill_stroke(mask, ys, xl, xr)

    # --- diffuse / fragmented outline specks ----------------------------
    if rng.random() < fragment_prob:
        scatter_fragments(mask, rng, ys[presence], xc[presence],
                          half_w[presence], rng.integers(4, 14),
                          spread=rng.uniform(0.8, 1.8))

    # --- branching: a shorter side spoke diverging from this one --------
    if rng.random() < branch_prob and not parent:
        idx = rng.integers(int(0.2 * n), int(0.8 * n))
        by0 = ys[idx]
        blen = int(rng.uniform(0.2, 0.5) * n)
        direction = rng.choice([-1, 1])
        b_y1 = by0 + direction * blen
        draw_spoke(
            mask, rng,
            y_start=by0, y_end=b_y1,
            x_base=xc[idx],
            w_max=w_max * rng.uniform(0.3, 0.6),
            tilt=rng.uniform(0.15, 0.5) * rng.choice([-1, 1]),
            curve_amp=curve_amp * 0.5,
            taper=rng.uniform(0.3, 0.6),
            branch_prob=0.0,
            fragment_prob=fragment_prob,
            parent=True,
        )


def generate_mask(seed=None):
    """Generate one strictly binary 512x512 spoke mask."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)

    # occasional empty mask
    if rng.random() < 0.10:
        return mask

    n_spokes = rng.integers(1, 7)  # 1..6

    # azimuthal positions: uniform, sometimes with one cluster
    clustered = rng.random() < 0.4
    if clustered:
        cx = rng.uniform(0.2, 0.8) * IMG_SIZE
        xs = np.concatenate([
            cx + rng.normal(0, rng.uniform(15, 45), size=max(n_spokes - 1, 1)),
            [rng.uniform(0.05, 0.95) * IMG_SIZE] * (n_spokes - max(n_spokes - 1, 1)),
        ])[:n_spokes]
    else:
        xs = rng.uniform(-0.05, 1.05, size=n_spokes) * IMG_SIZE

    for x_base in xs:
        # radial extent: often spanning most of the frame, truncated at edges
        full = rng.random() < 0.55
        if full:
            y_start = rng.integers(-80, int(0.25 * IMG_SIZE))
            y_end = rng.integers(int(0.75 * IMG_SIZE), IMG_SIZE + 80)
        else:
            y_start = rng.integers(-40, int(0.7 * IMG_SIZE))
            y_end = y_start + rng.integers(int(0.15 * IMG_SIZE), IMG_SIZE)
        w_max = rng.uniform(1.5, 16.0)  # half-width scale: narrow -> broad
        draw_spoke(
            mask, rng,
            y_start=int(y_start), y_end=int(y_end),
            x_base=float(x_base),
            w_max=w_max,
            tilt=rng.uniform(-0.15, 0.15),
            curve_amp=rng.uniform(3, 30),
            taper=rng.uniform(0.1, 0.45),
            branch_prob=0.35,
            fragment_prob=0.4,
        )

    # occasional overlapping: draw one more spoke on top of an existing one
    if n_spokes >= 2 and rng.random() < 0.3:
        i = rng.integers(0, len(xs))
        draw_spoke(
            mask, rng,
            y_start=rng.integers(-60, int(0.3 * IMG_SIZE)),
            y_end=rng.integers(int(0.7 * IMG_SIZE), IMG_SIZE + 60),
            x_base=float(xs[i]) + rng.uniform(-25, 25),
            w_max=rng.uniform(2, 10),
            tilt=rng.uniform(-0.2, 0.2),
            curve_amp=rng.uniform(5, 25),
            taper=rng.uniform(0.1, 0.4),
        )

    return mask


def save_mask(mask, path):
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--num", type=int, default=10, help="number of masks")
    ap.add_argument("-o", "--out", default="data/synthetic_masks/sample",
                    help="output directory")
    ap.add_argument("--seed", type=int, default=None,
                    help="base seed (None -> random, different per run)")
    ap.add_argument("--start-index", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    base_seed = args.seed if args.seed is not None else \
        int(np.random.SeedSequence().entropy)

    for i in range(args.num):
        seed = base_seed + i
        mask = generate_mask(seed=seed)
        path = out / f"synthetic_spoke_mask_{args.start_index + i:04d}.png"
        save_mask(mask, path)
        n_white = int(mask.sum())
        print(f"{path.name}  seed={seed}  spokes_px={n_white} "
              f"({'empty' if n_white == 0 else 'populated'})")


if __name__ == "__main__":
    main()
