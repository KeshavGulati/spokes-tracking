"""
Synthetic binary segmentation mask generator -- Cassini B-ring spoke blob
style, matching the reference masks in ./data/masks.

Style summary (from reference analysis):
  - Wide image ~4:1 (default 1488 x 380), strictly binary 0/255, hard edges.
  - 5-12 disconnected solid white regions inside a horizontal band in the
    vertical middle (top ~15% and bottom ~35% of the frame stay black).
  - Regions: teardrop / lens / leaf blobs (60-150 px), medium rounded or
    wedge blobs, small slivers and dots (10-30 px), often tilted 20-60 deg,
    sometimes with a flat bottom edge; smooth organic contours.
  - Usually ONE distinctive "three-prong" shape: three thin slightly curved
    vertical strokes above (touching or detached from) a small rounded blob.
  - Uneven left-to-right spacing with clusters and large gaps; no region
    touches the image border.

Rasterization is analytic (per-pixel predicate in each blob's local frame),
so edges are hard with no antialiasing.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

W, H = 1488, 380
BAND_TOP = 0.16 * H      # region centers/content stay below this
BAND_BOT = 0.62 * H      # ... and above this
MARGIN_X = 18


# ---------------------------------------------------------------------------
# blob profiles
# ---------------------------------------------------------------------------

def make_profile(rng, kind, L, Wd):
    """Return half-width function w(t), t in [-1, 1] along the blob axis."""
    # smooth organic perturbation of the contour
    a1 = rng.uniform(0.04, 0.12)
    f1 = rng.uniform(0.8, 2.2)
    p1 = rng.uniform(0, 2 * np.pi)
    a2 = rng.uniform(0.02, 0.07)
    f2 = rng.uniform(2.5, 5.0)
    p2 = rng.uniform(0, 2 * np.pi)

    def perturb(t):
        return 1.0 + a1 * np.sin(2 * np.pi * f1 * t + p1) \
                   + a2 * np.sin(2 * np.pi * f2 * t + p2)

    if kind == "lens":       # ellipse / leaf / lens
        p = rng.uniform(0.6, 1.5)
        base = lambda t: np.clip(1 - t ** 2, 0, None) ** p
    elif kind == "teardrop":  # asymmetric, pointed at one end
        p = rng.uniform(0.7, 1.4)
        q = rng.uniform(0.35, 0.9) * rng.choice([-1, 1])
        base = lambda t: np.clip(1 - t ** 2, 0, None) ** p * np.clip(1 + q * t, 0, None)
    elif kind == "wedge":     # triangular-ish, rounded
        p = rng.uniform(0.5, 1.1)
        base = lambda t: np.clip(1 - np.abs(t), 0, None) ** p
    else:
        raise ValueError(kind)

    def w(t):
        return (Wd / 2.0) * base(t) * perturb(t)
    return w


def draw_blob(mask, rng, cx, cy, L, Wd, tilt_deg, kind, flat_bottom=None):
    """Rasterize one solid tilted blob with hard edges."""
    th = np.deg2rad(tilt_deg)
    wfun = make_profile(rng, kind, L, Wd)
    rad = int(np.hypot(L, Wd) / 2) + 3
    x0, x1 = max(int(cx - rad), 0), min(int(cx + rad) + 1, W)
    y0, y1 = max(int(cy - rad), 0), min(int(cy + rad) + 1, H)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx, dy = xx - cx, yy - cy
    u = dx * np.cos(th) + dy * np.sin(th)     # along axis
    v = -dx * np.sin(th) + dy * np.cos(th)    # across axis
    t = u / (L / 2.0)
    inside = (np.abs(t) <= 1.0) & (np.abs(v) <= wfun(np.clip(t, -1, 1)))
    if flat_bottom is not None:               # image-frame horizontal cut
        inside &= yy <= flat_bottom
    mask[y0:y1, x0:x1] |= inside


def draw_prongs(mask, rng, cx, cy):
    """The signature 'three-prong' shape: small rounded blob + three thin,
    slightly curved vertical strokes above it (touching or detached)."""
    r = rng.uniform(7, 15)
    # base blob (slightly irregular round)
    draw_blob(mask, rng, cx, cy, 2 * r, 2 * r * rng.uniform(0.8, 1.1),
              rng.uniform(0, 180), "lens")

    detached = rng.random() < 0.4
    gap = rng.uniform(3, 10) if detached else -r * rng.uniform(0.2, 0.6)
    ph = rng.uniform(28, 62)                  # prong height
    d = r * rng.uniform(0.85, 1.25) + 4       # lateral prong spacing
    y_base = cy - r + gap                     # stroke bottom
    y_top = y_base - ph

    for i, off in enumerate((-d, 0.0, d)):
        h_i = ph * rng.uniform(0.75, 1.1) if i != 1 else ph
        ys = np.arange(int(np.ceil(y_top - (h_i - ph))), int(y_base) + 1)
        if len(ys) < 3:
            continue
        s = (y_base - ys) / max(h_i, 1)       # 0 at bottom -> 1 at top
        bend = rng.uniform(-8, 8)
        xc = cx + off + bend * s ** 2 + rng.uniform(-2, 2) * np.sin(np.pi * s)
        half_w = (1.2 + 2.2 * rng.uniform(0.5, 1.0)) * (1 - 0.55 * s)
        for y, xcv, hw in zip(ys, xc, half_w):
            if y < 0 or y >= H:
                continue
            xa, xb = int(np.floor(xcv - hw)), int(np.ceil(xcv + hw))
            xa, xb = max(xa, 0), min(xb, W - 1)
            if xb >= xa:
                mask[y, xa:xb + 1] = True


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------

def generate_mask(seed=None):
    rng = np.random.default_rng(seed)
    mask = np.zeros((H, W), dtype=bool)

    n = int(rng.integers(5, 13))              # 5..12 regions
    include_fork = rng.random() < 0.85

    # --- pre-generate blob params: size, kind, tilt, flat-bottom ----------
    params = []
    n_large = int(rng.integers(1, 4))
    n_small = int(rng.integers(1, max(2, n - n_large - 1)))
    for i in range(n):
        if i < n_large:
            L = rng.uniform(60, 150); Wd = L * rng.uniform(0.28, 0.6)
            kind = rng.choice(["teardrop", "lens", "lens"])
        elif i < n_large + n_small:
            L = rng.uniform(10, 30); Wd = L * rng.uniform(0.2, 0.9)
            kind = rng.choice(["lens", "wedge"])
        else:
            L = rng.uniform(30, 65); Wd = L * rng.uniform(0.35, 0.8)
            kind = rng.choice(["teardrop", "lens", "wedge"])
        tilt = rng.choice([-1, 1]) * rng.uniform(20, 60)
        if rng.random() < 0.2:
            tilt *= rng.uniform(0.0, 0.3)     # occasional near-vertical
        params.append({"L": L, "Wd": Wd, "kind": kind, "tilt": tilt})
    order = rng.permutation(n)

    # --- x layout: sequential, uneven gaps, spread across the full width --
    # projected horizontal extent of each tilted blob
    widths = []
    for idx in order:
        p = params[idx]
        th = np.deg2rad(abs(p["tilt"]))
        widths.append(p["L"] * np.sin(th) + p["Wd"] * np.cos(th) + 4)
    if include_fork:
        widths.append(48)
    widths = np.array(widths)

    gaps = rng.uniform(25, 90, size=len(widths) - 1)
    gaps[rng.random(len(gaps)) < 0.25] = rng.uniform(120, 260)   # big gaps
    gaps[rng.random(len(gaps)) < 0.30] = rng.uniform(12, 28)     # clusters
    avail = W - 2 * MARGIN_X
    slack = avail - widths.sum() - 24          # 24 = outer padding
    if slack < 10 * len(gaps):                 # too wide: shrink big blobs
        over = (10 * len(gaps) - slack) / max(widths.sum(), 1e-6)
        scale = max(1.0 - over, 0.45)
        for idx in order:
            params[idx]["L"] *= scale
            params[idx]["Wd"] *= scale
        widths *= scale
        slack = avail - widths.sum() - 24
    gaps = gaps * max(slack, 1e-6) / max(gaps.sum(), 1e-6)

    x = MARGIN_X + rng.uniform(4, 20)
    positions = []
    for j, wdt in enumerate(widths):
        positions.append(x + wdt / 2)
        if j < len(gaps):
            x += wdt + gaps[j]

    # --- draw blobs ------------------------------------------------------
    for j, idx in enumerate(order):
        p = params[idx]
        cx = positions[j]
        half_h = (p["L"] * abs(np.sin(np.deg2rad(p["tilt"])))
                  + p["Wd"] * abs(np.cos(np.deg2rad(p["tilt"])))) / 2 + 4
        cy_lo = max(BAND_TOP + half_h * 0.7, 18)
        cy_hi = min(BAND_BOT - half_h * 0.7, H - 22)
        cy = rng.uniform(cy_lo, max(cy_hi, cy_lo + 1))
        flat = None
        if rng.random() < 0.25:               # flat bottom cut
            flat = cy + p["Wd"] / 2 * rng.uniform(0.15, 0.75)
        draw_blob(mask, rng, cx, cy, p["L"], p["Wd"], p["tilt"], p["kind"],
                  flat_bottom=flat)

    if include_fork:
        cx = positions[-1]
        cy = rng.uniform(BAND_TOP + 45, BAND_BOT - 12)
        draw_prongs(mask, rng, cx, cy)

    # --- safety: nothing touches the border ------------------------------
    mask[:6, :] = False
    mask[-6:, :] = False
    mask[:, :6] = False
    mask[:, -6:] = False
    return mask


def check(mask):
    lab, n = ndimage.label(mask)
    ys = np.where(mask.any(axis=1))[0]
    xs = np.where(mask.any(axis=0))[0]
    assert set(np.unique((mask * 255).astype(np.uint8)).tolist()) <= {0, 255}
    border = mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any()
    return n, (ys.min(), ys.max()), (xs.min(), xs.max()), border


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--num", type=int, default=1)
    ap.add_argument("-o", "--out", default="data/synthetic_masks/blob_sample")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--start-index", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy)

    for i in range(args.num):
        seed = base + i
        mask = generate_mask(seed=seed)
        nreg, yr, xr, border = check(mask)
        name = f"blob_mask_{args.start_index + i:04d}.png"
        Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(out / name)
        print(f"{name}  seed={seed}  regions={nreg}  "
              f"y={yr[0]}-{yr[1]} ({yr[0]/H:.2f}-{yr[1]/H:.2f})  x={xr[0]}-{xr[1]}  "
              f"border_touch={border}")


if __name__ == "__main__":
    main()
