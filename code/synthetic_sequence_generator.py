"""
Synthetic binary segmentation mask SEQUENCE generator -- simulates N
back-to-back Cassini observations of Saturn's B-ring spokes at a fixed
interval (default 975 s, matching ./data/masks naming W<epoch>_mask.png).

Each sequence renders persistent blob features that:
  - drift azimuthally (x) frame-to-frame with per-feature velocities
    (measured from the demo series: ~7-40 px per 975 s frame),
  - drift slightly radially (y),
  - slowly evolve (tilt, length, width jitter),
  - appear / fade over their lifetime (some live all frames).

Style is identical to code/synthetic_blob_mask_generator.py: wide 1488x380,
strictly binary, 5-12 disconnected solid regions inside the central band,
usually one "three-prong" shape, hard edges (analytic rasterization).

Sequences whose frames violate the style constraints (region count, band,
border contact, accidental merging) are rejected and re-rolled.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from synthetic_blob_mask_generator import W, H, BAND_TOP, BAND_BOT, MARGIN_X

INTERVAL = 975          # seconds between back-to-back frames (demo cadence)
N_FRAMES = 10


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

class Feature:
    """One blob (or fork) with constant velocity + slow shape evolution."""

    def __init__(self, rng, f0, n_frames, kind):
        self.kind = kind                       # 'blob' or 'fork'
        self.born = f0
        self.die = f0 + rng.integers(4, n_frames + 1)   # exclusive
        self.die = min(self.die, n_frames)

        # motion
        self.vx = rng.choice([-1, 1]) * rng.uniform(6, 40)
        self.vy = rng.uniform(-4, 4)
        # per-frame shape evolution
        self.dtilt = rng.uniform(-1.8, 1.8)
        self.dlen = rng.uniform(-0.02, 0.02)
        self.dwid = rng.uniform(-0.025, 0.025)

        # shape
        if kind == "blob":
            big = rng.random() < 0.35
            small = rng.random() < 0.3 if not big else False
            if big:
                self.L = rng.uniform(60, 150); self.Wd = self.L * rng.uniform(0.28, 0.6)
                self.shape = rng.choice(["teardrop", "lens", "lens"])
            elif small:
                self.L = rng.uniform(10, 30); self.Wd = self.L * rng.uniform(0.3, 0.9)
                self.shape = rng.choice(["lens", "wedge"])
            else:
                self.L = rng.uniform(30, 65); self.Wd = self.L * rng.uniform(0.35, 0.8)
                self.shape = rng.choice(["teardrop", "lens", "wedge"])
            self.tilt = rng.choice([-1, 1]) * rng.uniform(20, 60)
            if rng.random() < 0.2:
                self.tilt *= rng.uniform(0.0, 0.3)
            self.flat = rng.random() < 0.25    # flat bottom cut
            self.flat_k = rng.uniform(0.15, 0.75)
        else:  # fork
            self.tilt = 0.0                    # forks stay upright
            self.shape = "lens"                # base blob is roundish
            self.detached = rng.random() < 0.4
            self.hw = rng.uniform(1.8, 3.4)
            # prong spacing wide enough that prongs never touch each other
            prong_d = 2 * self.hw + rng.uniform(5, 12)
            if self.detached:
                self.L = rng.uniform(14, 30)
                self.gap = rng.uniform(4, 9)   # prongs float above the blob
            else:
                # blob wide enough that all three prongs root inside it
                self.L = 2 * prong_d + rng.uniform(8, 18)
                self.gap = 0.0                 # prong bottoms at blob center
            self.Wd = self.L * rng.uniform(0.8, 1.0)
            # prong offset as a fraction of L so it scales with fade-shrink
            self.prong_frac = prong_d / self.L
            self.ph = rng.uniform(28, 62)
            self.bends = rng.uniform(-6, 6) + rng.uniform(-1.5, 1.5, size=3)

        # frozen contour perturbation (stable identity across frames)
        self.a1 = rng.uniform(0.04, 0.12); self.f1 = rng.uniform(0.8, 2.2)
        self.p1 = rng.uniform(0, 2 * np.pi)
        self.a2 = rng.uniform(0.02, 0.07); self.f2 = rng.uniform(2.5, 5.0)
        self.p2 = rng.uniform(0, 2 * np.pi)
        if self.kind == "blob" and self.shape == "teardrop":
            self.q = rng.uniform(0.35, 0.9) * rng.choice([-1, 1])
        else:
            self.q = 0.0
        self.pexp = rng.uniform(0.6, 1.4)

        # position at birth (set by layout)
        self.x0 = self.y0 = None

    # --- half-width profile ------------------------------------------------
    def half_w(self, t, Wd):
        if self.shape == "lens":
            base = np.clip(1 - t ** 2, 0, None) ** self.pexp
        elif self.shape == "teardrop":
            base = (np.clip(1 - t ** 2, 0, None) ** self.pexp
                    * np.clip(1 + self.q * t, 0, None))
        else:  # wedge
            base = np.clip(1 - np.abs(t), 0, None) ** self.pexp
        pert = (1.0 + self.a1 * np.sin(2 * np.pi * self.f1 * t + self.p1)
                + self.a2 * np.sin(2 * np.pi * self.f2 * t + self.p2))
        return (Wd / 2.0) * base * pert

    def alive(self, f):
        return self.born <= f < self.die

    def n_regions(self):
        """Expected connected regions when isolated (fork prongs detached)."""
        if self.kind == "fork":
            return 4 if self.detached else 1
        return 1

    def dims(self, f):
        """L, Wd, tilt at frame f, with fade-in/out scaling."""
        k = f - self.born
        L = self.L * (1 + self.dlen) ** k
        Wd = self.Wd * (1 + self.dwid) ** k
        tilt = self.tilt + self.dtilt * k
        life = self.die - self.born
        if life >= 3:
            if k == 0:
                L *= 0.65; Wd *= 0.65
            elif k == life - 1:
                L *= 0.7; Wd *= 0.7
        # rasterization floor: thinner shapes would fragment into
        # disconnected pixel runs or vanish entirely
        L = max(L, 9.0)
        Wd = max(Wd, 3.2)
        return L, Wd, tilt

    def state(self, f):
        """Position / shape at frame f."""
        k = f - self.born
        L, Wd, tilt = self.dims(f)
        x = self.x0 + self.vx * k
        y = self.y0 + self.vy * k
        return x, y, L, Wd, tilt

    def extent(self, f):
        L, Wd, tilt = self.dims(f)
        th = np.deg2rad(abs(tilt))
        hh = (L * np.sin(th) + Wd * np.cos(th)) / 2 + 5
        hw = (L * np.cos(th) + Wd * np.sin(th)) / 2 + 5
        if self.kind == "fork":
            hh = max(hh, self.ph + Wd / 2 + 10)
            hw = max(hw, self.prong_frac * L + self.hw
                     + abs(self.bends).max() + 4)
        return hw, hh

    # --- rendering ----------------------------------------------------------
    def render(self, mask, f):
        x, y, L, Wd, tilt = self.state(f)
        hwfun = lambda t: self.half_w(t, Wd)
        if self.kind == "blob":
            flat = (y + Wd / 2 * self.flat_k) if self.flat else None
            _draw_blob(mask, x, y, L, Wd, tilt, hwfun, flat)
        else:
            _draw_fork(mask, self, x, y, L, Wd, hwfun)


def _draw_blob(mask, cx, cy, L, Wd, tilt_deg, hwfun, flat_bottom=None):
    th = np.deg2rad(tilt_deg)
    rad = int(np.hypot(L, Wd) / 2) + 3
    x0, x1 = max(int(cx - rad), 0), min(int(cx + rad) + 1, W)
    y0, y1 = max(int(cy - rad), 0), min(int(cy + rad) + 1, H)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx, dy = xx - cx, yy - cy
    u = dx * np.cos(th) + dy * np.sin(th)
    v = -dx * np.sin(th) + dy * np.cos(th)
    t = u / (L / 2.0)
    inside = (np.abs(t) <= 1.0) & (np.abs(v) <= hwfun(np.clip(t, -1, 1)))
    if flat_bottom is not None:
        inside &= yy <= flat_bottom
    mask[y0:y1, x0:x1] |= inside


def _draw_fork(mask, feat, cx, cy, L, Wd, hwfun):
    _draw_blob(mask, cx, cy, L, Wd, 0.0, hwfun)
    if feat.detached:
        y_base = cy - Wd / 2 - feat.gap        # prongs float above the blob
    else:
        y_base = cy - feat.gap                 # prong bottoms root at center
    d = feat.prong_frac * L                    # scales with fade-shrink
    for i, off in enumerate((-d, 0.0, d)):
        ph = feat.ph * (1.0 if i == 1 else 0.85)
        ys = np.arange(int(y_base - ph), int(y_base) + 1)
        if len(ys) < 3:
            continue
        s = (y_base - ys) / ph
        xc = cx + off + feat.bends[i] * s ** 2
        hw = feat.hw * (1 - 0.55 * s)
        for y, xcv, hwi in zip(ys, xc, hw):
            if y < 0 or y >= H:
                continue
            xa, xb = int(np.floor(xcv - hwi)), int(np.ceil(xcv + hwi))
            xa, xb = max(xa, 0), min(xb, W - 1)
            if xb >= xa:
                mask[y, xa:xb + 1] = True


# ---------------------------------------------------------------------------
# sequence assembly
# ---------------------------------------------------------------------------

def place_features(rng, feats, n_frames):
    """Assign x0/y0 so every feature's full swept x-range is disjoint from
    all others (guarantees regions never touch in any frame)."""
    rng.shuffle(feats)
    avail = W - 2 * MARGIN_X

    def sweep_of(ft):
        hw = max(ft.extent(f)[0] for f in range(ft.born, ft.die))
        drift = abs(ft.vx) * (ft.die - ft.born - 1)
        return hw, drift

    sweeps = [sweep_of(ft) for ft in feats]
    slot_w = [2 * hw + drift for hw, drift in sweeps]
    min_gap = 28
    need = sum(slot_w) + min_gap * (len(feats) - 1)
    if need > avail:
        # slow everyone down and retry once
        scale = max((avail - 2 * sum(hw for hw, _ in sweeps)
                     - min_gap * (len(feats) - 1)) /
                    max(sum(d for _, d in sweeps), 1e-6), 0.0)
        if scale <= 0.05:
            return False
        for ft in feats:
            ft.vx *= scale
            ft.vy *= scale
        sweeps = [sweep_of(ft) for ft in feats]
        slot_w = [2 * hw + drift for hw, drift in sweeps]
        need = sum(slot_w) + min_gap * (len(feats) - 1)
        if need > avail:
            return False

    gaps = rng.uniform(0.6, 1.6, size=max(len(feats), 1))
    gaps = min_gap + gaps * (avail - need) / gaps.sum()
    x = MARGIN_X + rng.uniform(0, 12)
    for ft, sw, gp in zip(feats, slot_w, gaps):
        hw, drift = sweep_of(ft)
        if ft.vx >= 0:
            ft.x0 = x + hw
        else:
            ft.x0 = x + hw + drift
        # vertical placement inside the band, valid across the lifetime
        hh = max(ft.extent(f)[1] for f in range(ft.born, ft.die))
        ydrift = ft.vy * (ft.die - ft.born - 1)
        ylo = BAND_TOP + hh * 0.85 - min(ydrift, 0)
        yhi = BAND_BOT - hh * 0.85 - max(ydrift, 0)
        if yhi < ylo:
            ft.vy = 0.0
            ylo = BAND_TOP + hh * 0.85
            yhi = BAND_BOT - hh * 0.85
            if yhi < ylo:
                return False
        ft.y0 = rng.uniform(ylo, yhi)
        x += sw + gp
    return True


def _build_features(rng, n_frames):
    """Feature set with lifetimes/velocities; None if region-count budget
    (5-12 per frame) cannot be respected."""
    include_fork = rng.random() < 0.85
    n0 = int(rng.integers(6, 10))              # alive at frame 0 (incl. fork)
    n_trans = int(rng.integers(0, 3))          # appear mid-sequence

    feats = []
    for i in range(n0):
        kind = "fork" if (include_fork and i == 0) else "blob"
        ft = Feature(rng, 0, n_frames, kind)
        if kind == "fork":
            ft.die = n_frames if rng.random() < 0.7 else int(rng.integers(5, n_frames + 1))
        else:
            ft.die = n_frames if rng.random() < 0.55 else int(rng.integers(4, n_frames + 1))
        feats.append(ft)
    for _ in range(n_trans):
        born = int(rng.integers(1, max(n_frames - 2, 2)))
        ft = Feature(rng, born, n_frames, "blob")
        ft.die = min(n_frames, born + int(rng.integers(2, n_frames - born + 1)))
        feats.append(ft)

    # velocities: demo-measured azimuthal drift ~5-40 px per frame,
    # bounded so total drift fits the layout budget
    for ft in feats:
        life = ft.die - ft.born
        max_drift = rng.uniform(30, 150)
        vmax = max(max_drift / max(life - 1, 1), 2.0)
        ft.vx = rng.choice([-1, 1]) * min(rng.uniform(5, 25), vmax)
        ft.vy = rng.uniform(-3, 3)

    # per-frame region budget 5..12 (detached fork counts as 4)
    for f in range(n_frames):
        n = sum(ft.n_regions() for ft in feats if ft.alive(f))
        if not (5 <= n <= 12):
            return None
    return feats


def render_sequence(seed, n_frames=N_FRAMES):
    rng = np.random.default_rng(seed)
    for _ in range(30):
        feats = _build_features(rng, n_frames)
        if feats is None:
            continue
        if place_features(rng, feats, n_frames):
            break
    else:
        return None, None

    frames = []
    for f in range(n_frames):
        m = np.zeros((H, W), dtype=bool)
        for ft in feats:
            if ft.alive(f):
                ft.render(m, f)
        # hard safety margins
        m[:6, :] = False; m[-6:, :] = False
        m[:, :6] = False; m[:, -6:] = False
        # drop 1-3 px rasterization specks at tapered blob tips
        lab, n = ndimage.label(m)
        if n:
            sizes = np.bincount(lab.ravel())
            small = np.flatnonzero(sizes < 4)
            if small.size:
                m[np.isin(lab, small)] = False
        frames.append(m)
    return frames, feats


def validate(frames, feats):
    for f, m in enumerate(frames):
        if not m.any():
            return False, f"empty frame {f}"
        lab, n = ndimage.label(m)
        expected = sum(ft.n_regions() for ft in feats if ft.alive(f))
        if n != expected:
            return False, f"frame {f}: {n} regions != expected {expected} (merge/split)"
        if not (5 <= n <= 14):
            return False, f"frame {f}: region count {n} out of range"
        ys = np.where(m.any(axis=1))[0]
        if ys.min() < 0.13 * H or ys.max() > 0.66 * H:
            return False, f"frame {f}: band violation {ys.min()}-{ys.max()}"
        if m[0, :].any() or m[-1, :].any() or m[:, 0].any() or m[:, -1].any():
            return False, f"frame {f}: border touch"
    return True, "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default="data/synthetic_masks/blob_sequence")
    ap.add_argument("--frames", type=int, default=N_FRAMES)
    ap.add_argument("--interval", type=int, default=INTERVAL)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--t0", type=int, default=None, help="epoch of first frame")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    frames = None
    for attempt in range(40):
        s = seed if seed is not None else int(np.random.SeedSequence().entropy)
        if seed is not None:
            s = seed + attempt * 7919
        frames, feats = render_sequence(s, args.frames)
        if frames is None:
            continue
        ok, msg = validate(frames, feats)
        if ok:
            seed = s
            break
        frames = None
    if frames is None:
        raise SystemExit("could not generate a valid sequence after 40 attempts")

    rng = np.random.default_rng(seed)
    t0 = args.t0 if args.t0 is not None else int(rng.integers(1597900000, 1598100000) // 5 * 5)
    names = []
    for i, m in enumerate(frames):
        ts = t0 + i * args.interval
        name = f"W{ts}_mask.png"
        Image.fromarray((m * 255).astype(np.uint8), mode="L").save(out / name)
        n = ndimage.label(m)[1]
        names.append(name)
        print(f"{name}  frame={i}  regions={n}")
    print(f"\nsequence seed={seed}  interval={args.interval}s  -> {out}/")

    # contact sheet preview
    th = 96
    sheet = Image.new("L", (th * 5, int(th * H / W) * 2), 0)
    for i, name in enumerate(names):
        im = Image.open(out / name).resize((th, int(th * H / W)), Image.NEAREST)
        sheet.paste(im, ((i % 5) * th, (i // 5) * im.height))
    sheet.save(out / "sequence_contact_sheet.png")
    print(f"preview: {out}/sequence_contact_sheet.png")


if __name__ == "__main__":
    main()
