#!/usr/bin/env python3
"""Refine OSM building footprints in the Trisuli Bazar town core against the
Vantor Legion 5 Feb 2026 pre-flood mosaic.

    python3 scripts/refine_footprints.py --check       reliability diagnostic
    python3 scripts/refine_footprints.py --previews    before/after preview PNGs
    python3 scripts/refine_footprints.py --write       apply to assets/trisuli/osm.json

DEFAULT IS A DRY RUN.  --write is deliberately opt-in, because the diagnostic
below says the refinement is NOT reliable for this area (see VERDICT).

METHOD
  Footprints are near-rectangular (91% are 4-gons, median 7 x 10 m), so each is
  modelled by its minimum-area bounding rectangle and re-fitted to the imagery
  over translation (+-2.5 world px), rotation (+-6 deg) and per-axis scale.
  Two objectives were built and both are implemented here:

  "region"   mean roof-score inside the rectangle minus the mean in a 1-world-px
             outer ring, plus an oriented boundary term (the derivative
             perpendicular to each side on a 3-px band, minus the mean gradient
             of the interior so that the two are directly comparable).
             The roof score is a log-likelihood ratio over a smoothed 16^3 RGB
             histogram, learned from pixels well inside large existing
             footprints versus pixels >=2.5 world px outside every footprint.

  "chamfer"  mean distance from points sampled along the outline to the nearest
             strong image edge, from an exact Euclidean distance transform of
             the top-12% gradient pixels, split by edge orientation so that a
             side only matches edges running the same way.

ACCEPTANCE
  Each footprint is fitted from 7 different starting offsets.  A refinement is
  accepted only if the 7 fits agree to within 0.25 world px (a well defined
  optimum), the consensus moves 0.3..2.5 world px, and the cost improves by
  >=15%.  Multi-start agreement is the acceptance test because a fit that
  depends on where it started is noise, not evidence.

VERDICT (5 Sep 2026, run with --check on 200 town-core footprints)
  Mean spread of the fitted centre across the 7 starts is 0.7-1.2 world px
  (2-3.6 m) for every objective and every weighting tried; only 1-22% of
  footprints converge to within 0.25 world px.  That repeatability is the same
  size as the misalignment being corrected.  Independently, cross-correlating
  the footprint outlines against the image gradient over the whole area peaks
  at +0.12 world px, i.e. there is no bulk offset to remove, and per-cell
  estimates on a 40-world-px grid point in incoherent directions.  Visual
  checks at 3x on the clearest roofs (the riverside row south of the bazaar)
  show the existing outlines already on the roofs and the accepted refinements
  moving them off, onto road edges and terrace walls.  Conclusion: do not
  apply.  osm.json is left unchanged.
"""
import argparse, json, math, os, sys, time
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OSM_PATH = os.path.join(ROOT, "assets/trisuli/osm.json")
TILES = os.path.join(ROOT, "assets/trisuli/tiles/vantor_pre")
PREVIEWS = os.path.join(ROOT, "previews")

F = 8                                   # tile px per world px at level z8
M_PER_WPX = 2.93
BOX = (20905, 19518, 21205, 19800)      # town core, world px
PAD = 12
WX0, WY0 = BOX[0] - PAD, BOX[1] - PAD

MAX_SHIFT_PX = 12                       # +-1.5 world px translation search
ANGLES = np.deg2rad(np.arange(-6, 6.1, 1.5))
SCALES = np.array([0.88, 0.94, 1.00, 1.06, 1.13, 1.20])
STARTS = [(0, 0), (0.7, 0), (-0.7, 0), (0, 0.7), (0, -0.7), (0.5, 0.5), (-0.5, -0.5)]
AGREE_TOL = 0.25                        # world px
MIN_SHIFT, MAX_SHIFT = 0.3, 2.5         # world px
MIN_GAIN = 0.15                         # relative cost improvement
MIN_MINOR_DIM = 1.2                     # world px; smaller footprints are skipped


# ---------------------------------------------------------------- mosaic ----

def build_mosaic(wx0, wy0, wx1, wy1):
    idx = json.load(open(os.path.join(TILES, "index.json")))
    lv = [l for l in idx["levels"] if l["F"] == F][0]
    T, keys = lv["T"], set(lv["tiles"])
    tdir = os.path.join(TILES, lv["dir"])
    W, H = int(round((wx1 - wx0) * F)), int(round((wy1 - wy0) * F))
    out = np.zeros((H, W, 3), np.uint8)
    cov = np.zeros((H, W), bool)
    for tx in range(math.floor(wx0 / T), math.floor(wx1 / T) + 1):
        for ty in range(math.floor(wy0 / T), math.floor(wy1 / T) + 1):
            k = "%d_%d" % (tx, ty)
            if k not in keys:
                continue
            im = np.asarray(Image.open(os.path.join(tdir, k + ".webp")).convert("RGB"))
            px, py = int(round((tx * T - wx0) * F)), int(round((ty * T - wy0) * F))
            x0, y0 = max(0, px), max(0, py)
            x1, y1 = min(W, px + im.shape[1]), min(H, py + im.shape[0])
            if x1 <= x0 or y1 <= y0:
                continue
            out[y0:y1, x0:x1] = im[y0 - py:y1 - py, x0 - px:x1 - px]
            cov[y0:y1, x0:x1] = True
    return out, cov


# -------------------------------------------------------------- geometry ----

def boxmean(a, r):
    p = np.pad(a, r + 1, mode="edge")
    ii = p.cumsum(0).cumsum(1)
    H, W = a.shape
    k = 2 * r + 1
    s = ii[k:k + H, k:k + W] - ii[0:H, k:k + W] - ii[k:k + H, 0:W] + ii[0:H, 0:W]
    return s / (k * k)


def integral(a):
    return np.pad(a.astype(np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def rectsum(II, x0, y0, x1, y1):
    x0 = np.clip(x0, 0, II.shape[1] - 1); x1 = np.clip(x1, 0, II.shape[1] - 1)
    y0 = np.clip(y0, 0, II.shape[0] - 1); y1 = np.clip(y1, 0, II.shape[0] - 1)
    return (II[y1, x1] - II[y0, x1] - II[y1, x0] + II[y0, x0],
            np.maximum((x1 - x0) * (y1 - y0), 1))


def rasterize(rings, W, H):
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    for r in rings:
        pts = [((r[i] - WX0) * F, (r[i + 1] - WY0) * F) for i in range(0, len(r), 2)]
        if len(pts) >= 3:
            d.polygon(pts, fill=255)
    return np.asarray(m) > 0


def mabr(px, py):
    """minimum-area bounding rectangle of an open ring -> (cx, cy, angle, w, h), w>=h"""
    pts = np.stack([px, py], 1)
    best = None
    for i in range(len(pts)):
        e = pts[(i + 1) % len(pts)] - pts[i]
        L = math.hypot(*e)
        if L < 1e-9:
            continue
        c, s = e[0] / L, e[1] / L
        R = np.array([[c, s], [-s, c]])
        q = pts @ R.T
        x0, x1 = q[:, 0].min(), q[:, 0].max()
        y0, y1 = q[:, 1].min(), q[:, 1].max()
        w, h = x1 - x0, y1 - y0
        if best is None or w * h < best[0]:
            cen = np.array([(x0 + x1) / 2, (y0 + y1) / 2]) @ R
            best = (w * h, cen[0], cen[1], math.atan2(s, c), w, h)
    _, cx, cy, a, w, h = best
    if h > w:
        w, h = h, w
        a += math.pi / 2
    return cx, cy, (a + math.pi / 2) % math.pi - math.pi / 2, w, h


def rect_ring(cx, cy, ang, w, h):
    c, s = math.cos(ang), math.sin(ang)
    out = []
    for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        out += [cx + dx * c - dy * s, cy + dx * s + dy * c]
    return out + out[:2]


# ------------------------------------------------------------ roof score ----

def roof_llr(rgb, inb, big_mask, Q=16):
    """log-likelihood ratio of colour under roof vs ground, learned from the data"""
    core = boxmean(big_mask.astype(np.float32), int(1.2 * F)) > 0.999
    out = ~(boxmean(inb.astype(np.float32), int(2.5 * F)) > 1e-6)
    q = (rgb.astype(np.int32) * Q) // 256
    idx = (q[..., 0] * Q + q[..., 1]) * Q + q[..., 2]

    def hist(m):
        h = np.bincount(idx[m].ravel(), minlength=Q ** 3).astype(float).reshape(Q, Q, Q)
        for ax in (0, 1, 2):
            h = h + np.roll(h, 1, ax) + np.roll(h, -1, ax)
        return (h / h.sum()).ravel()

    llr = np.log((hist(core) + 3e-6) / (hist(out) + 3e-6))
    return np.clip(llr[idx], -3, 3).astype(np.float32) / 3.0


# ------------------------------------------- exact euclidean distance map ----

def _dt1(f):
    n = f.shape[-1]
    m = f.reshape(-1, n).astype(np.float64)
    R = m.shape[0]
    d = np.empty_like(m)
    v = np.zeros((R, n), np.int64)
    z = np.empty((R, n + 1)); z[:, 0] = -np.inf; z[:, 1] = np.inf
    k = np.zeros(R, np.int64)
    idx = np.arange(R)
    for q in range(1, n):
        s = ((m[:, q] + q * q) - (m[idx, v[idx, k]] + v[idx, k].astype(float) ** 2)) / (2.0 * q - 2.0 * v[idx, k])
        while True:
            bad = s <= z[idx, k]
            if not bad.any():
                break
            ib = idx[bad]
            k[bad] -= 1
            kb = k[bad]
            s[bad] = ((m[ib, q] + q * q) - (m[ib, v[ib, kb]] + v[ib, kb].astype(float) ** 2)) / (2.0 * q - 2.0 * v[ib, kb])
        k += 1
        v[idx, k] = q; z[idx, k] = s; z[idx, k + 1] = np.inf
    k[:] = 0
    for q in range(n):
        while True:
            adv = z[idx, k + 1] < q
            if not adv.any():
                break
            k[adv] += 1
        d[:, q] = (q - v[idx, k]) ** 2 + m[idx, v[idx, k]]
    return d.reshape(f.shape)


def edt(mask):
    f = np.where(mask, 0.0, 1e12)
    return np.sqrt(_dt1(np.ascontiguousarray(_dt1(f).T)).T)


# ----------------------------------------------------------------- model ----

class Model:
    """all imagery-derived maps for the refinement area"""

    def __init__(self, verbose=True):
        t = time.time()
        self.rgb, self.cov = build_mosaic(BOX[0] - PAD, BOX[1] - PAD, BOX[2] + PAD, BOX[3] + PAD)
        self.H, self.W = self.rgb.shape[:2]
        a = self.rgb.astype(np.float32) / 255.0
        self.L = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        Ls = np.asarray(Image.fromarray((self.L * 255).astype(np.uint8))
                        .filter(ImageFilter.GaussianBlur(0.25 * F)), np.float32) / 255.
        gy, gx = np.gradient(Ls)
        self.g90 = float(np.percentile(np.hypot(gx, gy), 90))

        osm = json.load(open(OSM_PATH))
        rings = [b["p"] for b in osm["buildings"]]
        big = [p for p in rings
               if min(max(p[0::2]) - min(p[0::2]), max(p[1::2]) - min(p[1::2])) > 4.0]
        inb = rasterize(rings, self.W, self.H)
        self.S = roof_llr(self.rgb, inb, rasterize(big, self.W, self.H))

        Lb = np.asarray(Image.fromarray((np.clip(self.L, 0, 1) * 255).astype(np.uint8))
                        .filter(ImageFilter.GaussianBlur(1.5)), np.float32) / 255.
        by, bx = np.gradient(Lb)
        strong = np.hypot(bx, by) > np.percentile(np.hypot(bx, by), 88)
        self.Dv = edt(strong & (np.abs(bx) >= np.abs(by)))
        self.Dh = edt(strong & (np.abs(bx) < np.abs(by)))
        if verbose:
            print("model built in %.1fs  mosaic %dx%d  coverage %.0f%%"
                  % (time.time() - t, self.W, self.H, 100 * self.cov.mean()))

    def covered(self, cx, cy, r):
        x0, y0 = int((cx - r - WX0) * F), int((cy - r - WY0) * F)
        x1, y1 = int((cx + r - WX0) * F), int((cy + r - WY0) * F)
        if x0 < 0 or y0 < 0 or x1 >= self.W or y1 >= self.H:
            return False
        return bool(self.cov[y0:y1, x0:x1].all())

    # ---- chamfer objective ----
    def chamfer_fit(self, cx, cy, ang0, w, h, ms=MAX_SHIFT_PX, angs=(0.0,), cap=6.0, wd=0.02):
        pu, pv, fl = [], [], []
        nu = max(2, int(w * F / 2)); nv = max(2, int(h * F / 2))
        for t in np.linspace(-w / 2, w / 2, nu):
            pu += [t, t]; pv += [-h / 2, h / 2]; fl += [1, 1]
        for t in np.linspace(-h / 2, h / 2, nv):
            pu += [-w / 2, w / 2]; pv += [t, t]; fl += [0, 0]
        pu, pv, fl = np.array(pu), np.array(pv), np.array(fl)
        o = np.arange(-ms, ms + 1, 1.0)
        OU, OV = (x.ravel() for x in np.meshgrid(o, o, indexing="ij"))
        disp = np.hypot(OU, OV) / F
        best = None
        for da in angs:
            a = ang0 + da
            c, s = math.cos(a), math.sin(a)
            dx, dy = (pu * c - pv * s) * F, (pu * s + pv * c) * F
            X = np.rint((cx - WX0) * F + dx[None, :] + OU[:, None]).astype(np.int64)
            Y = np.rint((cy - WY0) * F + dy[None, :] + OV[:, None]).astype(np.int64)
            np.clip(X, 0, self.W - 1, out=X); np.clip(Y, 0, self.H - 1, out=Y)
            wv, wh = abs(c), abs(s)
            dv, dh = self.Dv[Y, X], self.Dh[Y, X]
            dd = np.where(fl[None, :] == 0, wv * dv + wh * dh, wh * dv + wv * dh)
            cost = np.minimum(dd, cap).mean(1) / cap + wd * disp
            i = int(np.argmin(cost))
            if best is None or cost[i] < best[0]:
                best = (float(cost[i]), float(da), float(OU[i]), float(OV[i]))
        cost, da, bu, bv = best
        return dict(cost=cost, dang=da, cx=cx + bu / F, cy=cy + bv / F, ang=ang0 + da,
                    w=w, h=h, disp=math.hypot(bu, bv) / F)

    # ---- region objective (kept for --check comparison) ----
    def region_score(self, cx, cy, ang, w, h):
        P = int(math.ceil(max(w, h) * F / 2 + 8 + 6))
        Pe = int(P * 1.45) + 3
        px, py = (cx - WX0) * F, (cy - WY0) * F
        ix, iy = int(round(px)), int(round(py))
        n = 2 * Pe + 1
        sx0, sy0 = max(0, ix - Pe), max(0, iy - Pe)
        sx1, sy1 = min(self.W, ix - Pe + n), min(self.H, iy - Pe + n)
        if sx1 - sx0 < n * 0.55 or sy1 - sy0 < n * 0.55:
            return None
        def cut(A):
            o = np.full((n, n), float(A[sy0:sy1, sx0:sx1].mean()), np.float32)
            o[sy0 - iy + Pe:sy1 - iy + Pe, sx0 - ix + Pe:sx1 - ix + Pe] = A[sy0:sy1, sx0:sx1]
            return o
        rot = lambda A: np.asarray(Image.fromarray(cut(A)).rotate(
            -math.degrees(ang), resample=Image.BILINEAR))[Pe - P:Pe + P + 1, Pe - P:Pe + P + 1]
        Lr, Sr = rot(self.L), rot(self.S)
        gy, gx = np.gradient(Lr)
        IGX, IGY = integral(np.abs(gx)), integral(np.abs(gy))
        IGM, IS = integral((np.abs(gx) + np.abs(gy)) / 2), integral(Sr)
        fx, fy = px - ix, py - iy
        cu, cv = fx * math.cos(ang) + fy * math.sin(ang), -fx * math.sin(ang) + fy * math.cos(ang)
        x0 = int(round(P + cu - w * F / 2)); x1 = int(round(P + cu + w * F / 2))
        y0 = int(round(P + cv - h * F / 2)); y1 = int(round(P + cv + h * F / 2))
        B = 3
        sL, aL = rectsum(IGX, x0 - B, y0, x0 + B, y1)
        sR, aR = rectsum(IGX, x1 - B, y0, x1 + B, y1)
        sT, aT = rectsum(IGY, x0, y0 - B, x1, y0 + B)
        sB, aB = rectsum(IGY, x0, y1 - B, x1, y1 + B)
        gin, ain = rectsum(IGM, x0 + B, y0 + B, x1 - B, y1 - B)
        edge = ((sL + sR + sT + sB) / max(aL + aR + aT + aB, 1) - gin / ain) / self.g90
        Sin, Ain = rectsum(IS, x0, y0, x1, y1)
        Sou, Aou = rectsum(IS, x0 - 8, y0 - 8, x1 + 8, y1 + 8)
        cont = Sin / Ain - (Sou - Sin) / max(Aou - Ain, 1)
        return dict(edge=float(edge), cont=float(cont), score=float(cont + 0.15 * edge))


# ------------------------------------------------------------- refinement ----

def in_box(p):
    return (BOX[0] <= np.mean(p[0::2]) < BOX[2]) and (BOX[1] <= np.mean(p[1::2]) < BOX[3])


def refine_all(model, force=False, angles=False):
    osm = json.load(open(OSM_PATH))
    angs = ANGLES if angles else np.array([0.0])
    out = []
    counts = dict(in_box=0, skipped_small=0, no_coverage=0, already=0, accepted=0, kept=0)
    for b in osm["buildings"]:
        p = b["p"]
        if not in_box(p):
            continue
        counts["in_box"] += 1
        if b.get("r") == 1 and not force:
            counts["already"] += 1; continue
        xs, ys = np.array(p[0::2]), np.array(p[1::2])
        cx, cy, a0, w, h = mabr(xs[:-1], ys[:-1])
        if min(w, h) < MIN_MINOR_DIM:
            counts["skipped_small"] += 1; counts["kept"] += 1; continue
        if not model.covered(cx, cy, max(w, h) / 2 + 4):
            counts["no_coverage"] += 1; counts["kept"] += 1; continue
        fits = [model.chamfer_fit(cx + dx, cy + dy, a0, w, h, angs=angs) for dx, dy in STARTS]
        a = np.array([[f["cx"], f["cy"], f["dang"], f["cost"]] for f in fits])
        m = a[:, :2].mean(0)
        spread = float(np.hypot(*(a[:, :2] - m).T).mean())
        base = model.chamfer_fit(cx, cy, a0, w, h, ms=0, angs=(0.0,), wd=0.0)["cost"]
        best = float(a[:, 3].min())
        shift = float(math.hypot(*(m - [cx, cy])))
        ok = (spread <= AGREE_TOL and MIN_SHIFT <= shift <= MAX_SHIFT
              and best < base * (1 - MIN_GAIN))
        rec = dict(b=b, cx=cx, cy=cy, ang=a0, w=w, h=h, spread=spread, shift=shift,
                   cost_before=base, cost_after=best, dang=float(np.median(a[:, 2])),
                   ncx=float(m[0]), ncy=float(m[1]), accepted=ok)
        counts["accepted" if ok else "kept"] += 1
        out.append(rec)
    return osm, out, counts


# ----------------------------------------------------------- diagnostics ----

def check(model, n=200):
    osm = json.load(open(OSM_PATH))
    B = []
    for b in osm["buildings"]:
        p = b["p"]
        if not in_box(p):
            continue
        xs, ys = np.array(p[0::2]), np.array(p[1::2])
        cx, cy, a0, w, h = mabr(xs[:-1], ys[:-1])
        if min(w, h) >= MIN_MINOR_DIM:
            B.append((cx, cy, a0, w, h))
    B = B[:n]
    print("\nMULTI-START STABILITY  (%d town-core footprints, 7 starts each)" % len(B))
    for name, angs in (("translation only", np.array([0.0])), ("translation+rotation", ANGLES)):
        sp, bi = [], []
        for cx, cy, a0, w, h in B:
            r = np.array([[f["cx"], f["cy"]] for f in
                          (model.chamfer_fit(cx + dx, cy + dy, a0, w, h, angs=angs) for dx, dy in STARTS)])
            m = r.mean(0)
            sp.append(np.hypot(*(r - m).T).mean()); bi.append(math.hypot(*(m - [cx, cy])))
        sp, bi = np.array(sp), np.array(bi)
        print("  %-22s spread %.2f wpx (%.1f m)   agree<0.25 wpx %.0f%%   |bias| %.2f wpx"
              % (name, sp.mean(), sp.mean() * M_PER_WPX, 100 * np.mean(sp <= AGREE_TOL), bi.mean()))
    # bulk offset of the outlines against the image gradient
    rings = [b["p"] for b in osm["buildings"]]
    m = Image.new("L", (model.W, model.H), 0)
    d = ImageDraw.Draw(m)
    for r in rings:
        pts = [((r[i] - WX0) * F, (r[i + 1] - WY0) * F) for i in range(0, len(r), 2)]
        if len(pts) >= 3:
            d.line(pts + [pts[0]], fill=255, width=3)
    O = np.asarray(m, np.float32) / 255.
    gy, gx = np.gradient(model.L)
    G = np.hypot(gx, gy)
    A, Bm = O - O.mean(), G - G.mean()
    cc = np.fft.fftshift(np.fft.irfft2(np.fft.rfft2(Bm) * np.conj(np.fft.rfft2(A)), s=G.shape))
    MS = 32; cy0, cx0 = G.shape[0] // 2, G.shape[1] // 2
    win = cc[cy0 - MS:cy0 + MS + 1, cx0 - MS:cx0 + MS + 1]
    iy, ix = np.unravel_index(np.argmax(win), win.shape)
    print("  bulk outline-vs-gradient offset: %+.2f, %+.2f wpx (%+.1f, %+.1f m), peak/zero %.3f"
          % ((ix - MS) / F, (iy - MS) / F, (ix - MS) / F * M_PER_WPX, (iy - MS) / F * M_PER_WPX,
             win.max() / win[MS, MS]))


# -------------------------------------------------------------- previews ----

def previews(model, recs):
    os.makedirs(PREVIEWS, exist_ok=True)
    osm = json.load(open(OSM_PATH))
    others = [b["p"] for b in osm["buildings"] if not in_box(b["p"])]
    orig = [r["b"]["p"] for r in recs]
    cand = [rect_ring(r["ncx"], r["ncy"], r["ang"] + r["dang"], r["w"], r["h"]) for r in recs]
    acc = [rect_ring(r["ncx"], r["ncy"], r["ang"] + r["dang"], r["w"], r["h"])
           for r in recs if r["accepted"]]
    views = [(1, "dense bazaar", 21000, 19680, 60, 1.0),
             (2, "north (campus / bus station)", 21080, 19580, 60, 1.0),
             (3, "south row along the highway", 20990, 19760, 60, 1.0),
             (4, "riverside row, 3x", 21006, 19772, 14, 3.0)]
    paths = []
    for n, label, cx, cy, half, sc in views:
        x0, y0 = int((cx - half - WX0) * F), int((cy - half - WY0) * F)
        npx = int(2 * half * F)
        im = Image.fromarray(model.rgb[y0:y0 + npx, x0:x0 + npx]).convert("RGB")
        d = ImageDraw.Draw(im)
        for polys, col, wdt in ((others, (90, 90, 200), 1), (orig, (255, 40, 40), 2),
                                (cand, (255, 210, 60), 1), (acc, (60, 255, 60), 2)):
            for p in polys:
                pts = [((p[i] - WX0) * F - x0, (p[i + 1] - WY0) * F - y0) for i in range(0, len(p), 2)]
                d.line(pts + [pts[0]], fill=col, width=wdt)
        if sc != 1.0:
            im = im.resize((int(npx * sc),) * 2, Image.LANCZOS)
        d = ImageDraw.Draw(im)
        d.text((8, 8), "%s  -  red: current OSM   yellow: candidate fit   lime: accepted"
               % label, fill=(255, 255, 255))
        fn = os.path.join(PREVIEWS, "footprints_refine_%d.png" % n)
        im.save(fn); paths.append(fn)
    return paths


# ------------------------------------------------------------------ main ----

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="print the reliability diagnostic")
    ap.add_argument("--previews", action="store_true", help="write previews/footprints_refine_*.png")
    ap.add_argument("--write", action="store_true", help="apply the accepted refinements to osm.json")
    ap.add_argument("--force", action="store_true", help="re-refine footprints already marked r=1")
    ap.add_argument("--angles", action="store_true", help="also search rotation (default: translation only)")
    a = ap.parse_args()

    model = Model()
    if a.check:
        check(model)
    osm, recs, counts = refine_all(model, force=a.force, angles=a.angles)
    acc = [r for r in recs if r["accepted"]]
    print("\nin box %(in_box)d | too small %(skipped_small)d | no coverage %(no_coverage)d | "
          "already refined %(already)d | accepted %(accepted)d | kept %(kept)d" % counts)
    if acc:
        s = np.array([r["shift"] for r in acc])
        print("accepted shift: median %.2f wpx (%.1f m), p90 %.2f wpx"
              % (np.median(s), np.median(s) * M_PER_WPX, np.percentile(s, 90)))
    if a.previews:
        for p in previews(model, recs):
            print("wrote", p)
    if not a.write:
        print("\nDRY RUN - osm.json not modified.  See VERDICT in this file's docstring;"
              "\npass --write only if a visual check says the refinement helps.")
        return
    for r in acc:
        b = r["b"]
        ring = rect_ring(r["ncx"], r["ncy"], r["ang"] + r["dang"], r["w"], r["h"])
        b["p"] = [round(v, 1) for v in ring]
        b["r"] = 1
    osm.setdefault("meta", {})["refined"] = dict(
        date="2026-09-05", area=list(BOX), count=len(acc),
        method="min-area-rectangle re-fit to the Vantor Legion 5 Feb 2026 z8 mosaic by "
               "oriented chamfer matching, accepted only on 7-start agreement <=0.25 world px")
    json.dump(osm, open(OSM_PATH, "w"), ensure_ascii=False, separators=(",", ":"))
    print("\nwrote %s (%d footprints refined)" % (OSM_PATH, len(acc)))


if __name__ == "__main__":
    main()
