#!/usr/bin/env python3
"""Re-derive the Trisuli balancing-reservoir outline and the headrace-canal water surface
from the Vantor Legion (5 Feb 2026) pre-flood tile pyramid, and patch them into rivers.json.

OpenStreetMap gets the reservoir noticeably wrong (its ring sits ~10 world px NE of the real
water) and stops mapping the headrace canal north of the reservoir after a couple of short
stubs.  Both features are segmented straight off the imagery here.

Usage:
  python3 scripts/fix_water.py [--root DIR] [--rivers PATH] [--overpass PATH]
                               [--out PATH] [--previews DIR] [--report]

World pixel space:  x = (lon-84.52)/(85.60-84.52)*36169 ; y = (28.45-lat)/(28.45-27.70)*28011
1 world px ~ 2.93 m.  Focus box: x 20435.5..21899.0, y 18685.2..20481.6.
"""
import argparse, json, math, os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

BOX = (20435.5, 18685.2, 21899.0, 20481.6)
WORLD_PX_M = 2.93

# --- windows (world px) and pyramid level used for each feature -------------------------
RES_WIN = (20960.0, 18860.0, 21420.0, 19280.0)   # balancing reservoir + ~200 m margin
RES_F = 4
CAN_WIN = (21285.0, 18690.0, 21545.0, 18995.0)   # headrace canal: tunnel portal -> reservoir
CAN_F = 8

# --- water thresholds (calibrated on this scene; see --report) ---------------------------
BR_MIN = 0.95        # B/R: water >= ~1.05, vegetation/soil/gravel/roads <= ~0.90
RNG_MAX_RES = 14.0   # local luminance range over a 5x5 image-px window (open water is smooth)
RNG_MAX_CAN = 18.0
GAP_CANAL = 5.0      # world px the canal component chain may jump (road bridges, culverts)   # canal is narrower, so its window sees more bank contamination


# ============================ marching squares / simplify ===============================
_EDGE = {"N": (0.5, 0.0), "E": (1.0, 0.5), "S": (0.5, 1.0), "W": (0.0, 0.5)}
_TBL = {
    1: [("S", "W")], 2: [("E", "S")], 3: [("E", "W")], 4: [("N", "E")],
    5: [("N", "W"), ("S", "E")], 6: [("N", "S")], 7: [("N", "W")], 8: [("W", "N")],
    9: [("S", "N")], 10: [("E", "N"), ("W", "S")], 11: [("E", "N")], 12: [("W", "E")],
    13: [("S", "E")], 14: [("W", "S")],
}


def trace_rings(mask):
    """Closed rings of a boolean mask, as (n,2) arrays in pixel-index space."""
    m = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8)
    m[1:-1, 1:-1] = mask.astype(np.uint8)
    code = (m[:-1, :-1] << 3) | (m[:-1, 1:] << 2) | (m[1:, 1:] << 1) | m[1:, :-1]
    nxt = {}
    for k, segs in _TBL.items():
        ii, jj = np.nonzero(code == k)
        if ii.size == 0:
            continue
        for a, b in segs:
            ax, ay = _EDGE[a]
            bx, by = _EDGE[b]
            ka = (np.rint((jj + ax) * 2).astype(np.int64) << 24) + np.rint((ii + ay) * 2).astype(np.int64)
            kb = (np.rint((jj + bx) * 2).astype(np.int64) << 24) + np.rint((ii + by) * 2).astype(np.int64)
            for u, v in zip(ka.tolist(), kb.tolist()):
                nxt[u] = v
    rings, seen = [], set()
    for start in list(nxt.keys()):
        if start in seen:
            continue
        pts, cur = [], start
        while cur in nxt and cur not in seen:
            seen.add(cur)
            pts.append(cur)
            cur = nxt[cur]
        if cur != start or len(pts) < 4:
            continue
        arr = np.empty((len(pts), 2), float)
        for i, k in enumerate(pts):
            arr[i, 0] = (k >> 24) / 2.0 - 1.0
            arr[i, 1] = (k & 0xFFFFFF) / 2.0 - 1.0
        rings.append(arr)
    return rings


def ring_area(r):
    x, y = r[:, 0], r[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def rdp(pts, eps):
    n = len(pts)
    if n < 3:
        return pts
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        p, q = pts[i], pts[j]
        d = q - p
        L = float(np.hypot(d[0], d[1]))
        seg = pts[i + 1:j]
        if L < 1e-9:
            dist = np.hypot(seg[:, 0] - p[0], seg[:, 1] - p[1])
        else:
            dist = np.abs(d[0] * (p[1] - seg[:, 1]) - d[1] * (p[0] - seg[:, 0])) / L
        k = int(np.argmax(dist))
        if dist[k] > eps:
            k += i + 1
            keep[k] = True
            stack.append((i, k))
            stack.append((k, j))
    return pts[keep]


def rdp_ring(r, eps):
    if len(r) < 6:
        return r
    i0 = int(np.argmin(r[:, 0] + r[:, 1]))
    r = np.roll(r, -i0, axis=0)
    i1 = int(np.argmax(np.hypot(r[:, 0] - r[0, 0], r[:, 1] - r[0, 1])))
    a = rdp(r[:i1 + 1], eps)
    b = rdp(np.vstack([r[i1:], r[:1]]), eps)
    return np.vstack([a[:-1], b[:-1]])


# ================================ tiles / imagery ======================================
def load_level(root, layer, F):
    idx = json.load(open(f"{root}/assets/trisuli/tiles/{layer}/index.json"))
    for l in idx["levels"]:
        if l["F"] == F:
            return l
    raise SystemExit(f"level F={F} not in {layer}")


def mosaic(root, layer, F, win):
    wx0, wy0, wx1, wy1 = win
    lv = load_level(root, layer, F)
    T, d = lv["T"], lv["dir"]
    td = f"{root}/assets/trisuli/tiles/{layer}/{d}"
    W, H = int(round((wx1 - wx0) * F)), int(round((wy1 - wy0) * F))
    im = Image.new("RGB", (W, H), (0, 0, 0))
    keys, n = set(lv["tiles"]), 0
    for tx in range(math.floor(wx0 / T), math.floor(wx1 / T) + 1):
        for ty in range(math.floor(wy0 / T), math.floor(wy1 / T) + 1):
            k = f"{tx}_{ty}"
            if k not in keys:
                continue
            p = f"{td}/{k}.webp"
            if not os.path.exists(p):
                continue
            im.paste(Image.open(p).convert("RGB"),
                     (int(round((tx * T - wx0) * F)), int(round((ty * T - wy0) * F))))
            n += 1
    if n == 0:
        raise SystemExit(f"no tiles for {layer} F={F} in {win}")
    return im, n


def water_mask(im, rng_max, rng_win=5):
    """Opaque still water: strongly blue-shifted relative to red, and texturally smooth."""
    a = np.asarray(im, float)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    lum = Image.fromarray((0.3 * R + 0.6 * G + 0.1 * B).astype(np.uint8))
    rng = (np.asarray(lum.filter(ImageFilter.MaxFilter(rng_win)), float)
           - np.asarray(lum.filter(ImageFilter.MinFilter(rng_win)), float))
    return ((B / (R + 1e-6)) > BR_MIN) & (rng < rng_max) & (a.sum(-1) > 12)


def _shift_or(a, k, axis):
    out = a.copy()
    if axis == 0:
        if k > 0:
            out[k:] |= a[:-k]
            out[:-k] |= a[k:]
        return out
    if k > 0:
        out[:, k:] |= a[:, :-k]
        out[:, :-k] |= a[:, k:]
    return out


def dilate(mask, r):
    """Binary dilation by a (2r+1)-square, O(log r) array passes."""
    if r <= 0:
        return mask
    out = mask
    for axis in (0, 1):
        cur, k = 0, 1
        while cur < r:
            k = min(k, r - cur)
            out = _shift_or(out, k, axis)
            cur += k
            k *= 2
    return out


def erode(mask, r):
    return ~dilate(~mask, r)


def morph(mask, close=0, open_=0):
    """close/open radii in image pixels."""
    m = mask
    if close:
        m = erode(dilate(m, close), close)
    if open_:
        m = dilate(erode(m, open_), open_)
    return m


def components(mask):
    """8-connected labelling of a boolean mask via run-length union-find.
    Returns (labels int32, n) with 0 = background and labels 1..n."""
    H, W = mask.shape
    pad = np.zeros((H, W + 2), bool)
    pad[:, 1:-1] = mask
    d = np.diff(pad.astype(np.int8), axis=1)
    starts = np.argwhere(d == 1)
    ends = np.argwhere(d == -1)
    rows = starts[:, 0]
    s0 = starts[:, 1]
    e0 = ends[:, 1]                      # run covers columns [s0, e0)
    row_first = np.searchsorted(rows, np.arange(H))
    row_last = np.searchsorted(rows, np.arange(H), side="right")
    parent = list(range(len(rows)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(1, H):
        a0, a1 = row_first[i - 1], row_last[i - 1]
        b0, b1 = row_first[i], row_last[i]
        p = a0
        for q in range(b0, b1):
            while p < a1 and e0[p] < s0[q]:      # 8-connected: touch counts
                p += 1
            k = p
            while k < a1 and s0[k] <= e0[q]:
                union(k, q)
                k += 1
    lab = np.zeros((H, W), np.int32)
    remap, nxt = {}, 0
    for idx in range(len(rows)):
        r = find(idx)
        if r not in remap:
            nxt += 1
            remap[r] = nxt
        lab[rows[idx], s0[idx]:e0[idx]] = remap[r]
    return lab, nxt


def grow(seed, mask, gap=0, min_area=0):
    """Union of the mask components reachable from `seed`, allowing jumps of up to `gap`
    pixels between components (bridges and culverts break an otherwise continuous canal).
    Components smaller than `min_area` pixels are never chained through."""
    lab, n = components(mask)
    if min_area:
        keep = np.bincount(lab.ravel(), minlength=n + 1) >= min_area
        keep[0] = False
        lab = np.where(keep[lab], lab, 0)
    cur = np.isin(lab, np.unique(lab[seed & (lab > 0)]))
    if not cur.any():
        return np.zeros_like(mask)
    while True:
        ids = np.unique(lab[dilate(cur, gap) & (lab > 0)]) if gap else np.unique(lab[cur])
        ids = ids[ids > 0]
        nxt = np.isin(lab, ids)
        if nxt.sum() == cur.sum():
            return nxt
        cur = nxt


def fill_holes(comp):
    """comp: bool. Returns (filled, holes)."""
    lab, n = components(~comp)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    border = border[border > 0]
    holes = (lab > 0) & ~np.isin(lab, border)
    return comp | holes, holes


def best_shift(base, mask, rmax):
    """Integer (dx, dy) maximising |shift(base, dx, dy) & mask|, searched over +-rmax px."""
    H, W = mask.shape
    A = np.fft.rfft2(mask.astype(np.float32))
    B = np.fft.rfft2(base.astype(np.float32))
    c = np.fft.irfft2(A * np.conj(B), s=(H, W))
    ys = np.concatenate([np.arange(0, rmax + 1), np.arange(H - rmax, H)])
    xs = np.concatenate([np.arange(0, rmax + 1), np.arange(W - rmax, W)])
    sub = c[np.ix_(ys, xs)]
    k = int(np.argmax(sub))
    iy, ix = divmod(k, sub.shape[1])
    dy, dx = int(ys[iy]), int(xs[ix])
    if dy > H // 2:
        dy -= H
    if dx > W // 2:
        dx -= W
    return dx, dy


# ================================ vector helpers =======================================
def raster_poly(rings, win, F, size, width=None, closed=True):
    wx0, wy0 = win[0], win[1]
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    for r in rings:
        pts = [((r[i] - wx0) * F, (r[i + 1] - wy0) * F) for i in range(0, len(r), 2)]
        if len(pts) < 2:
            continue
        if width:
            d.line(pts + ([pts[0]] if closed else []), fill=255, width=width)
        elif len(pts) >= 3:
            d.polygon(pts, fill=255)
    return np.asarray(m) > 0


def px_to_world(r, win, F):
    out = np.empty_like(r)
    out[:, 0] = win[0] + (r[:, 0] + 0.5) / F
    out[:, 1] = win[1] + (r[:, 1] + 0.5) / F
    return out


def flat(r, nd=1):
    return [round(float(v), nd) for p in r for v in p]


def point_in_ring(x, y, ring):
    xs, ys = ring[0::2], ring[1::2]
    n, inside = len(xs), False
    j = n - 1
    for i in range(n):
        if (ys[i] > y) != (ys[j] > y) and x < (xs[j] - xs[i]) * (y - ys[i]) / (ys[j] - ys[i] + 1e-12) + xs[i]:
            inside = not inside
        j = i
    return inside


def poly_dist_frac(line, ref, tol):
    """Fraction of `line`'s vertices within `tol` world px of polyline `ref`."""
    a = np.array(ref, float).reshape(-1, 2)
    p = np.array(line, float).reshape(-1, 2)
    seg0, seg1 = a[:-1], a[1:]
    d = seg1 - seg0
    L2 = (d ** 2).sum(1) + 1e-12
    hit = 0
    for q in p:
        t = np.clip(((q - seg0) * d).sum(1) / L2, 0, 1)
        proj = seg0 + t[:, None] * d
        if np.hypot(*(q - proj).T).min() <= tol:
            hit += 1
    return hit / max(1, len(p))


# ============================== feature identification =================================
def bbox(ring):
    xs, ys = ring[0::2], ring[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def find_label(labels, name):
    for i, L in enumerate(labels):
        if L[3] == name:
            return i
    return None


# =================================== reservoir =========================================
def do_reservoir(root, riv, report):
    win, F = RES_WIN, RES_F
    im, ntiles = mosaic(root, "vantor_pre", F, win)
    size = im.size
    mask = morph(water_mask(im, RNG_MAX_RES, 5), close=3, open_=5)

    li = find_label(riv["labels"], "Balancing Reservoir")
    lab = riv["labels"][li]
    osm_i = None
    for i, p in enumerate(riv["polys"]):
        if len(p) >= 8 and point_in_ring(lab[0], lab[1], p):
            b = bbox(p)
            if osm_i is None or (b[2] - b[0]) * (b[3] - b[1]) < 200000:
                osm_i = i
    if osm_i is None:
        raise SystemExit("reservoir polygon not found in rivers.json")
    osm = riv["polys"][osm_i]

    # No corridor constraint: the previous outline is only a prior for identification, never a
    # boundary.  Segment the whole window freely and take the largest water body that is not the
    # Trishuli; the reservoir is ~1 km NE-SW and an order of magnitude bigger than anything else.
    river = max(riv["polys"], key=lambda p: (lambda b: (b[2] - b[0]) * (b[3] - b[1]))(bbox(p)))
    riv_r = dilate(raster_poly([river], win, F, size), int(2 * F))
    mask = mask & ~riv_r

    lab, nlab = components(mask)
    areas = np.bincount(lab.ravel(), minlength=nlab + 1)
    areas[0] = 0
    main = int(np.argmax(areas))
    comp = lab == main

    # rigid offset of the previous outline from the water, searched over +-60 world px
    base = raster_poly([osm], win, F, size)
    dx, dy = best_shift(base, comp, 60 * F)
    inside = np.roll(np.roll(base, dy, 0), dx, 1)

    # spits split the reservoir into a main basin and marginal channels; add any other water
    # body that lies mostly within the (shifted) prior footprint
    for i in range(1, nlab + 1):
        if i == main or areas[i] < 20 * F * F:
            continue
        m = lab == i
        if (m & inside).sum() / areas[i] >= 0.5:
            comp = comp | m
    comp = grow(comp, mask, gap=int(2 * F), min_area=int(6 * F * F))

    filled, holes = fill_holes(comp)

    rings = [r for r in trace_rings(filled) if abs(ring_area(r)) / (F * F) >= 100]
    rings.sort(key=lambda r: -abs(ring_area(r)))
    outer = rings[0]
    # The overlay is filled, so the ring set is kept hole-free on purpose: submerged shoals and
    # the two small vegetated islets are left inside the water surface.
    nouter = len(rings)
    dropped = int(holes.sum() / (F * F))

    out = [px_to_world(rdp_ring(r, 0.9 * F), win, F) for r in rings]
    if report:
        print(f"[reservoir] tiles={ntiles} win={win} F={F}")
        print(f"  OSM poly idx={osm_i} n={len(osm)//2}")
        print(f"  previous outline -> water, best rigid shift = ({dx/F:+.1f}, {dy/F:+.1f}) world px "
              f"= ({dx/F*WORLD_PX_M:+.0f}, {dy/F*WORLD_PX_M:+.0f}) m  (+x east, +y south)")
        pa = raster_poly([osm], win, F, size)
        na = raster_poly([flat(out[0], 2)], win, F, size)
        ha = lambda m: m.sum() / (F * F) * WORLD_PX_M ** 2 / 1e4
        print(f"  previous {ha(pa):.2f} ha vs traced main basin {ha(na):.2f} ha; "
              f"previous over dry ground {ha(pa & ~na):.2f} ha, water it missed {ha(na & ~pa):.2f} ha")
        o = np.array(osm, float).reshape(-1, 2)
        s0 = out[0]; s1 = np.roll(s0, -1, 0); dd = s1 - s0; L2 = (dd ** 2).sum(1) + 1e-12
        dist = np.array([np.hypot(*(q - (s0 + np.clip(((q - s0) * dd).sum(1) / L2, 0, 1)[:, None] * dd)).T).min()
                         for q in o]) * WORLD_PX_M
        print(f"  previous vertex -> traced shoreline: median {np.median(dist):.0f} m, "
              f"p90 {np.percentile(dist,90):.0f} m, max {dist.max():.0f} m")
        print(f"  traced water = {sum(abs(ring_area(r)) for r in rings)/(F*F)*WORLD_PX_M**2/1e4:.1f} ha "
              f"in {nouter} exterior ring(s), no holes; verts={[len(r) for r in out]}")
        print(f"  shoals/islets left inside the outline rather than cut out: {dropped} world px2")
    return dict(win=win, F=F, im=im, mask=mask, comp=filled, osm_i=osm_i, osm=osm,
                rings=out, shift=(dx / F, dy / F), label_i=li)


# ===================================== canal ===========================================
def do_canal(root, riv, res, report):
    win, F = CAN_WIN, CAN_F
    im, ntiles = mosaic(root, "vantor_pre", F, win)
    size = im.size
    mask = morph(water_mask(im, RNG_MAX_CAN, 9), close=2)

    river = max(riv["polys"], key=lambda p: (lambda b: (b[2] - b[0]) * (b[3] - b[1]))(bbox(p)))
    riv_r = dilate(raster_poly([river], win, F, size), int(3 * F))
    res_r = dilate(raster_poly([flat(res["rings"][0], 2)], win, F, size), int(2 * F))
    mask = mask & ~riv_r & ~res_r

    # seed from the OSM settling basin north-east of the reservoir
    cand = [i for i, p in enumerate(riv["polys"])
            if bbox(p)[0] > win[0] and bbox(p)[2] < win[2]
            and bbox(p)[1] > win[1] and bbox(p)[3] < win[3]]
    if not cand:
        raise SystemExit("no seed polygon for the canal")
    seed_i = max(cand, key=lambda i: (lambda b: (b[2] - b[0]) * (b[3] - b[1]))(bbox(riv["polys"][i])))
    seed = raster_poly([riv["polys"][seed_i]], win, F, size) & mask
    # the canal is broken into pieces by road bridges and short culverts: chain components
    # that come within GAP world px of one another
    comp = grow(seed, mask, gap=int(GAP_CANAL * F), min_area=int(20 * F * F))
    filled, _ = fill_holes(comp)

    rings = [r for r in trace_rings(filled) if abs(ring_area(r)) / (F * F) >= 25]
    rings.sort(key=lambda r: -abs(ring_area(r)))
    out = [px_to_world(rdp_ring(r, 0.8 * F), win, F) for r in rings]

    # centreline: walk down the ribbon row by row, always staying in the run nearest the
    # previous centre, so a side channel or the basin's far bank cannot pull it off course
    ys, xs, prev = [], [], None
    for i in range(filled.shape[0]):
        row = filled[i]
        if not row.any():
            continue
        e = np.diff(np.concatenate([[0], row.view(np.int8), [0]]))
        a = np.nonzero(e == 1)[0]
        b = np.nonzero(e == -1)[0]
        mids = (a + b - 1) / 2.0
        if prev is None:
            k = int(np.argmax(b - a))
        else:
            k = int(np.argmin(np.abs(mids - prev)))
            if abs(mids[k] - prev) > 25 * F:
                continue
        prev = mids[k]
        ys.append(i)
        xs.append(mids[k])
    ys = np.array(ys, float)
    xs = np.array(xs, float)
    full = np.arange(ys.min(), ys.max() + 1)
    xf = np.interp(full, ys, xs)
    k = max(3, int(4 * F) | 1)
    ker = np.ones(k) / k
    xs_s = np.convolve(np.pad(xf, (k // 2, k // 2), mode="edge"), ker, "valid")
    cl = np.stack([xs_s, full], 1)
    cl = px_to_world(cl, win, F)
    cl = rdp(cl, 0.6)
    # start it at the box top edge, end it on the reservoir shoreline
    resring = flat(res["rings"][0], 2)
    keep = [p for p in cl if not point_in_ring(p[0], p[1], resring)]
    cl = np.array(keep)

    if report:
        print(f"[canal] tiles={ntiles} win={win} F={F}")
        print(f"  seed poly idx={seed_i}")
        print(f"  water rings={len(out)} verts={[len(r) for r in out]} "
              f"area={sum(abs(ring_area(r)) for r in rings)/(F*F)*WORLD_PX_M**2/1e4:.2f} ha")
        print(f"  centreline verts={len(cl)} from ({cl[0][0]:.1f},{cl[0][1]:.1f}) "
              f"to ({cl[-1][0]:.1f},{cl[-1][1]:.1f}) "
              f"length={np.hypot(*np.diff(cl,axis=0).T).sum()*WORLD_PX_M:.0f} m")
    return dict(win=win, F=F, im=im, mask=mask, comp=filled, rings=out, line=cl, seed_i=seed_i)


# ==================================== patching =========================================
def patch(riv, res, can, report):
    polys, lines, labels = riv["polys"], riv["lines"], riv["labels"]
    newring = flat(res["rings"][0], 1)

    drop_p = {res["osm_i"], can["seed_i"]}
    # anything the previous file had inside the reservoir footprint is superseded
    for i, p in enumerate(polys):
        if i in drop_p or i == res["osm_i"]:
            continue
        b = bbox(p)
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if point_in_ring(cx, cy, polys[res["osm_i"]]) or point_in_ring(cx, cy, newring):
            drop_p.add(i)
    # any other polygon swallowed by the traced canal ribbon
    cw, cF = can["win"], can["F"]
    csize = can["comp"].shape[1], can["comp"].shape[0]
    for i, p in enumerate(polys):
        if i in drop_p:
            continue
        b = bbox(p)
        if b[0] < cw[0] or b[2] > cw[2] or b[1] < cw[1] or b[3] > cw[3]:
            continue
        r = raster_poly([p], cw, cF, csize)
        if r.any() and (r & can["comp"]).sum() / r.sum() > 0.4:
            drop_p.add(i)

    kept = [p for i, p in enumerate(polys) if i not in drop_p]
    added = [flat(r, 1) for r in res["rings"]] + [flat(r, 1) for r in can["rings"]]
    riv["polys"] = kept + added

    # canal centrelines superseded by the traced one
    ref = flat(can["line"], 2)
    # (named lines are hand-placed, e.g. the "aqueduct" carrying the Pasang Murrfu canal
    # across the Trishuli to the tunnel portal, and are always kept)
    drop_l = [i for i, L in enumerate(lines)
              if L["c"] == "c" and not L.get("n") and poly_dist_frac(L["p"], ref, 12.0) >= 0.5]
    riv["lines"] = [L for i, L in enumerate(lines) if i not in drop_l]
    riv["lines"].append({"c": "c", "p": flat(can["line"], 1)})

    # reservoir label -> centroid of the new outline
    r = res["rings"][0]
    A = ring_area(r)
    x, y = r[:, 0], r[:, 1]
    xr, yr = np.roll(x, -1), np.roll(y, -1)
    cr = x * yr - xr * y
    cx = float((( x + xr) * cr).sum() / (6 * A))
    cy = float((( y + yr) * cr).sum() / (6 * A))
    labels[res["label_i"]][0] = round(cx, 1)
    labels[res["label_i"]][1] = round(cy, 1)

    riv["source"] = ("© OpenStreetMap contributors (water areas, waterways, names); "
                     "Balancing Reservoir and Trishuli headrace canal re-traced from "
                     "Vantor Legion 5 Feb 2026 imagery")
    if report:
        print(f"[patch] dropped polys {sorted(drop_p)}, added {len(added)}; "
              f"dropped canal lines {drop_l}, added 1")
        print(f"  label 'Balancing Reservoir' -> ({cx:.1f}, {cy:.1f})")
    return riv


# ==================================== previews =========================================
def preview(path, root, win, F, before_polys, before_lines, after_polys, after_lines, scale=1.0):
    im, _ = mosaic(root, "vantor_pre", F, win)
    d = ImageDraw.Draw(im)

    def draw(items, col, w, closed):
        for p in items:
            xs, ys = p[0::2], p[1::2]
            if max(xs) < win[0] or min(xs) > win[2] or max(ys) < win[1] or min(ys) > win[3]:
                continue
            pts = [((p[i] - win[0]) * F, (p[i + 1] - win[1]) * F) for i in range(0, len(p), 2)]
            if len(pts) < 2:
                continue
            d.line(pts + ([pts[0]] if closed else []), fill=col, width=w)
    draw(before_polys, (255, 40, 40), max(2, F // 3), True)
    draw(before_lines, (255, 140, 0), max(2, F // 3), False)
    draw(after_polys, (60, 255, 60), max(2, F // 3), True)
    draw(after_lines, (0, 255, 200), max(2, F // 3), False)
    if scale != 1.0:
        im = im.resize((int(im.size[0] * scale), int(im.size[1] * scale)), Image.LANCZOS)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path)
    return im.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--rivers", default=None)
    ap.add_argument("--overpass", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--previews", default=None)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root = a.root
    rivers = a.rivers or f"{root}/assets/trisuli/rivers.json"
    out = a.out or rivers
    prev = a.previews or f"{root}/previews"

    riv = json.load(open(rivers))
    before = json.loads(json.dumps(riv))

    if a.overpass and os.path.exists(a.overpass):
        ov = json.load(open(a.overpass))
        n = sum(1 for e in ov["elements"]
                if e["type"] == "way" and (e.get("tags") or {}).get("waterway") in ("canal", "drain", "ditch"))
        if a.report:
            print(f"[osm] overpass dump has {n} canal/drain/ditch ways")

    res = do_reservoir(root, riv, a.report)
    can = do_canal(root, riv, res, a.report)
    patch(riv, res, can, a.report)

    if not a.dry_run:
        with open(out, "w") as f:
            json.dump(riv, f, separators=(",", ":"), ensure_ascii=False)
        print(f"wrote {out}  polys={len(riv['polys'])} lines={len(riv['lines'])} labels={len(riv['labels'])}")

    bp, bl = before["polys"], [L["p"] for L in before["lines"] if L["c"] == "c"]
    ap_, al = riv["polys"], [L["p"] for L in riv["lines"] if L["c"] == "c"]
    preview(f"{prev}/water_fix_reservoir.png", root, (20990, 18890, 21400, 19240), 4,
            bp, bl, ap_, al, 0.55)
    preview(f"{prev}/water_fix_canal_1.png", root, (21330, 18685.2, 21520, 18860), 8,
            bp, bl, ap_, al, 0.5)
    preview(f"{prev}/water_fix_canal_2.png", root, (21285, 18860, 21430, 19020), 8,
            bp, bl, ap_, al, 0.6)
    # southern headrace: reservoir outlet -> forebay/powerhouse at Trisuli Bazar (unchanged, OSM)
    preview(f"{prev}/water_fix_canal_3.png", root, (20910, 19170, 21090, 19770), 4,
            bp, bl, ap_, al, 0.62)
    print("previews written to", prev)


if __name__ == "__main__":
    main()
