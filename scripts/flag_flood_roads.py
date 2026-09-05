#!/usr/bin/env python3
"""Flag roads and bridges that overlap the mapped flood extent.

Reads assets/trisuli/roads.json ({"roads":[{c,p}], "bridges":[{c,rc,p,n}], ...}) and
assets/trisuli/flood.json ({"path":[flat rings]}, world px), and writes roads.json back with
  - "damaged": sub-polylines of roads (same class "c") whose samples fall inside the flood
    path or within TOL world px of its edge, and
  - "d": 1 on every bridge whose span touches the flood path the same way.
Exposure-based (the road/bridge lies where the post-flood channel was mapped), not a field
assessment. Re-runnable: "damaged" is rebuilt from scratch and "d" flags are reset first.
Usage: python3 scripts/flag_flood_roads.py [--tol 1.5] [--step 1.5]
"""
import argparse, json, math, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROADS = os.path.join(ROOT, "assets", "trisuli", "roads.json")
FLOOD = os.path.join(ROOT, "assets", "trisuli", "flood.json")


def point_in_rings(x, y, rings):
    inside = False
    for r in rings:
        n = len(r) // 2
        j = n - 1
        for i in range(n):
            xi, yi, xj, yj = r[2 * i], r[2 * i + 1], r[2 * j], r[2 * j + 1]
            if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
            j = i
    return inside


def dist_to_rings(x, y, rings, bbs, cutoff):
    best = cutoff
    for r, bb in zip(rings, bbs):
        if x < bb[0] - cutoff or x > bb[2] + cutoff or y < bb[1] - cutoff or y > bb[3] + cutoff:
            continue
        n = len(r) // 2
        for i in range(n):
            ax, ay = r[2 * i], r[2 * i + 1]
            bx, by = r[(2 * i + 2) % (2 * n)], r[(2 * i + 3) % (2 * n)]
            dx, dy = bx - ax, by - ay
            L = dx * dx + dy * dy
            t = 0.0 if L == 0 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / L))
            d = math.hypot(x - (ax + t * dx), y - (ay + t * dy))
            if d < best:
                best = d
    return best


def resample(p, step):
    """Points along the polyline every `step` world px, keeping the original vertices."""
    out = [(p[0], p[1])]
    for i in range(2, len(p), 2):
        ax, ay, bx, by = p[i - 2], p[i - 1], p[i], p[i + 1]
        L = math.hypot(bx - ax, by - ay)
        k = max(1, int(L // step))
        for s in range(1, k + 1):
            t = s / k
            out.append((ax + (bx - ax) * t, ay + (by - ay) * t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1.5, help="edge tolerance in world px (~2.93 m each)")
    ap.add_argument("--step", type=float, default=1.5, help="sampling step along roads, world px")
    a = ap.parse_args()
    roads = json.load(open(ROADS))
    rings = json.load(open(FLOOD))["path"]
    bbs = [(min(r[0::2]), min(r[1::2]), max(r[0::2]), max(r[1::2])) for r in rings]

    def hit(x, y):
        return point_in_rings(x, y, rings) or dist_to_rings(x, y, rings, bbs, a.tol) < a.tol

    damaged = []
    for R in roads["roads"]:
        pts = resample(R["p"], a.step)
        flags = [hit(x, y) for x, y in pts]
        i = 0
        while i < len(pts):
            if flags[i]:
                j = i
                while j + 1 < len(pts) and flags[j + 1]:
                    j += 1
                lo, hi = max(0, i - 1), min(len(pts) - 1, j + 1)   # include the crossing segments
                seg = [round(v, 1) for xy in pts[lo:hi + 1] for v in xy]
                if len(seg) >= 4:
                    damaged.append({"c": R["c"], "p": seg})
                i = j + 1
            else:
                i += 1
    roads["damaged"] = damaged

    nb = 0
    for B in roads["bridges"]:
        B.pop("d", None)
        if any(hit(x, y) for x, y in resample(B["p"], a.step)):
            B["d"] = 1
            nb += 1
    with open(ROADS, "w", encoding="utf-8") as f:
        json.dump(roads, f, separators=(",", ":"), ensure_ascii=False)
    total = sum(math.hypot(d["p"][i + 2] - d["p"][i], d["p"][i + 3] - d["p"][i + 1])
                for d in damaged for i in range(0, len(d["p"]) - 2, 2))
    print(f"damaged road sections: {len(damaged)} ({total * 2.93 / 1000:.2f} km); bridges flagged: {nb} of {len(roads['bridges'])}")
    for B in roads["bridges"]:
        if B.get("d"):
            print("  bridge:", B.get("n") or "(unnamed)", B["c"], B["rc"])


if __name__ == "__main__":
    main()
