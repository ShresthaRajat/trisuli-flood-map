#!/usr/bin/env python3
"""Crop the four full-extent layers to the Trisuli Bazar focus box (+ pad) for index.html.

The full-extent layers are 5425x4202 images stretched over the 36169x28011 world grid
(84.52-85.60E, 27.70-28.45N), so one image pixel is ~6.667 world px. The focus box is the
Trisuli Bazar / Bidur reach: 85.1302-85.1739E, 27.9016-27.9497N.
Outputs go to assets/focus/ and the script prints the placement (in world px) for place().

Usage: python3 scripts/crop_focus.py [--pad 300]
"""
import argparse, json, math, os, sys
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
OUT = os.path.join(ASSETS, "focus")

WORLD_W, WORLD_H = 36169, 28011
BW, BE, BN, BS = 84.52, 85.60, 28.45, 27.70
FOCUS_S, FOCUS_W, FOCUS_N, FOCUS_E = 27.9016, 85.1302, 27.9497, 85.1739

LAYERS = [
    ("pre_s2_20260603.jpg", "JPEG", {"quality": 92, "subsampling": 0}),
    ("post_s2_20260827.jpg", "JPEG", {"quality": 92, "subsampling": 0}),
    ("ps26_mosaic.webp", "WEBP", {"quality": 88, "method": 6}),
    ("ps28_mosaic.webp", "WEBP", {"quality": 88, "method": 6}),
]

def lon_to_x(lon): return (lon - BW) / (BE - BW) * WORLD_W
def lat_to_y(lat): return (BN - lat) / (BN - BS) * WORLD_H

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pad", type=int, default=60, help="pad around the box, in image pixels of the source layer")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    # focus box in world px
    fx0, fx1 = lon_to_x(FOCUS_W), lon_to_x(FOCUS_E)
    fy0, fy1 = lat_to_y(FOCUS_N), lat_to_y(FOCUS_S)
    print(f"focus box, world px: x {fx0:.1f}-{fx1:.1f}, y {fy0:.1f}-{fy1:.1f}")

    placement = {}
    for name, fmt, opts in LAYERS:
        src = os.path.join(ASSETS, name)
        im = Image.open(src)
        iw, ih = im.size
        sx, sy = WORLD_W / iw, WORLD_H / ih            # world px per image px
        # box in this image's pixels, padded and clamped to the image
        cx0 = max(0, math.floor(fx0 / sx) - args.pad); cy0 = max(0, math.floor(fy0 / sy) - args.pad)
        cx1 = min(iw, math.ceil(fx1 / sx) + args.pad);  cy1 = min(ih, math.ceil(fy1 / sy) + args.pad)
        crop = im.crop((cx0, cy0, cx1, cy1))
        dst = os.path.join(OUT, name)
        crop.save(dst, fmt, **opts)
        # where the crop sits on the world grid (what place() needs)
        left, top = cx0 * sx, cy0 * sy
        w, h = (cx1 - cx0) * sx, (cy1 - cy0) * sy
        placement[name] = {"left": round(left, 2), "top": round(top, 2), "width": round(w, 2), "height": round(h, 2),
                           "crop_px": [cx0, cy0, cx1, cy1]}
        print(f"{name}: crop image px x {cx0}-{cx1}, y {cy0}-{cy1} ({cx1-cx0}x{cy1-cy0}) -> "
              f"world offset ({left:.2f}, {top:.2f}), size {w:.2f}x{h:.2f}; {os.path.getsize(dst)/1e6:.1f} MB")

    with open(os.path.join(OUT, "placement.json"), "w") as f:
        json.dump(placement, f, indent=1)
    print("\nJS for index.html (world px):")
    for name, p in placement.items():
        print(f'  // {name}: crop {p["crop_px"]}')
    print("  var FOCUS_CROP = " + json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "crop_px"} for k, v in placement.items()}) + ";")

if __name__ == "__main__":
    main()
