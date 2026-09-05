#!/usr/bin/env python3
"""Fetch OpenStreetMap features for the Trisuli Bazar focus box from Overpass
and write assets/trisuli/osm.json in the compact world-pixel format used by
index.html / full-map.html.

Stdlib only (urllib, json, math). Run with: python3 scripts/fetch_osm.py
"""
import json
import re
import math
import os
import sys
import time
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Projection constants (must match index.html / full-map.html exactly)
# ---------------------------------------------------------------------------
WORLD_W, WORLD_H = 36169, 28011
BW, BE, BN, BS = 84.52, 85.60, 28.45, 27.70

def lon_to_x(lon):
    return (lon - BW) / (BE - BW) * WORLD_W

def lat_to_y(lat):
    return (BN - lat) / (BN - BS) * WORLD_H

def r1(v):
    return round(v, 1)

# ---------------------------------------------------------------------------
# Focus box
# ---------------------------------------------------------------------------
S, W, N, E = 27.9016, 85.1302, 27.9497, 85.1739
MARGIN = 0.0015
QS, QW, QN, QE = S - MARGIN, W - MARGIN, N + MARGIN, E + MARGIN

FOCUS_X0, FOCUS_Y0 = lon_to_x(W), lat_to_y(N)
FOCUS_X1, FOCUS_Y1 = lon_to_x(E), lat_to_y(S)
print(f"Focus box world px: x {FOCUS_X0:.1f}..{FOCUS_X1:.1f}, y {FOCUS_Y0:.1f}..{FOCUS_Y1:.1f}")

SCRATCH = "/private/tmp/claude-501/-Users-rajatshrestha-Documents-trisuli-flood-map/43665c9f-c3e6-44c1-a76e-86200a7ac26c/scratchpad"

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

BBOX = f"{QS},{QW},{QN},{QE}"

QUERY = f"""
[out:json][timeout:180];
(
  way["building"]({BBOX});
  relation["building"]({BBOX});
  nwr["amenity"]({BBOX});
  nwr["shop"]({BBOX});
  nwr["tourism"]({BBOX});
  nwr["leisure"]({BBOX});
  nwr["office"]({BBOX});
  nwr["healthcare"]({BBOX});
  nwr["historic"]({BBOX});
  nwr["man_made"]({BBOX});
  nwr["power"]({BBOX});
  nwr["emergency"]({BBOX});
  nwr["public_transport"]({BBOX});
  nwr["aeroway"]({BBOX});
  nwr["telecom"]({BBOX});
  nwr["pipeline"]({BBOX});
  nwr["place"]({BBOX});
  nwr["landuse"~"^(industrial|commercial|retail|cemetery|religious|education|institutional|military|construction)$"]({BBOX});
  nwr["waterway"~"^(dam|weir|canal|drain|ditch)$"]({BBOX});
  nwr["barrier"~"^(wall|retaining_wall|city_wall)$"]({BBOX});
);
out geom;
"""

def fetch_overpass():
    data = urllib.parse_data = urllib.parse.urlencode({"data": QUERY}).encode("utf-8") if False else ("data=" + urllib.parse.quote(QUERY)).encode("utf-8")
    last_err = None
    for endpoint in OVERPASS_ENDPOINTS:
        for attempt in range(2):
            try:
                req = urllib.request.Request(
                    endpoint, data=data,
                    headers={
                        "User-Agent": "trisuli-flood-map-osm-fetch/1.0 (contact: rajat.shrestha@learnamp.com)",
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                    method="POST",
                )
                print(f"POST {endpoint} (attempt {attempt+1})...")
                with urllib.request.urlopen(req, timeout=200) as resp:
                    body = resp.read()
                print(f"  -> {len(body)} bytes")
                return endpoint, body
            except urllib.error.HTTPError as e:
                last_err = e
                print(f"  HTTPError {e.code}: {e.reason}")
                if e.code in (429, 504):
                    print("  sleeping 20s then retrying...")
                    time.sleep(20)
                    continue
                else:
                    break
            except Exception as e:
                last_err = e
                print(f"  error: {e}")
                time.sleep(5)
                continue
    raise RuntimeError(f"All Overpass endpoints failed: {last_err}")

import urllib.parse

def load_or_fetch():
    cache_path = os.path.join(SCRATCH, "overpass_raw.json")
    if os.path.exists(cache_path) and os.environ.get("USE_CACHE") == "1":
        print(f"Using cached response at {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    endpoint, body = fetch_overpass()
    os.makedirs(SCRATCH, exist_ok=True)
    with open(cache_path, "wb") as f:
        f.write(body)
    print(f"Saved raw Overpass response to {cache_path}")
    data = json.loads(body)
    data["_endpoint_used"] = endpoint
    data["_response_bytes"] = len(body)
    return data

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


# Registration nudge for building/area footprints, in world px (1 world px ~ 2.93 m).
# OSM footprints here were traced from other imagery and sit a few metres SE of the roofs
# in the page's registered Vantor Legion 5 Feb 2026 frame. Measured 5 Sep 2026 by
# cross-correlating each footprint (7,089 buildings) against high-passed roof brightness
# in the Legion z8 tiles: the modal per-building offset is (-1.0, -1.0) world px
# (~3 m west, ~3 m north), with a broad spread that is per-building tracing error and
# cannot be fixed by a global shift. Roads/POI nodes are not shifted (they align as is).
# Override with BUILDING_SHIFT="dx,dy" (e.g. "0,0" to disable).
BUILDING_SHIFT = tuple(float(v) for v in os.environ.get("BUILDING_SHIFT", "-1.0,-1.0").split(","))

def shift_ring(ring):
    dx, dy = BUILDING_SHIFT
    if not dx and not dy:
        return ring
    out = []
    for i in range(0, len(ring), 2):
        out.append(r1(ring[i] + dx))
        out.append(r1(ring[i + 1] + dy))
    return out

def ring_from_geom(geom):
    """geom: list of {"lat":..,"lon":..} -> flat [x,y,x,y,...] world px, rounded."""
    pts = []
    for pt in geom:
        pts.append(r1(lon_to_x(pt["lon"])))
        pts.append(r1(lat_to_y(pt["lat"])))
    return pts

def ring_area2(ring):
    """Signed area*2 of a flat ring (for centroid / orientation)."""
    a = 0.0
    n = len(ring) // 2
    for i in range(n):
        x0, y0 = ring[2*i], ring[2*i+1]
        j = (i + 1) % n
        x1, y1 = ring[2*j], ring[2*j+1]
        a += x0 * y1 - x1 * y0
    return a

def ring_centroid(ring):
    n = len(ring) // 2
    if n < 3:
        # fallback: average
        xs = ring[0::2]; ys = ring[1::2]
        return sum(xs)/len(xs), sum(ys)/len(ys)
    a2 = ring_area2(ring)
    if abs(a2) < 1e-9:
        xs = ring[0::2]; ys = ring[1::2]
        return sum(xs)/len(xs), sum(ys)/len(ys)
    cx = cy = 0.0
    for i in range(n):
        x0, y0 = ring[2*i], ring[2*i+1]
        j = (i + 1) % n
        x1, y1 = ring[2*j], ring[2*j+1]
        cross = x0*y1 - x1*y0
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    factor = 1.0 / (3.0 * a2)
    return cx * factor, cy * factor

def point_in_ring(x, y, ring):
    """Even-odd rule, ring is flat [x,y,...]."""
    n = len(ring) // 2
    inside = False
    x0, y0 = ring[2*(n-1)], ring[2*(n-1)+1]
    for i in range(n):
        x1, y1 = ring[2*i], ring[2*i+1]
        if ((y1 > y) != (y0 > y)):
            xin = (x0 - x1) * (y - y1) / (y0 - y1 + 1e-12) + x1
            if x < xin:
                inside = not inside
        x0, y0 = x1, y1
    return inside

def point_in_polygon_with_holes(x, y, outer, holes):
    if not point_in_ring(x, y, outer):
        return False
    for h in holes:
        if point_in_ring(x, y, h):
            return False
    return True

def bbox_of_ring(ring):
    xs = ring[0::2]; ys = ring[1::2]
    return min(xs), min(ys), max(xs), max(ys)

def bbox_intersects(b1, b2):
    return not (b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3])

FOCUS_BBOX = (min(FOCUS_X0, FOCUS_X1), min(FOCUS_Y0, FOCUS_Y1), max(FOCUS_X0, FOCUS_X1), max(FOCUS_Y0, FOCUS_Y1))

def point_in_focus(x, y):
    return FOCUS_BBOX[0] <= x <= FOCUS_BBOX[2] and FOCUS_BBOX[1] <= y <= FOCUS_BBOX[3]

# ---------------------------------------------------------------------------
# Tag classification
# ---------------------------------------------------------------------------

SCHOOL_AMENITY = {"school", "college", "university", "kindergarten"}
HEALTH_AMENITY = {"hospital", "clinic", "doctors", "pharmacy"}
HEALTH_HEALTHCARE = {"hospital", "clinic", "centre", "pharmacy", "doctor", "health_post"}
WORSHIP_BUILDING = {"temple", "shrine", "church", "mosque", "monastery"}
CIVIC_AMENITY = {"townhall", "police", "fire_station", "public_building", "community_centre",
                 "library", "post_office", "bank", "courthouse", "government"}
COMMERCIAL_BUILDING = {"commercial", "retail", "industrial", "warehouse", "hotel", "office"}
RESIDENTIAL_BUILDING = {"house", "residential", "apartments", "hut", "detached", "terrace", "semidetached_house", "bungalow", "dormitory"}

def tags_get(tags, *keys):
    for k in keys:
        if k in tags and tags[k]:
            return tags[k]
    return None

def building_class_from_tags(tags):
    b = tags.get("building")
    amenity = tags.get("amenity")
    healthcare = tags.get("healthcare")
    if amenity in SCHOOL_AMENITY or b in ("school", "college", "university", "kindergarten"):
        return "school"
    if amenity in HEALTH_AMENITY or healthcare or b in ("hospital", "clinic"):
        return "health"
    if tags.get("amenity") == "place_of_worship" or b in WORSHIP_BUILDING or tags.get("building") == "religious":
        return "worship"
    if amenity in CIVIC_AMENITY:
        return "civic"
    if b in COMMERCIAL_BUILDING or tags.get("shop") or tags.get("tourism") in ("hotel", "guest_house"):
        return "commercial"
    if b in RESIDENTIAL_BUILDING:
        return "residential"
    return "other"

def area_class_from_tags(tags):
    amenity = tags.get("amenity")
    leisure = tags.get("leisure")
    landuse = tags.get("landuse")
    power = tags.get("power")
    man_made = tags.get("man_made")
    historic = tags.get("historic")
    waterway = tags.get("waterway")

    if amenity in SCHOOL_AMENITY:
        return "school"
    if amenity in HEALTH_AMENITY:
        return "health"
    if amenity == "place_of_worship":
        return "worship"
    if amenity in ("police", "townhall") or landuse in ("military", "institutional"):
        return "civic"
    if leisure in ("pitch", "sports_centre", "stadium", "track", "playground"):
        return "sport"
    if landuse in ("religious",) and amenity != "place_of_worship":
        return "worship"
    if landuse in ("recreation_ground",) or leisure in ("park", "garden"):
        return "park"
    if amenity == "grave_yard" or landuse == "cemetery":
        return "cemetery"
    if (landuse in ("industrial", "commercial", "retail", "construction")
            or power in ("plant", "substation")
            or man_made in ("works", "water_works", "wastewater_plant")):
        return "industrial"
    if amenity in ("bus_station", "parking") or tags.get("aeroway") in ("aerodrome", "airstrip"):
        return "transport"
    if historic:
        return "civic"
    return None

def line_class_from_tags(tags):
    power = tags.get("power")
    waterway = tags.get("waterway")
    man_made = tags.get("man_made")
    barrier = tags.get("barrier")
    pipeline = tags.get("pipeline") or tags.get("man_made") == "pipeline"
    if power in ("line", "minor_line", "cable"):
        return "power"
    if pipeline or "pipeline" in tags:
        return "pipeline"
    if waterway in ("canal", "drain", "ditch"):
        return "canal"
    if waterway in ("dam", "weir") or man_made in ("dyke", "embankment"):
        return "dam"
    if barrier in ("wall", "retaining_wall", "city_wall"):
        return "wall"
    if tags.get("aerialway"):
        return "aerialway"
    return None

def poi_class_from_tags(tags):
    amenity = tags.get("amenity")
    shop = tags.get("shop")
    tourism = tags.get("tourism")
    man_made = tags.get("man_made")
    power = tags.get("power")
    historic = tags.get("historic")
    place = tags.get("place")
    healthcare = tags.get("healthcare")

    if amenity in SCHOOL_AMENITY:
        return "school"
    if amenity in HEALTH_AMENITY or healthcare:
        return "health"
    if amenity == "place_of_worship":
        return "worship"
    if amenity == "police":
        return "police"
    if amenity in ("townhall", "courthouse", "post_office", "community_centre", "library") or tags.get("office") == "government":
        return "gov"
    if amenity == "bank" or amenity == "atm":
        return "bank"
    if amenity == "marketplace" or shop in ("supermarket", "mall", "department_store"):
        return "market"
    if amenity in ("bus_station", "bus_stop", "taxi", "fuel", "ferry_terminal") or tags.get("public_transport"):
        return "transport"
    if power in ("plant", "substation", "generator", "tower"):
        return "power"
    if amenity in ("water_works",) or man_made in ("water_tower", "water_well", "reservoir_covered", "water_works") or tags.get("natural") == "spring" or amenity == "drinking_water":
        return "water"
    if man_made in ("mast", "tower", "communications_tower") or tags.get("telecom"):
        return "telecom"
    if amenity in ("hotel",) or tourism in ("hotel", "guest_house", "hostel", "lodge") or shop == "hotel":
        return "hotel"
    if place in ("town", "village", "suburb", "neighbourhood", "hamlet", "locality"):
        return "place"
    if amenity in ("fire_station", "ambulance_station", "shelter", "assembly_point") or tags.get("emergency"):
        return "emergency"
    if historic or tourism in ("attraction", "viewpoint", "museum"):
        return "historic"
    return None

NEEDS_NAME_POI = {"market", "power", "hotel", "historic", "other", "gov", "bank"}

def poi_kind_label(tags, cls):
    amenity = tags.get("amenity")
    shop = tags.get("shop")
    healthcare = tags.get("healthcare")
    place = tags.get("place")
    religion = tags.get("religion")
    building = tags.get("building")
    if cls == "school":
        m = {"school": "School", "college": "College", "university": "University", "kindergarten": "Kindergarten"}
        return m.get(amenity, m.get(building, "School"))
    if cls == "health":
        if amenity == "hospital":
            return "Hospital"
        if amenity == "clinic" or healthcare in ("clinic", "centre"):
            return "Health post"
        if amenity == "pharmacy":
            return "Pharmacy"
        if amenity == "doctors":
            return "Doctors"
        return "Health facility"
    if cls == "worship":
        if religion == "hindu":
            return "Hindu temple"
        if religion == "buddhist":
            return "Buddhist monastery" if tags.get("building") == "monastery" or amenity == "monastery" else "Buddhist temple"
        if religion == "christian":
            return "Church"
        if religion == "muslim":
            return "Mosque"
        return "Place of worship"
    if cls == "police":
        return "Police"
    if cls == "gov":
        m = {"townhall": "Town hall", "courthouse": "Courthouse", "post_office": "Post office", "community_centre": "Community centre", "library": "Library"}
        return m.get(amenity, "Government office")
    if cls == "bank":
        return "ATM" if amenity == "atm" else "Bank"
    if cls == "market":
        return "Market"
    if cls == "transport":
        m = {"bus_station": "Bus station", "bus_stop": "Bus stop", "taxi": "Taxi stand", "fuel": "Fuel station", "ferry_terminal": "Ferry"}
        return m.get(amenity, "Transport")
    if cls == "power":
        return "Substation" if tags.get("power") == "substation" else ("Power plant" if tags.get("power") == "plant" else "Power")
    if cls == "water":
        return "Water tower" if tags.get("man_made") == "water_tower" else "Water source"
    if cls == "telecom":
        return "Telecom tower"
    if cls == "hotel":
        return "Hotel"
    if cls == "place":
        return (place or "place").capitalize()
    if cls == "emergency":
        return "Fire station" if amenity == "fire_station" else "Emergency"
    if cls == "historic":
        return "Historic site"
    return "Point of interest"

def get_name(tags):
    n = tags_get(tags, "name:en")
    if n:
        return n, None
    n = tags.get("name")
    if n:
        ne = tags.get("name:ne")
        if ne and ne != n:
            return n, ne
        return n, None
    return None, None

# ---------------------------------------------------------------------------
# Element processing
# ---------------------------------------------------------------------------

def dedupe(elements):
    seen = set()
    out = []
    for el in elements:
        key = (el.get("type"), el.get("id"))
        if key in seen:
            continue
        seen.add(key)
        out.append(el)
    return out

def way_ring(el):
    geom = el.get("geometry")
    if not geom or len(geom) < 4:
        return None
    ring = ring_from_geom(geom)
    if len(ring) < 8:
        return None
    # ensure closed
    if ring[0] != ring[-2] or ring[1] != ring[-1]:
        ring.append(ring[0]); ring.append(ring[1])
    return ring

def way_polyline(el):
    geom = el.get("geometry")
    if not geom or len(geom) < 2:
        return None
    return ring_from_geom(geom)

def is_closed_way(el):
    geom = el.get("geometry")
    if not geom or len(geom) < 4:
        return False
    return geom[0].get("lat") == geom[-1].get("lat") and geom[0].get("lon") == geom[-1].get("lon")

def relation_outers_inners(el):
    """Build outer/inner rings for a multipolygon-style relation by joining
    member way geometries end-to-end."""
    members = el.get("members", [])
    outer_segs = []
    inner_segs = []
    for m in members:
        if m.get("type") != "way":
            continue
        geom = m.get("geometry")
        if not geom:
            continue
        role = m.get("role", "outer")
        seg = [(p["lat"], p["lon"]) for p in geom]
        if role == "inner":
            inner_segs.append(seg)
        else:
            outer_segs.append(seg)

    def join_segments(segs):
        if not segs:
            return []
        segs = [list(s) for s in segs]
        rings = []
        used = [False] * len(segs)
        for i in range(len(segs)):
            if used[i]:
                continue
            used[i] = True
            cur = segs[i][:]
            changed = True
            while changed:
                changed = False
                for j in range(len(segs)):
                    if used[j]:
                        continue
                    s = segs[j]
                    if cur[-1] == s[0]:
                        cur.extend(s[1:]); used[j] = True; changed = True
                    elif cur[-1] == s[-1]:
                        cur.extend(list(reversed(s))[1:]); used[j] = True; changed = True
                    elif cur[0] == s[-1]:
                        cur = s[:-1] + cur; used[j] = True; changed = True
                    elif cur[0] == s[0]:
                        cur = list(reversed(s))[:-1] + cur; used[j] = True; changed = True
            rings.append(cur)
        return rings

    outer_rings_ll = join_segments(outer_segs)
    inner_rings_ll = join_segments(inner_segs)

    def ll_ring_to_px(ring_ll):
        flat = []
        for lat, lon in ring_ll:
            flat.append(r1(lon_to_x(lon)))
            flat.append(r1(lat_to_y(lat)))
        if len(flat) >= 4 and (flat[0] != flat[-2] or flat[1] != flat[-1]):
            flat.append(flat[0]); flat.append(flat[1])
        return flat

    outers = [ll_ring_to_px(r) for r in outer_rings_ll if len(r) >= 3]
    inners = [ll_ring_to_px(r) for r in inner_rings_ll if len(r) >= 3]
    outers = [r for r in outers if len(r) >= 8]
    inners = [r for r in inners if len(r) >= 8]
    return outers, inners

# ---------------------------------------------------------------------------
# Flood exposure (assets/trisuli/flood_extent.json) -- see note in main()
# ---------------------------------------------------------------------------

def load_flood_extent():
    path = os.path.join(os.path.dirname(__file__), "..", "assets", "trisuli", "flood_extent.json")
    path = os.path.normpath(path)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if not all(k in d for k in ("oldriver", "path", "affected")):
        return None
    return d

def flood_class(cx, cy, flood):
    if flood is None:
        return 0
    for ring in flood.get("path", []):
        if len(ring) >= 8 and point_in_ring(cx, cy, ring):
            return 1
    for ring in flood.get("affected", []):
        if len(ring) >= 8 and point_in_ring(cx, cy, ring):
            return 2
    return 0

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    raw = load_or_fetch()
    endpoint_used = raw.get("_endpoint_used", "unknown")
    response_bytes = raw.get("_response_bytes", -1)
    elements = dedupe(raw.get("elements", []))
    print(f"Total elements after dedupe: {len(elements)}")

    flood = load_flood_extent()
    if flood is None:
        print("WARNING: assets/trisuli/flood_extent.json is missing or does not match the "
              "{oldriver,path,affected} schema -- flood exposure (buildings.f) will be omitted "
              "(all buildings f=0 / key dropped where possible).")

    buildings = []  # each: dict p,c,n,ne,f,h
    areas = []      # dict p,c,n,ne
    lines = []      # dict p,c
    pois = []       # dict x,y,c,n,ne,k

    # area polygons for containment tests (class -> list of (outer, holes))
    area_polys_by_class = {"school": [], "health": [], "worship": [], "civic": []}
    # poi points inside buildings, by class
    poi_points_by_class = {"school": [], "health": [], "worship": [], "civic": []}

    counts = {"buildings": 0, "areas": 0, "lines": 0, "pois": 0}
    building_class_hist = {}
    area_class_hist = {}
    line_class_hist = {}
    poi_class_hist = {}

    raw_buildings = []   # (outer_rings[with holes], tags, is_way)
    raw_amenity_areas = []  # (outer, holes, tags)
    raw_lines = []
    raw_poi_nodes = []
    raw_place_named_ways = []  # named amenity ways that should also get a poi

    unclassified_tags_seen = []

    for el in elements:
        etype = el.get("type")
        tags = el.get("tags", {}) or {}
        if not tags:
            continue

        if etype == "node":
            lat, lon = el.get("lat"), el.get("lon")
            if lat is None or lon is None:
                continue
            x, y = r1(lon_to_x(lon)), r1(lat_to_y(lat))
            raw_poi_nodes.append((x, y, tags))
            continue

        if etype == "way":
            closed = is_closed_way(el)
            is_building = "building" in tags
            amenity_area_tag = area_class_from_tags(tags)
            line_tag = line_class_from_tags(tags)

            if is_building and closed:
                ring = way_ring(el)
                if ring:
                    raw_buildings.append(([ring], [], tags))
                continue

            if closed and amenity_area_tag:
                ring = way_ring(el)
                if ring:
                    raw_amenity_areas.append((ring, [], tags))
                continue

            if line_tag:
                pl = way_polyline(el)
                if pl:
                    raw_lines.append((pl, tags, line_tag))
                continue

            # Closed way that is a standalone amenity/shop/tourism/etc. footprint
            # without a "building" tag and not one of the fixed area classes
            # (e.g. a mapped tourism=hotel or amenity=marketplace outline) --
            # still worth a centroid POI so it isn't silently dropped.
            if closed:
                poi_tag = poi_class_from_tags(tags)
                if poi_tag:
                    ring = way_ring(el)
                    if ring:
                        cx, cy = ring_centroid(ring)
                        raw_poi_nodes.append((r1(cx), r1(cy), tags))
                continue

            continue

        if etype == "relation":
            rel_type = tags.get("type")
            is_building = "building" in tags
            amenity_area_tag = area_class_from_tags(tags)
            if not (is_building or amenity_area_tag):
                poi_tag = poi_class_from_tags(tags)
                if poi_tag and rel_type == "multipolygon":
                    outers, _inners = relation_outers_inners(el)
                    if outers:
                        cx, cy = ring_centroid(outers[0])
                        raw_poi_nodes.append((r1(cx), r1(cy), tags))
                continue
            outers, inners = relation_outers_inners(el)
            if not outers:
                continue
            if is_building:
                for outer in outers:
                    raw_buildings.append(([outer], inners, tags))
            elif amenity_area_tag:
                for outer in outers:
                    raw_amenity_areas.append((outer, inners, tags))
            continue

    print(f"Raw buildings(ways/rels outers): {len(raw_buildings)}, amenity areas: {len(raw_amenity_areas)}, "
          f"lines: {len(raw_lines)}, poi nodes: {len(raw_poi_nodes)}")

    # Build area class polygons for containment lookups + emit areas list
    for outer, holes, tags in raw_amenity_areas:
        cls = area_class_from_tags(tags)
        if cls is None:
            continue
        bb = bbox_of_ring(outer)
        if not bbox_intersects(bb, FOCUS_BBOX):
            continue
        name, name_ne = get_name(tags) if get_name(tags)[0] else (None, None)
        area_obj = {"p": shift_ring(outer), "c": cls}
        if name:
            area_obj["n"] = name
            if name_ne:
                area_obj["ne"] = name_ne
        areas.append(area_obj)
        area_class_hist[cls] = area_class_hist.get(cls, 0) + 1
        if cls in area_polys_by_class:
            area_polys_by_class[cls].append((outer, holes))
        # emit centroid POI for named school/health/worship/civic areas.
        # "civic" is a building/area class but NOT a valid poi class (pois.c has
        # police/gov/historic/other instead) -- remap it.
        if cls in ("school", "health", "worship", "civic") and name:
            poi_cls = cls
            if cls == "civic":
                if tags.get("amenity") == "police":
                    poi_cls = "police"
                elif tags.get("amenity") in ("townhall", "courthouse", "post_office",
                                              "community_centre", "library") or tags.get("office") == "government":
                    poi_cls = "gov"
                elif tags.get("historic"):
                    poi_cls = "historic"
                else:
                    poi_cls = "other"
            cx, cy = ring_centroid(outer)
            cx, cy = r1(cx + BUILDING_SHIFT[0]), r1(cy + BUILDING_SHIFT[1])
            if point_in_focus(cx, cy):
                pois.append({"x": r1(cx), "y": r1(cy), "c": poi_cls, "n": name,
                             "k": poi_kind_label(tags, poi_cls), **({"ne": name_ne} if name_ne else {}), "_tags": tags})
                poi_class_hist[poi_cls] = poi_class_hist.get(poi_cls, 0) + 1

    # Classify POI nodes (amenity/shop/etc on nodes)
    for x, y, tags in raw_poi_nodes:
        cls = poi_class_from_tags(tags)
        if cls is None:
            continue
        name, name_ne = get_name(tags)
        if not name and cls in NEEDS_NAME_POI:
            continue
        if not point_in_focus(x, y):
            continue
        obj = {"x": x, "y": y, "c": cls, "k": poi_kind_label(tags, cls), "_tags": tags}
        if name:
            obj["n"] = name
            if name_ne:
                obj["ne"] = name_ne
        pois.append(obj)
        poi_class_hist[cls] = poi_class_hist.get(cls, 0) + 1
        if cls in poi_points_by_class:
            poi_points_by_class[cls].append((x, y, name, name_ne))

    # Buildings: classify, apply area/poi containment override, clip by centroid, flood exposure
    f1_count = f2_count = f0_count = 0
    f1_important = f2_important = 0
    for outer_list, holes, tags in raw_buildings:
        outer = outer_list[0]
        bb = bbox_of_ring(outer)
        cx, cy = ring_centroid(outer)
        if not point_in_focus(cx, cy):
            continue
        cls = building_class_from_tags(tags)
        if cls == "other":
            # check containment in area polygons of priority classes
            for pref_cls in ("school", "health", "worship", "civic"):
                found = False
                for a_outer, a_holes in area_polys_by_class.get(pref_cls, []):
                    if point_in_polygon_with_holes(cx, cy, a_outer, a_holes):
                        cls = pref_cls
                        found = True
                        break
                if found:
                    break
        name, name_ne = get_name(tags)
        # if still no name and a poi point of matching class lies inside, inherit name
        if cls in poi_points_by_class:
            for px, py, pname, pname_ne in poi_points_by_class[cls]:
                if point_in_ring(px, py, outer):
                    if not name and pname:
                        name, name_ne = pname, pname_ne
                    break

        fclass = flood_class(cx, cy, flood)
        if fclass == 1:
            f1_count += 1
            if cls in ("school", "health", "worship", "civic"):
                f1_important += 1
        elif fclass == 2:
            f2_count += 1
            if cls in ("school", "health", "worship", "civic"):
                f2_important += 1
        else:
            f0_count += 1

        # Named government/administrative buildings without an amenity tag (e.g. the
        # "Bidur Municipality office") still deserve a key-location point.
        if name and cls in ("other", "civic") and re.search(r"\b(office|municipality|nagarpalika|ward|district|court|department)\b", name, re.I) \
                and not any(q.get("n") == name for q in pois):
            pcx, pcy = r1(cx + BUILDING_SHIFT[0]), r1(cy + BUILDING_SHIFT[1])
            pois.append({"x": pcx, "y": pcy, "c": "gov", "n": name, "k": "Government office", "_tags": tags})
            poi_class_hist["gov"] = poi_class_hist.get("gov", 0) + 1
        b = {"p": shift_ring(outer), "c": cls}
        if name:
            b["n"] = name
            if name_ne:
                b["ne"] = name_ne
        if flood is not None:
            b["f"] = fclass
        if holes:
            b["h"] = [shift_ring(h) for h in holes]
        buildings.append(b)
        building_class_hist[cls] = building_class_hist.get(cls, 0) + 1

    # Lines
    for pl, tags, cls in raw_lines:
        bb = bbox_of_ring(pl) if len(pl) >= 4 else (min(pl[0::2]), min(pl[1::2]), max(pl[0::2]), max(pl[1::2]))
        if not bbox_intersects(bb, FOCUS_BBOX):
            continue
        lines.append({"p": pl, "c": cls})
        line_class_hist[cls] = line_class_hist.get(cls, 0) + 1

    counts["buildings"] = len(buildings)
    counts["areas"] = len(areas)
    counts["lines"] = len(lines)
    # Keep only essential points (user request 5 Sep 2026): schools, health, temples,
    # police/government, emergency services, power stations, bus stations, place names
    # and historic sites. Banks, hotels/resorts, shops, taps, masts etc. are dropped.
    ESSENTIAL_POI_CLASSES = {"school", "health", "worship", "police", "gov", "emergency",
                             "power", "transport", "place", "historic"}
    def essential(p):
        t = p.get("_tags", {})
        nm = (p.get("n") or "").lower()
        if p["c"] not in ESSENTIAL_POI_CLASSES:
            return False
        if p["c"] == "transport":   # bus parks only: bus stations, not stops/fuel/taxi
            return (t.get("amenity") == "bus_station" or t.get("public_transport") == "station") and "bus stop" not in nm
        if p["c"] == "historic":    # real historic sites only, not viewpoints/picnic spots/attractions
            return bool(t.get("historic"))
        if p["c"] == "place":       # unnamed hamlets carry no information
            return bool(p.get("n"))
        if p["c"] == "emergency" and not (t.get("emergency") or t.get("amenity") in ("fire_station", "ambulance_station", "rescue_station")):
            p["c"] = "gov"          # e.g. a municipality office mis-bucketed as emergency
            p["k"] = "Government office"
        return True
    pois = [p for p in pois if essential(p)]
    for p in pois:
        p.pop("_tags", None)
    # Collapse stacked duplicates (same class + name within ~15 world px, e.g. the
    # hydropower station mapped as several nodes) into one point at their mean position.
    merged = []
    for p in pois:
        hit = None
        for q in merged:
            if q["c"] == p["c"] and q.get("n") == p.get("n") and abs(q["x"] - p["x"]) < 15 and abs(q["y"] - p["y"]) < 15:
                hit = q
                break
        if hit is None:
            p["_n"] = 1
            merged.append(p)
        else:
            k = hit["_n"]
            hit["x"] = r1((hit["x"] * k + p["x"]) / (k + 1))
            hit["y"] = r1((hit["y"] * k + p["y"]) / (k + 1))
            hit["_n"] = k + 1
    for p in merged:
        p.pop("_n", None)
    pois = merged
    counts["pois"] = len(pois)

    meta = {
        "source": "© OpenStreetMap contributors (ODbL)",
        "fetched": time.strftime("%Y-%m-%d", time.gmtime()),
        "bbox": [S, W, N, E],
        "counts": counts,
    }

    out = {
        "meta": meta,
        "buildings": buildings,
        "areas": areas,
        "lines": lines,
        "pois": pois,
    }

    out_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "assets", "trisuli", "osm.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"), ensure_ascii=False)

    # --- Verification ---
    with open(out_path, "r", encoding="utf-8") as f:
        reloaded = json.load(f)
    size_kb = os.path.getsize(out_path) / 1024.0

    print("\n=== VERIFY ===")
    print("json.load OK:", bool(reloaded))
    print("counts:", counts)
    print("file size KB: %.1f" % size_kb)
    print("building class histogram:", building_class_hist)
    print("area class histogram:", area_class_hist)
    print("line class histogram:", line_class_hist)
    print("poi class histogram:", poi_class_hist)
    print(f"flood exposure: f1(path)={f1_count} f2(affected)={f2_count} f0={f0_count} "
          f"| f1 important(school/health/worship/civic)={f1_important} f2 important={f2_important}")

    named_pois = [p for p in pois if "n" in p][:5]
    print("\n5 example named POIs:")
    for p in named_pois:
        print(" ", p.get("n"), p.get("c"), p.get("k"), "x=%.1f y=%.1f" % (p["x"], p["y"]))

    print(f"\nendpoint used: {endpoint_used}, raw response bytes: {response_bytes}")
    print(f"focus box world px check: x {FOCUS_BBOX[0]:.1f}..{FOCUS_BBOX[2]:.1f}, "
          f"y {FOCUS_BBOX[1]:.1f}..{FOCUS_BBOX[3]:.1f}")

    # collect unfitted tag combos for report (best-effort, not exhaustive)
    return {
        "endpoint": endpoint_used,
        "response_bytes": response_bytes,
        "counts": counts,
        "building_class_hist": building_class_hist,
        "area_class_hist": area_class_hist,
        "line_class_hist": line_class_hist,
        "poi_class_hist": poi_class_hist,
        "flood": {"f1": f1_count, "f2": f2_count, "f0": f0_count,
                  "f1_important": f1_important, "f2_important": f2_important},
        "size_kb": size_kb,
        "named_pois_sample": named_pois,
        "flood_file_present": flood is not None,
    }

if __name__ == "__main__":
    result = main()
    print("\n=== SUMMARY JSON ===")
    print(json.dumps(result, ensure_ascii=False, default=str))
