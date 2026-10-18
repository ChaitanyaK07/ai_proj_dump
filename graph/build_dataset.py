"""
Build the raw routing dataset for one study area from OpenStreetMap.

    pip install osmnx rasterio   # osmnx 2.x; rasterio needed for --canopy-raster / --dem

    python build_dataset.py --lat 17.4981608 --lon 78.3567628 --dist 2000 --name miyapur_2km \
        --canopy-raster data/worldcover_N15E078.tif

Canopy source:
    --canopy-raster given  -> share of ESA WorldCover "tree cover" pixels in each edge's 15 m buffer
    not given              -> share of edge length within 15 m of OSM vegetation (sparse in Hyderabad)

    WorldCover tile covering Hyderabad:
    https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N15E078_Map.tif

Writes to data/:
    <name>.graphml        projected walking graph, with a `canopy` value on every edge
    <name>_coverage.csv   the Week 1 coverage table (share of edges carrying each tag)

Run it on two or three candidate areas and compare the coverage tables before
committing to one.
"""
import argparse
from pathlib import Path

import networkx as nx
import osmnx as ox
import pandas as pd

UTM_44N = "EPSG:32644"      # covers Hyderabad; units are metres
CANOPY_BUFFER_M = 15        # Section 3 of the proposal
WORLDCOVER_TREE = 10        # ESA WorldCover class code for "tree cover"

# OSMnx only keeps a short default list of way tags and silently drops the rest.
# Anything the cost model or the audit needs must be added BEFORE downloading.
EXTRA_WAY_TAGS = ["surface", "lit", "sidewalk", "foot", "incline", "step_count", "footway"]

# Mapped vegetation. natural=wood is included because mappers use it
# interchangeably with landuse=forest.
VEGETATION_TAGS = {
    "natural": ["tree", "tree_row", "wood"],
    "leisure": ["park", "garden"],
    "landuse": ["forest"],
}

AUDIT_TAGS = ["highway", "name", "surface", "lit", "sidewalk", "foot", "incline", "maxspeed", "lanes"]


OVERPASS_SERVERS = [
    "https://overpass.kumi.systems/api",
    "https://overpass.private.coffee/api",
    "https://overpass-api.de/api",
]


def download(lat, lon, dist):
    """Fetch the walking network and vegetation features. Both come back in lat/lon.
    Tries each Overpass server in turn until one responds."""
    import requests

    ox.settings.use_cache = True          # responses saved in ./cache, so reruns don't re-download
    ox.settings.log_console = True
    ox.settings.overpass_rate_limit = False   # mirrors don't serve /status reliably
    ox.settings.useful_tags_way = sorted(set(ox.settings.useful_tags_way) | set(EXTRA_WAY_TAGS))

    last_err = None
    for url in OVERPASS_SERVERS:
        ox.settings.overpass_url = url
        print(f"Trying Overpass server: {url}")
        try:
            # Simplify ourselves so that a staircase is never merged into the road it joins:
            # edges are only merged when their highway class matches.
            G = ox.graph_from_point((lat, lon), dist=dist, network_type="walk", simplify=False)
            G = ox.simplify_graph(G, edge_attrs_differ=["highway"])
            try:
                veg = ox.features_from_point((lat, lon), tags=VEGETATION_TAGS, dist=dist + CANOPY_BUFFER_M)
            except ox._errors.InsufficientResponseError:
                veg = None                # nothing mapped at all in this area
            return G, veg
        except requests.exceptions.RequestException as e:
            print(f"  failed: {type(e).__name__}, trying next server")
            last_err = e

    raise RuntimeError("All Overpass servers failed. Check your network or try a phone hotspot.") from last_err


def add_elevation(G, dem_path):
    """Optional. Needs a local SRTM GeoTIFF; must run while G is still in lat/lon."""
    G = ox.elevation.add_node_elevations_raster(G, dem_path)
    return ox.elevation.add_edge_grades(G, add_absolute=False)


def canopy_from_raster(edges, raster_path):
    """Share of tree-cover pixels inside each edge's 15 m buffer (0..1)."""
    import rasterio
    from rasterio.mask import mask

    with rasterio.open(raster_path) as src:
        buffers = edges.geometry.buffer(CANOPY_BUFFER_M).to_crs(src.crs)
        vals = []
        for geom in buffers:
            try:
                data, _ = mask(src, [geom], crop=True, filled=False, all_touched=True)
                px = data[0].compressed()
                vals.append(float((px == WORLDCOVER_TREE).mean()) if px.size else 0.0)
            except ValueError:            # buffer falls outside the raster
                vals.append(0.0)
    return pd.Series(vals, index=edges.index)


def canopy_from_osm(edges, veg):
    """Share of edge length within 15 m of mapped OSM vegetation (0..1)."""
    if veg is None or not len(veg):
        return pd.Series(0.0, index=edges.index)
    shade_zone = veg.to_crs(UTM_44N).geometry.buffer(CANOPY_BUFFER_M).union_all()
    return (edges.geometry.intersection(shade_zone).length / edges.geometry.length).clip(0, 1)


def process(G, veg, canopy_raster=None):
    """Project to metres, compute canopy per edge, build the coverage table."""
    G = ox.project_graph(G, to_crs=UTM_44N)
    edges = ox.graph_to_gdfs(G, nodes=False)          # index is (u, v, key)

    if canopy_raster:
        edges["canopy"] = canopy_from_raster(edges, canopy_raster)
        source = "WorldCover tree cover"
    else:
        edges["canopy"] = canopy_from_osm(edges, veg)
        source = "mapped OSM vegetation"
    nx.set_edge_attributes(G, edges["canopy"].to_dict(), "canopy")

    return G, coverage_table(G, edges, veg, source)


def coverage_table(G, edges, veg, canopy_source):
    total_m = edges["length"].sum()
    rows = [
        ("nodes", G.number_of_nodes(), "", ""),
        ("directed edges", len(edges), "", ""),
        ("network length (km, both directions)", round(total_m / 1000, 1), "", ""),
    ]

    def share(mask):
        return round(100 * mask.mean(), 1), round(100 * edges.loc[mask, "length"].sum() / total_m, 1)

    for tag in AUDIT_TAGS:
        mask = edges[tag].notna() if tag in edges else pd.Series(False, index=edges.index)
        rows.append((f"tag: {tag}", int(mask.sum()), *share(mask)))

    # After simplification a merged edge can carry a list of highway values.
    is_steps = edges["highway"].apply(lambda h: "steps" in h if isinstance(h, list) else h == "steps")
    rows.append(("highway=steps", int(is_steps.sum()), *share(is_steps)))

    # Canopy: any shade, and substantial shade (>= 25% of buffer / length)
    has_canopy = edges["canopy"] > 0
    rows.append((f"canopy > 0 ({canopy_source}, {CANOPY_BUFFER_M} m)", int(has_canopy.sum()), *share(has_canopy)))
    good_canopy = edges["canopy"] >= 0.25
    rows.append((f"canopy >= 0.25 ({canopy_source})", int(good_canopy.sum()), *share(good_canopy)))
    rows.append(("mean canopy (length-weighted)",
                 round(float((edges["canopy"] * edges["length"]).sum() / total_m), 3), "", ""))

    if "grade" in edges:
        mask = edges["grade"].notna()
        rows.append(("has elevation grade", int(mask.sum()), *share(mask)))

    if veg is not None and len(veg):
        for key, values in VEGETATION_TAGS.items():
            if key in veg:
                for value in values:
                    rows.append((f"vegetation objects: {key}={value}", int((veg[key] == value).sum()), "", ""))
    else:
        rows.append(("vegetation objects (all)", 0, "", ""))

    return pd.DataFrame(rows, columns=["item", "count", "% of edges", "% of length"], dtype=object)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--dist", type=float, default=2000, help="metres from the centre point")
    p.add_argument("--name", required=True, help="output file prefix, e.g. miyapur_2km")
    p.add_argument("--dem", help="optional SRTM GeoTIFF covering the area")
    p.add_argument("--canopy-raster", help="optional ESA WorldCover GeoTIFF covering the area")
    args = p.parse_args()

    G, veg = download(args.lat, args.lon, args.dist)
    if args.dem:
        G = add_elevation(G, args.dem)
    G, coverage = process(G, veg, args.canopy_raster)

    out = Path("data")
    out.mkdir(exist_ok=True)
    ox.save_graphml(G, out / f"{args.name}.graphml")
    coverage.to_csv(out / f"{args.name}_coverage.csv", index=False)
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()