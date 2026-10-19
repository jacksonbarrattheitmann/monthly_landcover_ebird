import ee
import geemap
import geopandas as gpd
import numpy as np
from pathlib import Path

ee.Initialize(project="ephem-wetlands-macro-2023")

PROJECT_DIR = Path.home() / "Documents" / "ebird_landcover"
HOTSPOT_FILE = PROJECT_DIR / "ebird_hotspots" / "wost_localties.shp"
LOG_FILE = PROJECT_DIR / "output" / "gee_batches_submitted.txt"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

START, N_MONTHS = "2010-01-01", 202
BUFFER_M = 5000
GRID_DEG = 2          # size of the spatial grid cells used for grouping
BATCH_SIZE = 500      # max hotspots per export task
MAX_SUBMIT = 2        # tasks submitted per run; start small to measure cost

# ---------- Build spatially compact batches ----------
gdf = gpd.read_file(HOTSPOT_FILE).to_crs("EPSG:4326")[["lclty_d", "stat_cd", "geometry"]]
gdf["row"] = np.floor((gdf.geometry.y + 90) / GRID_DEG).astype(int)
gdf["col"] = np.floor((gdf.geometry.x + 180) / GRID_DEG).astype(int)
gdf = gdf.sort_values(["row", "col"])

batches = []
for (r, c), group in gdf.groupby(["row", "col"]):
    for i in range(0, len(group), BATCH_SIZE):
        name = f"mndwi_r{r}_c{c}_b{i // BATCH_SIZE:02d}"
        batches.append((name, group.iloc[i:i + BATCH_SIZE]))

print(f"{len(gdf)} hotspots -> {len(batches)} batches")

# ---------- Earth Engine processing (same logic as before) ----------
def prep(img, green, swir):
    clear = img.select("QA_PIXEL").bitwiseAnd(0b111111).eq(0)
    sr = img.select([green, swir]).multiply(0.0000275).add(-0.2)
    return (sr.normalizedDifference([green, swir]).rename("mndwi")
              .updateMask(clear)
              .copyProperties(img, ["system:time_start"]))

def export_batch(batch_gdf, name):
    buffers = geemap.gdf_to_ee(
        batch_gdf[["lclty_d", "stat_cd", "geometry"]]
    ).map(lambda f: f.buffer(BUFFER_M))

    def load(cid, green, swir):
        return (ee.ImageCollection(cid)
                  .filterBounds(buffers)
                  .map(lambda i: prep(i, green, swir)))

    landsat = (load("LANDSAT/LT05/C02/T1_L2", "SR_B2", "SR_B5")
        .merge(load("LANDSAT/LE07/C02/T1_L2", "SR_B2", "SR_B5"))
        .merge(load("LANDSAT/LC08/C02/T1_L2", "SR_B3", "SR_B6"))
        .merge(load("LANDSAT/LC09/C02/T1_L2", "SR_B3", "SR_B6")))

    empty = ee.Image.constant(0).rename("mndwi").updateMask(0)
    reducer = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)

    def monthly(m):
        s = ee.Date(START).advance(m, "month")
        comp = (landsat.filterDate(s, s.advance(1, "month"))
                       .merge(ee.ImageCollection([empty]))
                       .median())
        stack = comp.addBands(comp.gt(0).rename("water"))
        stats = stack.reduceRegions(collection=buffers, reducer=reducer, scale=30)
        return stats.map(lambda f: f.set("month", s.format("YYYY-MM")))

    results = ee.FeatureCollection(
        ee.List.sequence(0, N_MONTHS - 1).map(monthly)
    ).flatten()

    task = ee.batch.Export.table.toDrive(
        collection=results,
        description=name,
        folder="gee_mndwi",
        fileFormat="CSV",
        selectors=["lclty_d", "stat_cd", "month",
                   "mndwi_mean", "water_mean", "mndwi_count"],
    )
    task.start()
    return task


# ---------- Submit a limited number of new batches ----------
done = set(LOG_FILE.read_text().split()) if LOG_FILE.exists() else set()

submitted = 0
for name, batch_gdf in batches:
    if name in done:
        continue
    if submitted >= MAX_SUBMIT:
        break

    task = export_batch(batch_gdf, name)
    with open(LOG_FILE, "a") as f:
        f.write(name + "\n")

    print(f"Submitted {name} ({len(batch_gdf)} hotspots), task id {task.id}")
    submitted += 1

remaining = sum(1 for n, _ in batches if n not in done) - submitted
print(f"Submitted {submitted} tasks this run; {remaining} batches remaining.")