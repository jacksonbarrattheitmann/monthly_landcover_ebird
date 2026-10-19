import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from pystac_client import Client
import planetary_computer as pc
from shapely.geometry import mapping
from tqdm import tqdm


# ============================================================
# SETTINGS
# ============================================================

PROJECT_DIR = Path.home() / "Documents" / "ebird_landcover"

HOTSPOT_FILE = PROJECT_DIR / "ebird_hotspots" / "localities_clean.shp"

OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR / "hotspot_monthly_cover_best_scene_2010_2026.csv"
)

START_DATE = "2010-01-01"
END_DATE   = "2026-10-13"

BUFFER_M     = 5000
MAX_CLOUD    = 80

MIN_VALID_COVERAGE_PCT = 80.0

MNDWI_WATER_THRESHOLD      = 0.0
NDVI_VEGETATION_THRESHOLD  = 0.30

COLLECTION = "landsat-c2-l2"

REQUIRED_ASSETS = {"green", "red", "nir08", "swir16", "qa_pixel"}

ALL_MONTHS = pd.period_range(start=START_DATE, end="2026-10", freq="M")


# ============================================================
# HELPERS
# ============================================================

def get_utm_epsg(lon, lat):
    """Choose a local UTM CRS from the hotspot coordinates."""
    zone = int((lon + 180) // 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def scale_reflectance(array):
    """
    Apply Landsat Collection 2 surface reflectance scaling.
    Handles both np.ma.MaskedArray and plain ndarray inputs.
    """
    if isinstance(array, np.ma.MaskedArray):
        values = array.astype("float32").filled(np.nan)
    else:
        values = array.astype("float32")
    return values * 0.0000275 - 0.2


def read_clipped_band(asset_url, geometry):
    """
    Read a raster band clipped to a buffer geometry.
    Reprojects the geometry to the raster CRS before masking.
    """
    with rasterio.open(asset_url) as src:
        geometry_raster_crs = (
            gpd.GeoSeries([geometry], crs="EPSG:4326")
            .to_crs(src.crs)
            .iloc[0]
        )
        result, _ = mask(
            src,
            [mapping(geometry_raster_crs)],
            crop=True,
            filled=False,
        )
        return result[0]


def read_qa_band(asset_url, geometry):
    """Read an integer QA band clipped to the buffer."""
    return read_clipped_band(asset_url, geometry)


def scene_sensor(item):
    """Return a readable sensor label."""
    platform = item.properties.get("platform", "unknown")
    return {
        "landsat-4": "Landsat 4",
        "landsat-5": "Landsat 5",
        "landsat-7": "Landsat 7",
        "landsat-8": "Landsat 8",
        "landsat-9": "Landsat 9",
    }.get(platform, platform)


def get_scene_coverage(item, geometry):
    """
    Estimate clear-pixel coverage within the buffer portion
    represented by the candidate Landsat scene.
    """
    try:
        signed_item = pc.sign(item)

        if "qa_pixel" not in signed_item.assets:
            print(f"{item.id}: missing qa_pixel asset")
            return None

        with rasterio.open(signed_item.assets["qa_pixel"].href) as src:
            geometry_raster_crs = (
                gpd.GeoSeries([geometry], crs="EPSG:4326")
                .to_crs(src.crs)
                .iloc[0]
            )
            qa, _ = mask(
                src,
                [mapping(geometry_raster_crs)],
                crop=True,
                filled=False,
            )
            qa = qa[0]

        masked           = np.ma.getmaskarray(qa)
        qa_values        = qa.filled(0).astype("uint16")
        inside           = ~masked
        total_buffer_pixels = int(inside.sum())

        if total_buffer_pixels == 0:
            print(f"{item.id}: zero pixels inside buffer")
            return None

        platform = item.properties.get("platform", "")
        bad_bits = (1 << 0) | (1 << 1) | (1 << 3) | (1 << 4) | (1 << 5)
        if platform in ("landsat-8", "landsat-9"):
            bad_bits |= 1 << 2

        clear = (qa_values & bad_bits) == 0
        clear &= inside

        return 100.0 * int(clear.sum()) / total_buffer_pixels

    except Exception as e:
        print(f"Coverage assessment failed for {item.id}: {e}")
        return None


def calculate_scene_cover(item, geometry):
    """Calculate exclusive water, vegetation, and other cover."""
    signed_item = pc.sign(item)
    assets      = signed_item.assets

    if not REQUIRED_ASSETS.issubset(assets.keys()):
        raise ValueError("Scene is missing required spectral or QA bands.")

    green_raw = read_clipped_band(assets["green"].href,   geometry)
    red_raw   = read_clipped_band(assets["red"].href,     geometry)
    nir_raw   = read_clipped_band(assets["nir08"].href,   geometry)
    swir_raw  = read_clipped_band(assets["swir16"].href,  geometry)
    qa_raw    = read_qa_band(     assets["qa_pixel"].href, geometry)

    if not (
        green_raw.shape == red_raw.shape ==
        nir_raw.shape   == swir_raw.shape == qa_raw.shape
    ):
        raise ValueError("Band dimensions do not match.")

    outside = (
        np.ma.getmaskarray(green_raw)
        | np.ma.getmaskarray(red_raw)
        | np.ma.getmaskarray(nir_raw)
        | np.ma.getmaskarray(swir_raw)
        | np.ma.getmaskarray(qa_raw)
    )

    green = scale_reflectance(green_raw)
    red   = scale_reflectance(red_raw)
    nir   = scale_reflectance(nir_raw)
    swir  = scale_reflectance(swir_raw)

    qa_values = qa_raw.filled(0)
    platform  = item.properties.get("platform", "")

    bad_bits = (1 << 0) | (1 << 1) | (1 << 3) | (1 << 4) | (1 << 5)
    if platform in ("landsat-8", "landsat-9"):
        bad_bits |= 1 << 2

    clear = (qa_values & bad_bits) == 0

    if "qa_radsat" in assets:
        radsat_raw = read_qa_band(assets["qa_radsat"].href, geometry)
        if radsat_raw.shape != qa_raw.shape:
            raise ValueError("QA_RADSAT dimensions do not match QA_PIXEL.")
        clear   &= (radsat_raw.filled(0) == 0)
        outside |= np.ma.getmaskarray(radsat_raw)

    valid = (
        clear
        & ~outside
        & np.isfinite(green)
        & np.isfinite(red)
        & np.isfinite(nir)
        & np.isfinite(swir)
    )

    denominator_mndwi = green + swir
    denominator_ndvi  = nir   + red

    mndwi = np.full(green.shape, np.nan, dtype="float32")
    ndvi  = np.full(green.shape, np.nan, dtype="float32")

    np.divide(
        green - swir, denominator_mndwi, out=mndwi,
        where=np.abs(denominator_mndwi) > 1e-6,
    )
    np.divide(
        nir - red, denominator_ndvi, out=ndvi,
        where=np.abs(denominator_ndvi) > 1e-6,
    )

    valid &= np.isfinite(mndwi) & np.isfinite(ndvi)

    total_pixels  = int((~outside).sum())
    valid_pixels  = int(valid.sum())
    coverage_pct  = (
        100.0 * valid_pixels / total_pixels
        if total_pixels > 0 else np.nan
    )

    if valid_pixels == 0:
        return {
            "valid_pixel_observations": 0,
            "valid_coverage_pct":       coverage_pct,
            "water_pct":      np.nan,
            "vegetation_pct": np.nan,
            "other_pct":      np.nan,
        }

    water      = valid & (mndwi > MNDWI_WATER_THRESHOLD)
    vegetation = valid & (ndvi >= NDVI_VEGETATION_THRESHOLD) & ~water
    other      = valid & ~water & ~vegetation

    return {
        "valid_pixel_observations": valid_pixels,
        "valid_coverage_pct":       coverage_pct,
        "water_pct":      100.0 * water.sum()      / valid_pixels,
        "vegetation_pct": 100.0 * vegetation.sum() / valid_pixels,
        "other_pct":      100.0 * other.sum()      / valid_pixels,
    }


def build_buffer(hotspot_geom, hotspots_crs):
    """Reproject a hotspot point to UTM, buffer it, return WGS84 polygon."""
    hotspot_wgs84 = (
        gpd.GeoSeries([hotspot_geom], crs=hotspots_crs)
        .to_crs("EPSG:4326")
        .iloc[0]
    )
    lon, lat = hotspot_wgs84.x, hotspot_wgs84.y
    utm_epsg = get_utm_epsg(lon, lat)

    buffer_wgs84 = (
        gpd.GeoSeries([hotspot_wgs84], crs="EPSG:4326")
        .to_crs(f"EPSG:{utm_epsg}")
        .buffer(BUFFER_M)
        .to_crs("EPSG:4326")
        .iloc[0]
    )
    return hotspot_wgs84, utm_epsg, buffer_wgs84


def process_month(
    month_period, month_items, hotspot_id, state, buffer_wgs84
):
    """Return a result row dict for one month."""
    month    = str(month_period)
    base_row = {
        "hotspot_id": hotspot_id,
        "state":      state,
        "month":      month,
        "buffer_m":   BUFFER_M,
    }
    empty_cover = {
        "selected_scene_id":        None,
        "acquisition_date":         None,
        "sensor":                   None,
        "scene_cloud_cover_pct":    np.nan,
        "valid_pixel_observations": 0,
        "valid_coverage_pct":       np.nan,
        "water_pct":                np.nan,
        "vegetation_pct":           np.nan,
        "other_pct":                np.nan,
    }

    if not month_items:
        return {**base_row, **empty_cover, "quality_flag": "NO_SCENES"}

    candidates = []
    for item in month_items:
        try:
            coverage = get_scene_coverage(item, buffer_wgs84)
            if coverage is None:
                continue
            candidates.append({
                "item":        item,
                "coverage":    coverage,
                "cloud_cover": item.properties.get("eo:cloud_cover", np.nan),
            })
        except Exception as exc:
            warnings.warn(f"Could not assess scene {item.id}: {exc}")

    if not candidates:
        return {
            **base_row, **empty_cover,
            "quality_flag": "NO_ASSESSABLE_SCENES",
        }

    candidates.sort(
        key=lambda c: (
            c["coverage"],
            -(c["cloud_cover"] if pd.notna(c["cloud_cover"]) else 100.0),
            c["item"].datetime.timestamp(),
        ),
        reverse=True,
    )

    selected = candidates[0]
    item     = selected["item"]

    scene_meta = {
        "selected_scene_id":     item.id,
        "acquisition_date":      item.datetime.strftime("%Y-%m-%d"),
        "sensor":                scene_sensor(item),
        "scene_cloud_cover_pct": item.properties.get("eo:cloud_cover", np.nan),
    }

    try:
        cover        = calculate_scene_cover(item, buffer_wgs84)
        valid_cov    = cover["valid_coverage_pct"]
        quality_flag = (
            "NO_VALID_PIXELS" if not np.isfinite(valid_cov)
            else "LOW_COVERAGE" if valid_cov < MIN_VALID_COVERAGE_PCT
            else "PASS"
        )
        return {**base_row, **scene_meta, **cover, "quality_flag": quality_flag}

    except Exception as exc:
        warnings.warn(
            f"Could not calculate cover for {month}, scene {item.id}: {exc}"
        )
        return {
            **base_row, **scene_meta,
            "valid_pixel_observations": 0,
            "valid_coverage_pct":       selected["coverage"],
            "water_pct":      np.nan,
            "vegetation_pct": np.nan,
            "other_pct":      np.nan,
            "quality_flag":   "PROCESSING_ERROR",
        }


# ============================================================
# LOAD HOTSPOTS
# ============================================================

print("Loading hotspot shapefile...")
hotspots = gpd.read_file(HOTSPOT_FILE)

if hotspots.empty:
    raise ValueError("The hotspot shapefile contains no features.")
if hotspots.crs is None:
    raise ValueError("The hotspot shapefile has no defined CRS.")

print(f"Total localities to process: {len(hotspots)}")


# ============================================================
# CHECKPOINT: load any already-completed rows
# ============================================================

if OUTPUT_FILE.exists():
    df_existing     = pd.read_csv(OUTPUT_FILE)
    completed_ids   = set(df_existing["hotspot_id"].unique())
    all_results     = df_existing.to_dict("records")
    print(
        f"Resuming — {len(completed_ids)} localities already done, "
        f"{len(hotspots) - len(completed_ids)} remaining."
    )
else:
    df_existing   = None
    completed_ids = set()
    all_results   = []


# ============================================================
# CONNECT TO CATALOG (once, outside the locality loop)
# ============================================================

print("Connecting to Planetary Computer...")
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)


# ============================================================
# MAIN LOOP: one locality at a time
# ============================================================

for _, hotspot in tqdm(
    hotspots.iterrows(), total=len(hotspots), desc="Localities"
):
    hotspot_id = hotspot["lclty_d"]
    state      = hotspot["stat_cd"]

    # Skip localities already written to the checkpoint file.
    if hotspot_id in completed_ids:
        continue

    # --- Build buffer geometry ---
    try:
        hotspot_wgs84, utm_epsg, buffer_wgs84 = build_buffer(
            hotspot.geometry, hotspots.crs
        )
    except Exception as e:
        warnings.warn(f"Geometry build failed for {hotspot_id}: {e}")
        continue

    # --- Search for Landsat scenes ---
    try:
        search = catalog.search(
            collections=[COLLECTION],
            intersects=buffer_wgs84.__geo_interface__,
            datetime=f"{START_DATE}/{END_DATE}",
        )
        items = [
            item for item in search.items()
            if REQUIRED_ASSETS.issubset(item.assets.keys())
        ]
    except Exception as e:
        warnings.warn(f"Scene search failed for {hotspot_id}: {e}")
        continue

    items.sort(key=lambda i: i.datetime)

    items_by_month: dict = {}
    for item in items:
        month_key = item.datetime.strftime("%Y-%m")
        items_by_month.setdefault(month_key, []).append(item)

    # --- Process every month for this locality ---
    locality_results = []
    for month_period in ALL_MONTHS:
        row = process_month(
            month_period,
            items_by_month.get(str(month_period), []),
            hotspot_id,
            state,
            buffer_wgs84,
        )
        locality_results.append(row)

    all_results.extend(locality_results)

    # --- Write checkpoint after every completed locality ---
    pd.DataFrame(all_results).to_csv(OUTPUT_FILE, index=False)


# ============================================================
# FINAL SAVE + SUMMARY
# ============================================================

df = pd.DataFrame(all_results)
df.to_csv(OUTPUT_FILE, index=False)

print("\n--- FINAL SUMMARY ---")
print(f"Total locality-months in output : {len(df)}")
print(f"Months with selected scenes     : {df['selected_scene_id'].notna().sum()}")
print(f"Months passing coverage threshold: {(df['quality_flag'] == 'PASS').sum()}")
print(f"Months with low coverage        : {(df['quality_flag'] == 'LOW_COVERAGE').sum()}")
print(f"Months with no scenes           : {(df['quality_flag'] == 'NO_SCENES').sum()}")
print(f"CSV saved to: {OUTPUT_FILE}")

print("\nLowest water cover (among calculated months):")
print(
    df.dropna(subset=["water_pct"])
      .nsmallest(10, "water_pct")[
          [
              "hotspot_id",
              "month",
              "selected_scene_id",
              "acquisition_date",
              "sensor",
              "valid_coverage_pct",
              "water_pct",
              "vegetation_pct",
              "quality_flag",
          ]
      ]
      .to_string(index=False)
)