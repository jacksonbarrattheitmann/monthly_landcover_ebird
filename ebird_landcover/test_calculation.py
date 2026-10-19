
from pathlib import Path

import geopandas as gpd
import numpy as np
import planetary_computer as pc
import rasterio
from rasterio.mask import mask
from pystac_client import Client
from shapely.geometry import mapping
from shapely.ops import transform
from pyproj import Transformer

# --------------------------------------------------
# 1. Load the hotspot data
# --------------------------------------------------

project_dir = Path(__file__).resolve().parent
shapefile_path = (
    project_dir / "ebird_hotspots" / "wost_localties.shp"
)

hotspots = gpd.read_file(shapefile_path)

# Use the first hotspot for this test
hotspot = hotspots.iloc[0]
point = hotspot.geometry

print("Hotspot:", hotspot["lclty_d"])
print("State:", hotspot["stat_cd"])

# --------------------------------------------------
# 2. Create a 500 m buffer
# --------------------------------------------------

# This Florida hotspot is in UTM zone 17N.
# EPSG:32617 uses meters.
project_to_utm = Transformer.from_crs(
    "EPSG:4326", "EPSG:32617", always_xy=True
).transform

projected_point = transform(project_to_utm, point)
buffer_geometry = projected_point.buffer(5000)

# --------------------------------------------------
# 3. Find a relatively clear Landsat 8 scene
# --------------------------------------------------

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

search = catalog.search(
    collections=["landsat-c2-l2"],
    intersects=point.__geo_interface__,
    datetime="2021-03-01/2021-03-31",
    query={"eo:cloud_cover": {"lt": 80}},
    max_items=100,
)

items = list(search.items())

landsat8_items = [
    item for item in items
    if item.properties.get("platform") == "landsat-8"
]

if not landsat8_items:
    raise RuntimeError("No Landsat 8 scenes found for March 2021.")

# Lowest scene-level cloud cover first
landsat8_items.sort(
    key=lambda item: item.properties.get("eo:cloud_cover", 100)
)

item = pc.sign(landsat8_items[0])

print("\nSelected scene:", item.id)
print("Acquisition date:", item.datetime)
print(
    "Scene cloud cover:",
    item.properties.get("eo:cloud_cover"),
)

# --------------------------------------------------
# 4. Read and mask the satellite bands
# --------------------------------------------------

required_bands = [
    "green", "red", "nir08", "swir16", "qa_pixel"
]

missing = [b for b in required_bands if b not in item.assets]
if missing:
    raise RuntimeError(f"Missing required assets: {missing}")

bands = {}

for band_name in required_bands:
    asset_url = item.assets[band_name].href

    with rasterio.open(asset_url) as src:
        # Reproject the buffer into the raster's CRS.
        project_buffer = Transformer.from_crs(
            "EPSG:32617", src.crs, always_xy=True
        ).transform

        raster_buffer = transform(
            project_buffer, buffer_geometry
        )

        # Crop to the buffer. Outside-buffer pixels are masked.
        data, _ = mask(
            src,
            [mapping(raster_buffer)],
            crop=True,
            filled=False,
        )

        bands[band_name] = data[0]

# The masked arrays preserve the buffer boundary.
green_raw = bands["green"]
red_raw = bands["red"]
nir_raw = bands["nir08"]
swir_raw = bands["swir16"]
qa = bands["qa_pixel"]

# --------------------------------------------------
# 5. Exclude cloudy and invalid pixels
# --------------------------------------------------

# QA_PIXEL bits 0–5:
# fill, dilated cloud, cirrus, cloud, cloud shadow, snow
qa_values = np.asarray(qa.filled(0), dtype=np.uint16)
qa_masked = np.ma.getmaskarray(qa)

bad_bits = sum(1 << bit for bit in range(6))
clear_pixels = (qa_values & bad_bits) == 0

# Require all spectral bands to be inside the buffer.
# Also exclude pixels masked in any band.
outside_buffer = (
    np.ma.getmaskarray(green_raw)
    | np.ma.getmaskarray(red_raw)
    | np.ma.getmaskarray(nir_raw)
    | np.ma.getmaskarray(swir_raw)
    | qa_masked
)

valid = clear_pixels & ~outside_buffer

# Landsat Collection 2 Level-2 surface reflectance scaling
def scale_reflectance(array):
    # Convert to floating point before inserting NaN values.
    values = array.astype("float32").filled(np.nan)
    return values * 0.0000275 - 0.2

green = scale_reflectance(green_raw)
red = scale_reflectance(red_raw)
nir = scale_reflectance(nir_raw)
swir = scale_reflectance(swir_raw)

# Require finite reflectance values
valid &= (
    np.isfinite(green)
    & np.isfinite(red)
    & np.isfinite(nir)
    & np.isfinite(swir)
)

# --------------------------------------------------
# 6. Calculate water and vegetation indices
# --------------------------------------------------

mndwi_denominator = green + swir
ndvi_denominator = nir + red

mndwi = np.full(green.shape, np.nan, dtype="float32")
ndvi = np.full(green.shape, np.nan, dtype="float32")

np.divide(
    green - swir,
    mndwi_denominator,
    out=mndwi,
    where=np.abs(mndwi_denominator) > 1e-6,
)

np.divide(
    nir - red,
    ndvi_denominator,
    out=ndvi,
    where=np.abs(ndvi_denominator) > 1e-6,
)

valid &= np.isfinite(mndwi) & np.isfinite(ndvi)

water = valid & (mndwi > 0)
vegetation = valid & (ndvi >= 0.30) & ~water

# --------------------------------------------------
# 7. Print the test results
# --------------------------------------------------

valid_count = int(valid.sum())

print("\n--- 5000 m buffer results ---")
print("Pixels in cropped window:", valid.size)
print("Valid clear pixels:", valid_count)

if valid_count == 0:
    print("No valid pixels. Check scene coverage and masking.")
else:
    print(
        "Water cover (%):",
        round(100 * water.sum() / valid_count, 2),
    )
    print(
        "Vegetation cover (%):",
        round(100 * vegetation.sum() / valid_count, 2),
    )
    print(
        "Other valid pixels (%):",
        round(
            100 * (valid_count - water.sum() - vegetation.sum())
            / valid_count,
            2,
        ),
    )

print("\nTest complete.")
