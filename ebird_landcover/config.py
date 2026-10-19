import geopandas as gpd

# Update this path if your shapefile has a different name.
shapefile_path = "hotspots/ebird_hotspots.shp"

# Read the shapefile.
hotspots = gpd.read_file(shapefile_path)

# Print basic information.
print("\n--- Dataset overview ---")
print("Number of hotspots:", len(hotspots))
print("Coordinate reference system:", hotspots.crs)
print("Geometry types:")
print(hotspots.geom_type.value_counts())

print("\n--- Column names ---")
print(hotspots.columns.tolist())

print("\n--- First five rows ---")
print(hotspots.head())

print("\n--- Missing geometries ---")
print(hotspots.geometry.isna().sum())

print("\n--- Bounding box ---")
print(hotspots.total_bounds)