import fiona
import geopandas as gpd

#gpkg = 'gnatsgo__south_manitou.gpkg'
gpkg = 'gNATSGO_02_03_2025.gpkg'

for layer in fiona.listlayers(gpkg):
    df = gpd.read_file(filename=gpkg, layer=layer, rows=10)
    print(f"Layer: {layer}")
    print("Columns:")
    for colname, coltype in df.dtypes.to_dict().items():
        print(f"column name: {colname} data type: {coltype}")
    print(f"Geometry type: {df.geometry.iloc[0].geom_type}\n")