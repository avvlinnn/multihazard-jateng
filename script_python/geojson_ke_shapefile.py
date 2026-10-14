"""
Konversi desa_jawa_tengah.geojson jadi Shapefile (.shp/.shx/.dbf/.prj),
terus di-zip jadi satu file - GEE cuma nerima shapefile (atau zip berisi
shapefile), bukan .geojson langsung.

Butuh library pyshp (pure Python, gak perlu install GDAL):
    pip install pyshp

Cara pakai, dari root project:
    python script_python/geojson_ke_shapefile.py
"""

import json
import os
import zipfile
import shapefile  # dari pip install pyshp

INPUT_FILE = "data_mentah/batas_wilayah/desa_jawa_tengah.geojson"
OUTPUT_DIR = "data_mentah/batas_wilayah/desa_jawa_tengah_shp"
OUTPUT_ZIP = "data_mentah/batas_wilayah/desa_jawa_tengah_shp.zip"
OUTPUT_ZIP_NOPRJ = "data_mentah/batas_wilayah/desa_jawa_tengah_shp_noprj.zip"

# Isi standar file .prj buat WGS84 (EPSG:4326) - datum yang dipakai data ini
WKT_WGS84 = (
    'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
    'SPHEROID["WGS_1984",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]'
)


def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        geojson = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    shp_path = os.path.join(OUTPUT_DIR, "desa_jawa_tengah")

    with shapefile.Writer(shp_path, shapeType=shapefile.POLYGON) as w:
        w.field("provinsi", "C", size=50)
        w.field("kabupaten", "C", size=50)
        w.field("kecamatan", "C", size=50)
        w.field("desa", "C", size=50)

        for feat in geojson["features"]:
            geom = feat["geometry"]
            props = feat["properties"]

            if geom["type"] == "Polygon":
                rings = geom["coordinates"]
            elif geom["type"] == "MultiPolygon":
                rings = [ring for part in geom["coordinates"] for ring in part]
            else:
                continue

            w.poly(rings)
            w.record(
                props.get("provinsi", ""),
                props.get("kabupaten", ""),
                props.get("kecamatan", ""),
                props.get("desa", ""),
            )

    with open(shp_path + ".prj", "w") as f:
        f.write(WKT_WGS84)

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for ext in [".shp", ".shx", ".dbf", ".prj"]:
            zf.write(shp_path + ext, arcname="desa_jawa_tengah" + ext)

    # Versi tanpa .prj - GEE otomatis anggap WGS84 kalau prj gak ada,
    # coba ini dulu kalau upload versi lengkap gagal di web UI.
    with zipfile.ZipFile(OUTPUT_ZIP_NOPRJ, "w", zipfile.ZIP_DEFLATED) as zf:
        for ext in [".shp", ".shx", ".dbf"]:
            zf.write(shp_path + ext, arcname="desa_jawa_tengah" + ext)

    print(f"Selesai. Ada 2 pilihan file buat diupload ke GEE:")
    print(f"  1. {OUTPUT_ZIP} (lengkap dgn .prj)")
    print(f"  2. {OUTPUT_ZIP_NOPRJ} (tanpa .prj - coba ini duluan kalau yg pertama gagal)")


if __name__ == "__main__":
    main()