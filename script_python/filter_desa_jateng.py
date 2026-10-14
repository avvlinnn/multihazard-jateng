"""
Filter file "indonesia_villages_border.geojson" (data desa se-Indonesia dari
Portal GIS Dukcapil Kemendagri, https://github.com/pararawendy/border-desa-indonesia-geojson)
jadi khusus provinsi Jawa Tengah, terus konversi ke format GeoJSON standar
(Polygon per desa) yang siap diupload sebagai GEE asset.

Cara pakai:
1. Download & extract dulu file sumbernya:
   https://raw.githubusercontent.com/pararawendy/border-desa-indonesia-geojson/master/indonesia_villages_border.geojson.zip
   taruh hasil extract-nya (indonesia_villages_border.geojson) di data_mentah/batas_wilayah/
2. Jalankan script ini dari root project:
   python script_python/filter_desa_jateng.py
"""

import json

INPUT_FILE = "data_mentah/batas_wilayah/indonesia_villages_border.geojson"
OUTPUT_FILE = "data_mentah/batas_wilayah/desa_jawa_tengah.geojson"
PROVINSI_TARGET = "JAWA TENGAH"


def to_geometry(border):
    """
    Data sumbernya nyimpen 'border' dalam 2 struktur beda:
    - ring polos: [[lon,lat], [lon,lat], ...]              -> Polygon biasa
    - nested (desa dgn bentuk kompleks/berpulau): [[[lon,lat], ...], ...] -> tiap elemen itu satu ring/part
    Fungsi ini deteksi otomatis strukturnya, return None kalau beneran kosong/gak valid.
    """
    if not border:
        return None
    first = border[0]
    if not first:
        return None
    if isinstance(first[0], (int, float)):
        # ring polos
        if len(border) < 3:
            return None
        return {"type": "Polygon", "coordinates": [border]}
    else:
        # nested: border isinya beberapa ring/part
        parts = [[ring] for ring in border if ring and len(ring) >= 3]
        if not parts:
            return None
        if len(parts) == 1:
            return {"type": "Polygon", "coordinates": parts[0]}
        return {"type": "MultiPolygon", "coordinates": parts}


def main():
    print(f"Baca {INPUT_FILE} (filenya lumayan gede, ~237 MB, agak lama sedikit)...")
    with open(INPUT_FILE, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Total desa se-Indonesia: {len(data)}")

    jateng = [d for d in data if d.get("province") == PROVINSI_TARGET]
    print(f"Total desa {PROVINSI_TARGET}: {len(jateng)}")

    features = []
    for d in jateng:
        geometry = to_geometry(d.get("border"))
        if geometry is None:
            print(f"⚠️  Dilewati (border kosong/gak valid): {d.get('village')}, {d.get('sub_district')}")
            continue
        features.append({
            "type": "Feature",
            "properties": {
                "provinsi": d.get("province"),
                "kabupaten": d.get("district"),
                "kecamatan": d.get("sub_district"),
                "desa": d.get("village"),
            },
            "geometry": geometry,
        })

    geojson = {"type": "FeatureCollection", "features": features}
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(geojson, f)

    print(f"\nDisimpan ke: {OUTPUT_FILE}")
    print(f"Jumlah fitur (desa) di output: {len(features)}")
    print(f"Jumlah kabupaten/kota unik: {len(set(d['district'] for d in jateng))}")


if __name__ == "__main__":
    main()