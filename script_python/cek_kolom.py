"""
cek_kolom.py
Nyari desa_jawa_tengah.geojson otomatis, lalu nampilin nama kolom
geojson dan dataset_kerawanan_desa.csv.

Jalankan dari root project (C:\\multihazard_jateng):
    python script_python\\cek_kolom.py
"""
import json
from pathlib import Path

import pandas as pd

# ---------- geojson ----------
ketemu = list(Path(".").rglob("desa_jawa_tengah.geojson"))
if not ketemu:
    raise SystemExit("desa_jawa_tengah.geojson gak ketemu. Pastikan jalanin dari C:\\multihazard_jateng")

path_geo = ketemu[0]
print(f"File geojson : {path_geo.resolve()}")

with open(path_geo, encoding="utf-8") as f:
    geo = json.load(f)

fitur = geo["features"]
print(f"Jumlah fitur : {len(fitur):,}")
print(f"Tipe geometri: {fitur[0]['geometry']['type']}")
print("\nKolom di geojson + 3 contoh isi:")
for k in fitur[0]["properties"]:
    contoh = [fitur[i]["properties"].get(k) for i in range(3)]
    print(f"  {k:<20} {contoh}")

# ---------- csv ----------
print("\n" + "=" * 60)
df = pd.read_csv(Path("data_olahan") / "dataset_kerawanan_desa.csv", nrows=3)
print("Kolom di dataset_kerawanan_desa.csv + 3 contoh isi:")
for k in df.columns:
    print(f"  {k:<24} {df[k].tolist()}")