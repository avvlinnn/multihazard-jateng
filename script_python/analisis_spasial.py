"""
analisis_spasial.py
Bukti kuantitatif untuk pembahasan peta:
  1. Tabel akurasi, recall per kelas, dan bias prediksi per kabupaten/kota
  2. Peta ketepatan prediksi (benar vs salah) per desa
  3. Cek resolusi fitur hujan (berapa desa yang berbagi nilai hujan yang sama)

Prasyarat: peta_kerawanan.py sudah dijalankan (tidak perlu melatih ulang model).
Jalankan dari root project (C:\\multihazard_jateng):
    python script_python\\analisis_spasial.py

Output di hasil\\spasial\\:
    tabel_per_kabupaten.csv, peta_ketepatan.png
"""
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")
pd.options.display.float_format = "{:.2f}".format
pd.options.display.width = 250
pd.options.display.max_rows = 100

PATH_PRED = Path("hasil/peta/prediksi_oof_desa.csv")
PATH_GEO = Path("hasil/peta/desa_prediksi.geojson")
PATH_CSV = Path("data_olahan/dataset_kerawanan_desa.csv")
OUT = Path("hasil/spasial")
OUT.mkdir(parents=True, exist_ok=True)

KUNCI = ["none", "flood", "longsor", "multi"]
BG = "#f6f4ee"
WARNA_BENAR = "#cdd3c4"
WARNA_SALAH = "#a63d2f"
WARNA_KOSONG = "#b5b5b5"

# ---------- 1. Tabel per kabupaten ----------
p = pd.read_csv(PATH_PRED)
baris = []
for kab, d in p.groupby("kabupaten"):
    r = {"kabupaten": kab, "n_desa": len(d), "akurasi": d["benar"].mean() * 100}
    for k, nama in enumerate(KUNCI):
        aktual = d[d["kelas"] == k]
        r[f"n_aktual_{nama}"] = len(aktual)
        r[f"recall_{nama}"] = (aktual["pred_oof"] == k).mean() * 100 if len(aktual) else np.nan
        r[f"pct_aktual_{nama}"] = (d["kelas"] == k).mean() * 100
        r[f"pct_prediksi_{nama}"] = (d["pred_oof"] == k).mean() * 100
    baris.append(r)

tabel = pd.DataFrame(baris).sort_values("akurasi", ascending=False)
tabel.to_csv(OUT / "tabel_per_kabupaten.csv", index=False)

print("=== Akurasi per kabupaten/kota (%), diurutkan dari tertinggi ===")
kolom = ["kabupaten", "n_desa", "akurasi", "recall_flood", "recall_longsor", "recall_multi"]
print(tabel[kolom].to_string(index=False))

print("\n=== Bias prediksi: persen desa aktual vs prediksi (%) ===")
bias = tabel[["kabupaten", "pct_aktual_flood", "pct_prediksi_flood",
              "pct_aktual_longsor", "pct_prediksi_longsor",
              "pct_aktual_multi", "pct_prediksi_multi"]].copy()
bias["selisih_longsor"] = bias["pct_prediksi_longsor"] - bias["pct_aktual_longsor"]
bias = bias.sort_values("selisih_longsor", ascending=False)
print(bias.to_string(index=False))

print("\n=== Total seluruh desa: aktual vs prediksi (%) ===")
for k, nama in enumerate(KUNCI):
    print(f"{nama:<8} aktual {(p['kelas'] == k).mean() * 100:5.1f} | prediksi {(p['pred_oof'] == k).mean() * 100:5.1f}")

# ---------- 2. Peta ketepatan ----------
gdf = gpd.read_file(PATH_GEO)
ada = gdf["benar"].notna()
rasio = 1.0 / np.cos(np.deg2rad(7.2))

fig, ax = plt.subplots(figsize=(11, 6.4), facecolor=BG)
ax.set_facecolor(BG)
gdf[~ada].plot(ax=ax, color=WARNA_KOSONG, linewidth=0)
gdf[gdf["benar"] == 1].plot(ax=ax, color=WARNA_BENAR, edgecolor="#8f9486", linewidth=0.03)
gdf[gdf["benar"] == 0].plot(ax=ax, color=WARNA_SALAH, edgecolor="#6e2a20", linewidth=0.03)
ax.set_aspect(rasio)
ax.set_axis_off()
ax.set_title("Ketepatan prediksi per desa (out-of-fold)", fontsize=13, loc="left", pad=8)

n_benar, n_salah = int((gdf["benar"] == 1).sum()), int((gdf["benar"] == 0).sum())
handles = [
    Patch(facecolor=WARNA_BENAR, edgecolor="#8f9486", linewidth=0.5,
          label=f"Benar ({n_benar:,} desa, {n_benar / (n_benar + n_salah) * 100:.1f}%)"),
    Patch(facecolor=WARNA_SALAH, label=f"Salah ({n_salah:,} desa, {n_salah / (n_benar + n_salah) * 100:.1f}%)"),
    Patch(facecolor=WARNA_KOSONG, label=f"Tanpa data ({(~ada).sum()})"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10)
fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.08)
fig.savefig(OUT / "peta_ketepatan.png", dpi=300, facecolor=BG)
plt.close(fig)

# ---------- 3. Resolusi fitur hujan ----------
df = pd.read_csv(PATH_CSV)
print("\n=== Resolusi fitur hujan (berapa desa berbagi nilai yang sama) ===")
for kol in ["ch_tahunan_mean", "ch_harian_max", "ch_7hari_max", "hari_hujan_thn"]:
    unik = df[kol].nunique()
    dibagi = df.duplicated(kol, keep=False).mean() * 100
    print(f"{kol:<18} nilai unik: {unik:>5,} dari {len(df):,} desa | desa yang nilainya sama dengan desa lain: {dibagi:5.1f}%")

print(f"\nSelesai. Semua output ada di folder: {OUT.resolve()}")