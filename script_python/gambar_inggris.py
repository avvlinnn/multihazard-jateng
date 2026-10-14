r"""
gambar_inggris.py

Bikin ulang Gambar 1-3 naskah dengan label bahasa Inggris TANPA melatih ulang model.
Hanya membaca hasil yang sudah tersimpan:
    hasil\confusion_random_forest_smote.csv
    hasil\importance_rf_smote.csv
    hasil\peta\desa_prediksi.geojson

Jalankan dari root project (C:\multihazard_jateng):
    python script_python\gambar_inggris.py

Output di hasil\gambar_inggris\:
    fig1_confusion_matrix.png
    fig2_feature_importance.png
    fig3_map_actual_vs_predicted.png
Lebar gambar disesuaikan dengan lebar teks template (sekitar 15 cm), resolusi 300 dpi.
"""
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

try:
    import geopandas as gpd
except ImportError:
    raise SystemExit("geopandas belum ada. Jalankan dulu:  pip install geopandas")

warnings.filterwarnings("ignore")

HASIL = Path("hasil")
OUT = HASIL / "gambar_inggris"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
})

KELAS = ["None recorded", "Flood", "Landslide", "Multi-hazard"]
LABEL = [0, 1, 2, 3]

NAMA_FITUR = {
    "elevasi": "Elevation",
    "kemiringan": "Slope",
    "tekstur_tanah": "Soil texture",
    "tutupan_lahan": "Land cover",
    "ch_tahunan_mean": "Mean annual rainfall",
    "hari_hujan_thn": "Rainy days per year (≥1 mm)",
    "hari_lebat_thn": "Heavy-rain days per year (≥50 mm)",
    "hari_sangat_lebat_thn": "Very-heavy-rain days per year (≥100 mm)",
    "ch_harian_max": "Maximum daily rainfall",
    "ch_3hari_max": "Maximum 3-day rainfall",
    "ch_7hari_max": "Maximum 7-day rainfall",
}

# ---------- Gambar 1: confusion matrix ----------
cm = pd.read_csv(HASIL / "confusion_random_forest_smote.csv", index_col=0).to_numpy()
cmn = cm / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(4.8, 4.2))
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(LABEL, KELAS, rotation=20, ha="right")
ax.set_yticks(LABEL, KELAS)
ax.set_xlabel("Predicted class")
ax.set_ylabel("Actual class")
for a in LABEL:
    for b in LABEL:
        ax.text(b, a, f"{cmn[a, b]:.2f}\n({cm[a, b]:,})", ha="center", va="center",
                color="white" if cmn[a, b] > 0.5 else "black", fontsize=8)
cb = fig.colorbar(im, ax=ax, fraction=0.046)
cb.set_label("Row-normalized proportion")
fig.tight_layout()
fig.savefig(OUT / "fig1_confusion_matrix.png", dpi=300)
plt.close(fig)

# ---------- Gambar 2: permutation importance ----------
imp = pd.read_csv(HASIL / "importance_rf_smote.csv", index_col=0)
imp.index = [NAMA_FITUR.get(i, i) for i in imp.index]
urut = imp.sort_values("permutasi_mean")

fig, ax = plt.subplots(figsize=(5.9, 3.6))
ax.barh(urut.index, urut["permutasi_mean"], xerr=urut["permutasi_std"],
        color="#3b5b7a", capsize=2, error_kw={"elinewidth": 0.8})
ax.axvline(0, color="#444444", linewidth=0.6)
ax.set_xlabel("Permutation importance (decrease in macro-F1)")
ax.tick_params(axis="y", labelsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig2_feature_importance.png", dpi=300)
plt.close(fig)

# ---------- Gambar 3: peta aktual vs prediksi ----------
WARNA = {0: "#e6dfcf", 1: "#2f6f9f", 2: "#c27c2c", 3: "#6e1a3a"}
WARNA_KOSONG = "#b5b5b5"
BG = "#ffffff"

gdf = gpd.read_file(HASIL / "peta" / "desa_prediksi.geojson")
ada = gdf["kelas_aktual"].notna()
rasio = 1.0 / np.cos(np.deg2rad(7.2))

fig, axes = plt.subplots(1, 2, figsize=(5.9, 2.9), facecolor=BG)
for ax, kol, judul in zip(axes, ["kelas_aktual", "kelas_prediksi"],
                          ["(a) Actual class", "(b) Predicted class (out-of-fold)"]):
    gdf[~ada].plot(ax=ax, color=WARNA_KOSONG, linewidth=0)
    for k in LABEL:
        sub = gdf[gdf[kol] == k]
        if len(sub):
            sub.plot(ax=ax, color=WARNA[k], edgecolor="#7a7468", linewidth=0.02)
    ax.set_aspect(rasio)
    ax.set_axis_off()
    ax.set_title(judul, fontsize=8.5, loc="left", pad=3)

jumlah = gdf["kelas_aktual"].value_counts()
patches = [Patch(facecolor=WARNA[k], edgecolor="#7a7468", linewidth=0.4,
                 label=f"{KELAS[k]} (n = {int(jumlah.get(k, 0)):,})") for k in LABEL]
if (~ada).any():
    patches.append(Patch(facecolor=WARNA_KOSONG, label=f"No data (n = {int((~ada).sum()):,})"))
fig.legend(handles=patches, loc="lower center", ncol=3, frameon=False, fontsize=7.5)
fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.17, wspace=0.03)
fig.savefig(OUT / "fig3_map_actual_vs_predicted.png", dpi=300, facecolor=BG)
plt.close(fig)

print(f"Selesai. Gambar ada di: {OUT.resolve()}")