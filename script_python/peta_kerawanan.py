"""
peta_kerawanan.py
Peta kelas aktual vs prediksi per desa (RF + SMOTENC, prediksi OUT-OF-FOLD).

Tiap desa diprediksi oleh model yang tidak pernah melihat kecamatannya saat training,
jadi peta ini bisa dipakai untuk klaim akurasi spasial di paper.

Setting model, fold, dan seed sama persis dengan model_kerawanan.py.

Prasyarat: pip install geopandas
Jalankan dari root project (C:\\multihazard_jateng):
    python script_python\\peta_kerawanan.py

Output di hasil\\peta\\:
    prediksi_oof_desa.csv, desa_prediksi.geojson, peta_aktual_vs_prediksi.png
"""
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

try:
    import geopandas as gpd
except ImportError:
    raise SystemExit("geopandas belum ada. Jalankan dulu:  pip install geopandas")

warnings.filterwarnings("ignore")

SEED = 42
PATH_CSV = Path("data_olahan/dataset_kerawanan_desa.csv")
PATH_GEO = Path("data_mentah/batas_wilayah/desa_jawa_tengah.geojson")
OUT = Path("hasil/peta")
OUT.mkdir(parents=True, exist_ok=True)

NAMA_KELAS = ["None", "Flood", "Landslide", "Multi-hazard"]
WARNA = {0: "#e6dfcf", 1: "#2f6f9f", 2: "#c27c2c", 3: "#6e1a3a"}
WARNA_KOSONG = "#b5b5b5"
BG = "#f6f4ee"

KAT = ["tekstur_tanah", "tutupan_lahan"]
FITUR = ["elevasi", "kemiringan", "tekstur_tanah", "tutupan_lahan",
         "ch_tahunan_mean", "hari_hujan_thn", "hari_lebat_thn", "hari_sangat_lebat_thn",
         "ch_harian_max", "ch_3hari_max", "ch_7hari_max"]

# ---------- 1. Prediksi out-of-fold (sama dengan model_kerawanan.py) ----------
df = pd.read_csv(PATH_CSV)
df[KAT] = df[KAT].round().astype(int)
X = df[FITUR].reset_index(drop=True)
y = df["kelas"].to_numpy()
groups = pd.factorize(df["kabupaten"] + "|" + df["kecamatan"])[0]
cat_idx = [FITUR.index(c) for c in KAT]

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
pred = np.zeros(len(y), dtype=int)
f1_fold = []
for i, (tr, te) in enumerate(skf.split(X, y, groups), 1):
    model = Pipeline([
        ("smote", SMOTENC(categorical_features=cat_idx, k_neighbors=5, random_state=SEED)),
        ("clf", RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=SEED)),
    ])
    model.fit(X.iloc[tr], y[tr])
    pred[te] = model.predict(X.iloc[te])
    f1_fold.append(f1_score(y[te], pred[te], average="macro", labels=[0, 1, 2, 3], zero_division=0))
    print(f"Fold {i}/5 selesai | F1 macro = {f1_fold[-1]:.3f}")

print(f"\nCek reproduksi: F1 macro rata-rata fold = {np.mean(f1_fold):.3f} (harusnya ~0.441)")
print(f"Akurasi gabungan = {accuracy_score(y, pred):.3f} | balanced acc = {balanced_accuracy_score(y, pred):.3f}")

hasil = df[["provinsi", "kabupaten", "kecamatan", "desa", "kelas"]].copy()
hasil["pred_oof"] = pred
hasil["benar"] = (hasil["kelas"] == hasil["pred_oof"]).astype(int)
hasil.to_csv(OUT / "prediksi_oof_desa.csv", index=False)


# ---------- 2. Join ke geojson ----------
def buat_kunci(d):
    bagian = [d[k].astype(str).str.upper().str.replace(r"\s+", " ", regex=True).str.strip()
              for k in ("kabupaten", "kecamatan", "desa")]
    return bagian[0] + "|" + bagian[1] + "|" + bagian[2]


gdf = gpd.read_file(PATH_GEO)
if gdf.crs is None:
    gdf = gdf.set_crs(epsg=4326)
gdf["kunci"] = buat_kunci(gdf)
hasil["kunci"] = buat_kunci(hasil)

kembar_geo = gdf["kunci"].duplicated(keep=False)
kembar_csv = hasil["kunci"].duplicated(keep=False)
peta_kelas = hasil[~kembar_csv].set_index("kunci")

gdf["kelas_aktual"] = gdf["kunci"].map(peta_kelas["kelas"])
gdf["kelas_prediksi"] = gdf["kunci"].map(peta_kelas["pred_oof"])
gdf.loc[kembar_geo, ["kelas_aktual", "kelas_prediksi"]] = np.nan  # ambigu, jangan ditebak

ada = gdf["kelas_aktual"].notna()
print("\n=== Hasil join geojson dan CSV ===")
print(f"Polygon geojson            : {len(gdf):,}")
print(f"Baris CSV                  : {len(hasil):,}")
print(f"Polygon berhasil dicocokkan: {ada.sum():,}")
print(f"Polygon tanpa data (abu)   : {(~ada).sum():,} "
      f"(kunci kembar di geojson: {kembar_geo.sum():,}, kunci kembar di CSV: {kembar_csv.sum():,})")

gdf["benar"] = np.where(ada, (gdf["kelas_aktual"] == gdf["kelas_prediksi"]).astype(float), np.nan)
gdf.drop(columns="kunci").to_file(OUT / "desa_prediksi.geojson", driver="GeoJSON")

# ---------- 3. Peta ----------
fig, axes = plt.subplots(1, 2, figsize=(17, 7.2), facecolor=BG)
rasio = 1.0 / np.cos(np.deg2rad(7.2))  # koreksi aspek untuk koordinat derajat di sekitar lintang 7 S

for ax, kol, judul in zip(axes, ["kelas_aktual", "kelas_prediksi"],
                          ["Kelas aktual", "Prediksi RF + SMOTENC (out-of-fold)"]):
    ax.set_facecolor(BG)
    gdf[~ada].plot(ax=ax, color=WARNA_KOSONG, linewidth=0)
    for k in range(4):
        sub = gdf[gdf[kol] == k]
        if len(sub):
            sub.plot(ax=ax, color=WARNA[k], edgecolor="#7a7468", linewidth=0.03)
    ax.set_aspect(rasio)
    ax.set_axis_off()
    ax.set_title(judul, fontsize=13, loc="left", pad=8)

jumlah = gdf["kelas_aktual"].value_counts()
patches = [Patch(facecolor=WARNA[k], edgecolor="#7a7468", linewidth=0.5,
                 label=f"{NAMA_KELAS[k]} (aktual: {int(jumlah.get(k, 0)):,})") for k in range(4)]
if (~ada).any():
    patches.append(Patch(facecolor=WARNA_KOSONG, label=f"Tanpa data ({(~ada).sum():,})"))
fig.legend(handles=patches, loc="lower center", ncol=len(patches), frameon=False, fontsize=10)
fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.07, wspace=0.02)
fig.savefig(OUT / "peta_aktual_vs_prediksi.png", dpi=300, facecolor=BG)
plt.close(fig)

print(f"\nSelesai. Semua output ada di folder: {OUT.resolve()}")