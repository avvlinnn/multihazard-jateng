"""
Modeling kerawanan multi-hazard per desa.

Model  : Decision Tree, Random Forest, Random Forest + SMOTENC, XGBoost, XGBoost + SMOTENC
Validasi: StratifiedGroupKFold 5-fold, grup = kecamatan (desa bertetangga tidak terpisah antara train dan test)
SMOTENC : di dalam pipeline, jadi hanya menyentuh data train di tiap fold

Prasyarat: bangun_dataset_kerawanan.py sudah dijalankan.
Jalankan dari root project (C:\\multihazard_jateng):
    python script_python\\model_kerawanan.py

Output di folder hasil/:
    hasil_per_fold.csv, ringkasan_model.csv, importance_rf_smote.csv,
    confusion_*.csv, confusion_rf_smote.png, importance_rf_smote.png
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
pd.options.display.float_format = "{:.3f}".format
pd.options.display.width = 250

SEED = 42
HASIL = Path("hasil")
HASIL.mkdir(exist_ok=True)
NAMA_KELAS = ["None", "Flood", "Landslide", "Multi-hazard"]
LABEL = [0, 1, 2, 3]

# ---------- Data ----------
df = pd.read_csv("data_olahan/dataset_kerawanan_desa.csv")
KAT = ["tekstur_tanah", "tutupan_lahan"]
FITUR = ["elevasi", "kemiringan", "tekstur_tanah", "tutupan_lahan",
         "ch_tahunan_mean", "hari_hujan_thn", "hari_lebat_thn", "hari_sangat_lebat_thn",
         "ch_harian_max", "ch_3hari_max", "ch_7hari_max"]
df[KAT] = df[KAT].round().astype(int)
X = df[FITUR].reset_index(drop=True)
y = df["kelas"].to_numpy()
groups = pd.factorize(df["kabupaten"] + "|" + df["kecamatan"])[0]
cat_idx = [FITUR.index(c) for c in KAT]

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
lipatan = list(skf.split(X, y, groups))  # dipakai bersama semua model, jadi perbandingannya adil
for i, (tr, te) in enumerate(lipatan, 1):
    print(f"Fold {i}: train {len(tr):,} | test {len(te):,} | test per kelas {np.bincount(y[te], minlength=4).tolist()}")


# ---------- Konfigurasi model (hyperparameter sama untuk versi dengan/tanpa SMOTE) ----------
def dt():
    return DecisionTreeClassifier(min_samples_leaf=2, random_state=SEED)


def rf():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=SEED)


def xgb():
    return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.8,
                         colsample_bytree=0.8, objective="multi:softprob", eval_metric="mlogloss",
                         tree_method="hist", n_jobs=-1, random_state=SEED)


KONFIG = {
    "Decision Tree": (False, dt),
    "Random Forest": (False, rf),
    "Random Forest + SMOTE": (True, rf),
    "XGBoost": (False, xgb),
    "XGBoost + SMOTE": (True, xgb),
}

# ---------- Cross-validation ----------
baris, imp_rows = [], []
oof = {n: np.zeros(len(y), dtype=int) for n in KONFIG}

for nama, (pakai_smote, pabrik) in KONFIG.items():
    for i, (tr, te) in enumerate(lipatan, 1):
        langkah = []
        if pakai_smote:
            langkah.append(("smote", SMOTENC(categorical_features=cat_idx, k_neighbors=5, random_state=SEED)))
        langkah.append(("clf", pabrik()))
        model = Pipeline(langkah)
        model.fit(X.iloc[tr], y[tr])
        pred = model.predict(X.iloc[te])
        oof[nama][te] = pred

        p, r, f, _ = precision_recall_fscore_support(y[te], pred, labels=LABEL, zero_division=0)
        row = dict(model=nama, fold=i, akurasi=accuracy_score(y[te], pred),
                   balanced_acc=balanced_accuracy_score(y[te], pred), f1_macro=f.mean())
        for k in LABEL:
            row[f"f1_{k}"], row[f"recall_{k}"], row[f"prec_{k}"] = f[k], r[k], p[k]
        baris.append(row)

        if nama == "Random Forest + SMOTE":
            pi = permutation_importance(model, X.iloc[te], y[te], scoring="f1_macro",
                                        n_repeats=5, random_state=SEED, n_jobs=1)
            mdi = model.named_steps["clf"].feature_importances_
            for j, c in enumerate(FITUR):
                imp_rows.append(dict(fitur=c, fold=i, permutasi=pi.importances_mean[j], mdi=mdi[j]))
    rata = np.mean([b["f1_macro"] for b in baris if b["model"] == nama])
    print(f"{nama:<24} selesai | F1 macro rata-rata fold: {rata:.3f}")

# ---------- Ringkasan ----------
hasil = pd.DataFrame(baris)
hasil.to_csv(HASIL / "hasil_per_fold.csv", index=False)
ringkas = hasil.drop(columns="fold").groupby("model", sort=False).agg(["mean", "std"])
ringkas.columns = [f"{a}_{b}" for a, b in ringkas.columns]
ringkas.to_csv(HASIL / "ringkasan_model.csv")

print("\n=== RINGKASAN (rata-rata 5 fold) ===")
kolom = ["akurasi_mean", "balanced_acc_mean", "f1_macro_mean", "f1_macro_std",
         "f1_0_mean", "f1_1_mean", "f1_2_mean", "f1_3_mean", "recall_3_mean"]
print(ringkas[kolom].to_string())

for nama in ["Random Forest", "Random Forest + SMOTE"]:
    print(f"\n=== {nama} (prediksi out-of-fold gabungan) ===")
    print(classification_report(y, oof[nama], labels=LABEL, target_names=NAMA_KELAS, digits=3, zero_division=0))
    cm = confusion_matrix(y, oof[nama], labels=LABEL)
    berkas = nama.lower().replace(" + ", "_").replace(" ", "_")
    pd.DataFrame(cm, index=NAMA_KELAS, columns=NAMA_KELAS).to_csv(HASIL / f"confusion_{berkas}.csv")

# ---------- Confusion matrix RF + SMOTE ----------
cm = confusion_matrix(y, oof["Random Forest + SMOTE"], labels=LABEL)
cmn = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(5.4, 4.6))
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(LABEL, NAMA_KELAS, rotation=20)
ax.set_yticks(LABEL, NAMA_KELAS)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
for a in LABEL:
    for b in LABEL:
        ax.text(b, a, f"{cmn[a, b]:.2f}\n({cm[a, b]})", ha="center", va="center",
                color="white" if cmn[a, b] > 0.5 else "black", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig(HASIL / "confusion_rf_smote.png", dpi=300)
plt.close(fig)

# ---------- Feature importance RF + SMOTE ----------
imp = pd.DataFrame(imp_rows).groupby("fitur").agg(
    permutasi_mean=("permutasi", "mean"), permutasi_std=("permutasi", "std"), mdi_mean=("mdi", "mean")
).sort_values("permutasi_mean", ascending=False)
imp.to_csv(HASIL / "importance_rf_smote.csv")
print("\n=== FEATURE IMPORTANCE (RF + SMOTE) ===")
print(imp.to_string())

urut = imp.sort_values("permutasi_mean")
fig, ax = plt.subplots(figsize=(6.2, 4.4))
ax.barh(urut.index, urut["permutasi_mean"], xerr=urut["permutasi_std"], color="#3b5b7a", capsize=2)
ax.set_xlabel("Permutation importance (drop in macro-F1)")
fig.tight_layout()
fig.savefig(HASIL / "importance_rf_smote.png", dpi=300)
plt.close(fig)

print(f"\nSelesai. Semua output ada di folder: {HASIL.resolve()}")