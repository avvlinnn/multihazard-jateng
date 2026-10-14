r"""
model_tambahan.py  |  Sisa Fase 5 (disamakan dengan model_kerawanan.py)

Isi:
  1. RF, RF + class_weight, RF + SMOTENC, RF + SMOTENC (tuned)
     dibandingkan di fold yang SAMA dengan model_kerawanan.py
     (StratifiedGroupKFold 5-fold, seed 42, grup = kabupaten|kecamatan)
  2. Tuning hyperparameter di dalam fold (nested), tidak bocor ke data uji
  3. Confusion matrix ternormalisasi per kelas (gabungan prediksi out-of-fold)
  4. Permutasi per KELOMPOK fitur (topografi, tanah, lahan, hujan),
     dihitung pada dua model: RF + SMOTENC (tuned) dan RF + SMOTENC (bawaan)

Cek reproduksi: RF dan RF + SMOTENC di sini harus menghasilkan
F1 macro rata-rata fold sekitar 0.456 dan 0.472 (sama dengan model_kerawanan.py,
label hasil pemulihan desa).

Jalankan dari root project (C:\multihazard_jateng):
    python script_python\model_tambahan.py

Output ke folder hasil\tambahan
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# KONFIGURASI
# ------------------------------------------------------------------
PATH_DATA = os.path.join("data_olahan", "dataset_kerawanan_desa.csv")
DIR_OUT = os.path.join("hasil", "tambahan")

SEED = 42
N_SPLIT = 5
N_ITER_TUNING = 15      # turunin ke 8 kalau kelamaan
N_REPEAT_PERM = 10

KAT = ["tekstur_tanah", "tutupan_lahan"]
FITUR = ["elevasi", "kemiringan", "tekstur_tanah", "tutupan_lahan",
         "ch_tahunan_mean", "hari_hujan_thn", "hari_lebat_thn", "hari_sangat_lebat_thn",
         "ch_harian_max", "ch_3hari_max", "ch_7hari_max"]

KELOMPOK = {
    "Topografi (elevasi, kemiringan)": ["elevasi", "kemiringan"],
    "Tanah (tekstur_tanah)": ["tekstur_tanah"],
    "Lahan (tutupan_lahan)": ["tutupan_lahan"],
    "Curah hujan (7 fitur)": [
        "ch_tahunan_mean", "hari_hujan_thn", "hari_lebat_thn", "hari_sangat_lebat_thn",
        "ch_harian_max", "ch_3hari_max", "ch_7hari_max",
    ],
}

NAMA_KELAS = ["None", "Flood", "Landslide", "Multi-hazard"]
LABEL = [0, 1, 2, 3]

PARAM_TUNING = {
    "rf__n_estimators": [300, 500],
    "rf__max_depth": [None, 10, 15, 20, 30],
    "rf__min_samples_leaf": [1, 2, 3, 5, 8],
    "rf__max_features": ["sqrt", "log2", 0.5],
}


# ------------------------------------------------------------------
# UTIL
# ------------------------------------------------------------------
def hitung_metrik(y_true, y_pred):
    f1_kelas = f1_score(y_true, y_pred, average=None, labels=LABEL, zero_division=0)
    return {
        "akurasi": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", labels=LABEL, zero_division=0),
        "f1_0": f1_kelas[0],
        "f1_1": f1_kelas[1],
        "f1_2": f1_kelas[2],
        "f1_3": f1_kelas[3],
        "recall_1": recall_score(y_true, y_pred, labels=[1], average="macro", zero_division=0),
        "recall_3": recall_score(y_true, y_pred, labels=[3], average="macro", zero_division=0),
        "prec_3": precision_score(y_true, y_pred, labels=[3], average="macro", zero_division=0),
    }


def permutasi_kelompok(model, Xte, yte, f1_dasar, rng, fold, baris):
    """Acak tiap kelompok fitur di data uji, catat penurunan F1 macro."""
    for nama_kel, kolom in KELOMPOK.items():
        turun = []
        for _ in range(N_REPEAT_PERM):
            Xp = Xte.copy()
            idx = rng.permutation(len(Xte))
            Xp[kolom] = Xte[kolom].values[idx]
            f1_p = f1_score(yte, model.predict(Xp), average="macro", labels=LABEL, zero_division=0)
            turun.append(f1_dasar - f1_p)
        baris.append({"fold": fold, "kelompok": nama_kel, "penurunan_f1": np.mean(turun)})


def ringkas_permutasi(baris, nama_berkas):
    df_perm = pd.DataFrame(baris)
    perm = df_perm.groupby("kelompok")["penurunan_f1"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    perm.columns = ["penurunan_f1_mean", "penurunan_f1_std"]
    perm.to_csv(os.path.join(DIR_OUT, nama_berkas))
    print(perm.round(3).to_string())


def rf_dasar(**kw):
    # hyperparameter sama persis dengan model_kerawanan.py
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1,
                                  random_state=SEED, **kw)


def rf_smote(**kw):
    cat_idx = [FITUR.index(c) for c in KAT]
    return ImbPipeline([
        ("smote", SMOTENC(categorical_features=cat_idx, k_neighbors=5, random_state=SEED)),
        ("rf", RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1,
                                      random_state=SEED, **kw)),
    ])


def rf_smote_untuk_tuning():
    # n_jobs=1 di dalam RF, paralelnya dipegang RandomizedSearchCV
    cat_idx = [FITUR.index(c) for c in KAT]
    return ImbPipeline([
        ("smote", SMOTENC(categorical_features=cat_idx, k_neighbors=5, random_state=SEED)),
        ("rf", RandomForestClassifier(n_jobs=1, random_state=SEED)),
    ])


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    os.makedirs(DIR_OUT, exist_ok=True)

    df = pd.read_csv(PATH_DATA)
    df[KAT] = df[KAT].round().astype(int)
    X = df[FITUR].reset_index(drop=True)
    y = df["kelas"].to_numpy()
    g = pd.factorize(df["kabupaten"] + "|" + df["kecamatan"])[0]
    print(f"Baris: {len(df):,} | kecamatan unik (kab|kec): {len(np.unique(g)):,}")

    if X.isna().any().any():
        raise SystemExit("Ada NaN di fitur. SMOTENC gak bisa jalan, isi/drop dulu barisnya.")

    skf = StratifiedGroupKFold(n_splits=N_SPLIT, shuffle=True, random_state=SEED)
    lipatan = list(skf.split(X, y, g))

    model_dasar = {
        "RF": lambda: rf_dasar(),
        "RF + class_weight": lambda: rf_dasar(class_weight="balanced_subsample"),
        "RF + SMOTENC": lambda: rf_smote(),
    }
    NAMA_TUNED = "RF + SMOTENC (tuned)"

    semua_nama = list(model_dasar) + [NAMA_TUNED]
    oof = {n: np.zeros(len(y), dtype=int) for n in semua_nama}
    baris_metrik, baris_param = [], []
    baris_perm, baris_perm_bawaan = [], []

    t0 = time.time()
    for fold, (tr, te) in enumerate(lipatan, 1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]
        print(f"\n=== Fold {fold}/{N_SPLIT} | train {len(tr):,} | test {len(te):,} | test per kelas {np.bincount(yte, minlength=4).tolist()} ===")

        # model dasar
        for nama, buat in model_dasar.items():
            m = buat()
            m.fit(Xtr, ytr)
            pred = m.predict(Xte)
            oof[nama][te] = pred
            if nama == "RF + SMOTENC":
                model_bawaan = m  # dipakai untuk permutasi kelompok di bawah
                pred_bawaan = pred
            baris_metrik.append({"fold": fold, "model": nama, **hitung_metrik(yte, pred)})
            print(f"  {nama:<24} F1 macro = {baris_metrik[-1]['f1_macro']:.3f}")

        # tuning nested: pencarian hanya pakai data train fold ini
        cv_dalam = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
        cari = RandomizedSearchCV(
            rf_smote_untuk_tuning(),
            param_distributions=PARAM_TUNING,
            n_iter=N_ITER_TUNING,
            scoring="f1_macro",
            cv=cv_dalam,
            random_state=SEED,
            n_jobs=-1,
            refit=True,
        )
        cari.fit(Xtr, ytr, groups=g[tr])
        best = cari.best_estimator_
        pred = best.predict(Xte)
        oof[NAMA_TUNED][te] = pred
        baris_metrik.append({"fold": fold, "model": NAMA_TUNED, **hitung_metrik(yte, pred)})
        baris_param.append({"fold": fold, "skor_cv_dalam": cari.best_score_, **cari.best_params_})
        print(f"  {NAMA_TUNED:<24} F1 macro = {baris_metrik[-1]['f1_macro']:.3f} | {cari.best_params_}")

        # permutasi per kelompok fitur, data uji fold ini
        # (a) model tuned (hasil sama dengan versi sebelumnya)
        rng = np.random.RandomState(SEED + fold)
        f1_tuned = f1_score(yte, pred, average="macro", labels=LABEL, zero_division=0)
        permutasi_kelompok(best, Xte, yte, f1_tuned, rng, fold, baris_perm)
        # (b) model RF + SMOTENC bawaan (model akhir artikel)
        f1_bawaan = f1_score(yte, pred_bawaan, average="macro", labels=LABEL, zero_division=0)
        permutasi_kelompok(model_bawaan, Xte, yte, f1_bawaan, rng, fold, baris_perm_bawaan)

        print(f"  (waktu berjalan: {(time.time() - t0) / 60:.1f} menit)")

    # ---------------- simpan & tampilkan ----------------
    df_fold = pd.DataFrame(baris_metrik)
    df_fold.to_csv(os.path.join(DIR_OUT, "metrik_per_fold.csv"), index=False)

    kolom_metrik = [c for c in df_fold.columns if c not in ("fold", "model")]
    rata = df_fold.groupby("model")[kolom_metrik].mean().loc[semua_nama]
    sd = df_fold.groupby("model")[kolom_metrik].std().loc[semua_nama]
    rata.add_suffix("_mean").join(sd.add_suffix("_std")).to_csv(os.path.join(DIR_OUT, "ringkasan_model.csv"))
    pd.DataFrame(baris_param).to_csv(os.path.join(DIR_OUT, "param_terbaik_per_fold.csv"), index=False)

    print("\n=== RINGKASAN (rata-rata 5 fold) ===")
    tampil = rata[["akurasi", "balanced_acc", "f1_macro", "f1_1", "f1_3", "recall_1", "recall_3", "prec_3"]]
    tampil = tampil.join(sd[["f1_macro"]].rename(columns={"f1_macro": "f1_macro_std"}))
    print(tampil.round(3).to_string())
    print(f"\nCek reproduksi: RF = {rata.loc['RF', 'f1_macro']:.3f} (harusnya ~0.456), "
          f"RF + SMOTENC = {rata.loc['RF + SMOTENC', 'f1_macro']:.3f} (harusnya ~0.472)")

    print("\n=== Classification report gabungan out-of-fold:", NAMA_TUNED, "===")
    lap = classification_report(y, oof[NAMA_TUNED], labels=LABEL, target_names=NAMA_KELAS,
                                digits=3, zero_division=0)
    print(lap)
    with open(os.path.join(DIR_OUT, "classification_report_tuned.txt"), "w", encoding="utf-8") as f:
        f.write(lap)

    print("\n=== Permutasi per kelompok fitur, model TUNED (penurunan F1 macro, makin besar makin penting) ===")
    ringkas_permutasi(baris_perm, "permutasi_per_kelompok.csv")
    print("\n=== Permutasi per kelompok fitur, model RF + SMOTENC BAWAAN (model akhir) ===")
    ringkas_permutasi(baris_perm_bawaan, "permutasi_per_kelompok_bawaan.csv")

    # confusion matrix ternormalisasi (baris = kelas sebenarnya)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    for ax, nama in zip(axes.ravel(), semua_nama):
        ConfusionMatrixDisplay.from_predictions(
            y, oof[nama],
            labels=LABEL,
            display_labels=NAMA_KELAS,
            normalize="true",
            values_format=".2f",
            cmap="Blues",
            colorbar=False,
            ax=ax,
        )
        ax.set_title(nama, fontsize=11)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(DIR_OUT, "confusion_matrix.png"), dpi=200)
    plt.close(fig)

    print(f"\nSelesai dalam {(time.time() - t0) / 60:.1f} menit. Semua output ada di: {os.path.abspath(DIR_OUT)}")


if __name__ == "__main__":
    main()