"""
Bangun dataset kerawanan per desa (unit statis, riwayat 2016-2025).

Label 4 kelas per desa:
    0 = tidak tercatat kejadian | 1 = banjir-only | 2 = longsor-only | 3 = multi-hazard (keduanya)

Prasyarat: bangun_fitur_desa.py dan cocokkan_desa.py sudah dijalankan.
Jalankan dari root project (C:\\multihazard_jateng):
    python script_python\\bangun_dataset_kerawanan.py

Output: data_olahan/dataset_kerawanan_desa.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data_olahan")
LEVEL_DESA = ["exact", "fuzzy_kec", "fuzzy_kab", "fuzzy_kab_tanpa_kec", "desa_sama_kec_beda", "manual"]
KUNCI = ["kabupaten", "kecamatan", "desa"]
TGL0 = pd.Timestamp("2015-12-01")  # hari ke-0 matriks curah hujan
MULAI = pd.Timestamp("2016-01-01")  # Des 2015 cuma buffer, tidak ikut statistik
N_TAHUN = 10

# ---------- 1. Fitur statis + statistik curah hujan ----------
df = pd.read_csv(OUT / "fitur_statis_desa.csv")
M = np.load(OUT / "curah_hujan_harian.npy")
assert M.shape[0] == len(df), "Jumlah baris matriks beda dengan fitur_statis_desa.csv"

R = M[:, (MULAI - TGL0).days:]
kosong = np.isnan(R).all(axis=1)
R0 = np.nan_to_num(R, nan=0.0).astype(np.float32)


def maks_akumulasi(x, w):
    c = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(x, axis=1, dtype=np.float64)], axis=1)
    return (c[:, w:] - c[:, :-w]).max(axis=1)


hujan = {
    "ch_tahunan_mean": R0.sum(axis=1) / N_TAHUN,
    "hari_hujan_thn": (R0 >= 1).sum(axis=1) / N_TAHUN,
    "hari_lebat_thn": (R0 >= 50).sum(axis=1) / N_TAHUN,
    "hari_sangat_lebat_thn": (R0 >= 100).sum(axis=1) / N_TAHUN,
    "ch_harian_max": R0.max(axis=1),
    "ch_3hari_max": maks_akumulasi(R0, 3),
    "ch_7hari_max": maks_akumulasi(R0, 7),
}
for nama, nilai in hujan.items():
    nilai = nilai.astype(np.float64)
    nilai[kosong] = np.nan
    df[nama] = nilai
FITUR_HUJAN = list(hujan)

# ---------- 2. Label dari BPBD ----------
b = pd.read_csv(OUT / "bpbd_banjir_longsor_split_desa.csv", parse_dates=["tanggal"])
b = b[b["tanggal"].dt.year.between(2016, 2025)].copy()
p = pd.read_csv(OUT / "padanan_desa_bpbd_gee.csv")
for d in (b, p):
    for c in ("kab_kota", "kecamatan", "desa"):
        d[c] = d[c].fillna("")
b = b.merge(
    p[["kab_kota", "kecamatan", "desa", "kab_gee", "kec_gee", "desa_gee", "level"]],
    on=["kab_kota", "kecamatan", "desa"], how="left",
)
j = b["jenis_bencana"].fillna("").str.lower()
b["banjir"] = j.str.contains("banjir")
b["longsor"] = j.str.contains("longsor")

dd = b[b["level"].isin(LEVEL_DESA)].drop(columns=["kab_kota", "kecamatan", "desa"])
dd = dd.rename(columns={"kab_gee": "kabupaten", "kec_gee": "kecamatan", "desa_gee": "desa"})

# Label tambahan hasil pulihkan_label_desa.py (TIDAK PASTI & desa kosong yang berhasil dipulihkan)
tmb = pd.read_csv(OUT / "label_tambahan_desa.csv", parse_dates=["tanggal"])
dd = pd.concat([dd, tmb[KUNCI + ["tanggal", "banjir", "longsor"]]], ignore_index=True)

def hitung_hari(flag):
    x = dd[dd[flag]].drop_duplicates(KUNCI + ["tanggal"])
    return x.groupby(KUNCI).size()


lab = pd.concat([hitung_hari("banjir").rename("n_hari_banjir"),
                 hitung_hari("longsor").rename("n_hari_longsor")], axis=1).fillna(0).astype(int).reset_index()
df = df.merge(lab, on=KUNCI, how="left")
df[["n_hari_banjir", "n_hari_longsor"]] = df[["n_hari_banjir", "n_hari_longsor"]].fillna(0).astype(int)

df["kelas"] = np.select(
    [(df["n_hari_banjir"] > 0) & (df["n_hari_longsor"] > 0), df["n_hari_banjir"] > 0, df["n_hari_longsor"] > 0],
    [3, 1, 2], default=0,
)

# Kejadian yang cuma terlacak sampai kecamatan: dipakai buat uji sensitivitas label, BUKAN fitur model
kk = b[b["level"] == "kecamatan"].groupby(["kab_gee", "kec_gee"]).size().rename("n_kejadian_kec_tanpa_desa").reset_index()
kk = kk.rename(columns={"kab_gee": "kabupaten", "kec_gee": "kecamatan"})
df = df.merge(kk, on=["kabupaten", "kecamatan"], how="left")
df["n_kejadian_kec_tanpa_desa"] = df["n_kejadian_kec_tanpa_desa"].fillna(0).astype(int)

# ---------- 3. Buang desa tanpa fitur lengkap ----------
FITUR = ["elevasi", "kemiringan", "tekstur_tanah", "tutupan_lahan"] + FITUR_HUJAN
print("Distribusi kelas SEBELUM buang desa tanpa data:")
print(df["kelas"].value_counts().sort_index().to_string())
tanpa_fitur = df[FITUR].isna().any(axis=1)
print(f"\nDesa dibuang karena fitur kosong: {tanpa_fitur.sum()} (kelas: {df.loc[tanpa_fitur, 'kelas'].value_counts().to_dict()})")
df = df[~tanpa_fitur].reset_index(drop=True)

print(f"\nDataset akhir: {len(df):,} desa")
print("Distribusi kelas:")
dist = df["kelas"].value_counts().sort_index()
print(pd.DataFrame({"jumlah": dist, "persen": (dist / len(df) * 100).round(1)}).to_string())
print("\nRata-rata fitur per kelas:")
print(df.groupby("kelas")[FITUR].mean().round(2).T.to_string())

df.to_csv(OUT / "dataset_kerawanan_desa.csv", index=False)
print(f"\nTersimpan: {OUT / 'dataset_kerawanan_desa.csv'}")