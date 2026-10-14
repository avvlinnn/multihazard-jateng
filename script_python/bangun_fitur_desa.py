"""
Gabung fitur statis (topografi, tanah, tutupan lahan) + bikin matriks curah hujan harian per desa.

Jalankan dari root project (C:\\multihazard_jateng):
    python script_python\\bangun_fitur_desa.py

Output (di data_olahan/):
    fitur_statis_desa.csv       -> 8.559 desa, urutan barisnya = urutan baris matriks
    curah_hujan_harian.npy      -> matriks float32 [desa x hari], hari ke-0 = 2015-12-01
    desa_tanpa_hujan.csv        -> desa yang curah hujannya kosong total
"""
from pathlib import Path

import numpy as np
import pandas as pd

GEE = Path("data_mentah/gee")
OUT = Path("data_olahan")
OUT.mkdir(exist_ok=True)

KUNCI = ["kabupaten", "kecamatan", "desa"]
TGL0 = pd.Timestamp("2015-12-01")
TGL1 = pd.Timestamp("2025-12-31")
N_HARI = (TGL1 - TGL0).days + 1  # 3684


def baca_satu(nama):
    f = list(GEE.rglob(nama))
    assert len(f) == 1, f"Harusnya 1 file {nama}, ketemu {len(f)}"
    return pd.read_csv(f[0])


# ---------- 1. Fitur statis ----------
topo = baca_satu("topografi_desa_jateng.csv")
tanah = baca_satu("jenistanah_desa_jateng.csv")
lahan = baca_satu("tutupanlahan_desa_jateng.csv")

kunci_gabung = ["provinsi"] + KUNCI
statis = (
    topo.merge(tanah, on=kunci_gabung, how="outer", validate="one_to_one")
    .merge(lahan, on=kunci_gabung, how="outer", validate="one_to_one")
    .reset_index(drop=True)
)
print(f"Fitur statis: {len(statis):,} desa")
print("Null per kolom:")
print(statis.isna().sum()[lambda s: s > 0].to_string() or "  (tidak ada)")

statis.to_csv(OUT / "fitur_statis_desa.csv", index=False)

# ---------- 2. Matriks curah hujan ----------
mi = pd.MultiIndex.from_frame(statis[KUNCI])
M = np.full((len(statis), N_HARI), np.nan, dtype=np.float32)

files = sorted(GEE.rglob("curahhujan_desa_jateng_*.csv"))
print(f"\nBaca {len(files)} file curah hujan...")

for i, f in enumerate(files, 1):
    df = pd.read_csv(f, usecols=KUNCI + ["tanggal", "precipitation"])
    pos = mi.get_indexer(pd.MultiIndex.from_frame(df[KUNCI]))
    t = (pd.to_datetime(df["tanggal"]) - TGL0).dt.days.to_numpy()
    ok = (pos >= 0) & (t >= 0) & (t < N_HARI)
    if not ok.all():
        print(f"  PERINGATAN {f.name}: {(~ok).sum()} baris tidak masuk (desa/tanggal tidak dikenali)")
    M[pos[ok], t[ok]] = df["precipitation"].to_numpy(dtype=np.float32)[ok]
    if i % 20 == 0 or i == len(files):
        print(f"  {i}/{len(files)} selesai")

n_neg = int((M < 0).sum())
if n_neg:
    print(f"PERINGATAN: {n_neg} nilai negatif (kemungkinan sentinel), diubah jadi NaN")
    M[M < 0] = np.nan

# ---------- 3. Cek kelengkapan ----------
kosong_total = np.isnan(M).all(axis=1)
kosong_sebagian = np.isnan(M).any(axis=1) & ~kosong_total
print(f"\nDesa kosong total: {kosong_total.sum()}  (harapannya 22)")
print(f"Desa kosong sebagian: {kosong_sebagian.sum()}  (harapannya 0)")
print(f"Total sel NaN: {np.isnan(M).sum():,}  (harapannya {22 * N_HARI:,})")
print(f"Rentang nilai: min {np.nanmin(M):.2f}, max {np.nanmax(M):.2f} mm/hari")

statis.loc[kosong_total, KUNCI].to_csv(OUT / "desa_tanpa_hujan.csv", index=False)
np.save(OUT / "curah_hujan_harian.npy", M)
print(f"\nSelesai. Matriks {M.shape}, {M.nbytes / 1e6:.0f} MB.")