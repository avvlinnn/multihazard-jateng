"""
Cek distribusi kelas (banjir-only, longsor-only, multi-hazard) di beberapa pilihan unit analisis,
buat bantu putusin skema label 4 kelas (Decision 2) sebelum bikin dataset.

Jalankan dari root project (C:\\multihazard_jateng), setelah cocokkan_desa.py:
    python script_python\\cek_distribusi_kelas.py > hasil_kelas.txt
"""
from pathlib import Path

import pandas as pd

OUT = Path("data_olahan")
N_DESA_GEE = 8559

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

print("Nilai jenis_bencana:")
print(b["jenis_bencana"].value_counts(dropna=False).to_string(), "\n")

j = b["jenis_bencana"].fillna("").str.lower()
b["banjir"] = j.str.contains("banjir")
b["longsor"] = j.str.contains("longsor")

LEVEL_DESA = ["exact", "fuzzy_kec", "fuzzy_kab", "fuzzy_kab_tanpa_kec", "desa_sama_kec_beda", "manual"]
print("Level match (baris):")
print(b["level"].value_counts(dropna=False).to_string(), "\n")


def kelas(df_flag):
    """df_flag: DataFrame dengan kolom banjir & longsor (bool) per unit -> hitung kelas."""
    bo = (df_flag["banjir"] & ~df_flag["longsor"]).sum()
    lo = (df_flag["longsor"] & ~df_flag["banjir"]).sum()
    both = (df_flag["banjir"] & df_flag["longsor"]).sum()
    return bo, lo, both


def laporkan(judul, df, kunci):
    agg = df.groupby(kunci)[["banjir", "longsor"]].any().reset_index()
    bo, lo, both = kelas(agg)
    print(f"{judul:<52} banjir-only {bo:>6,} | longsor-only {lo:>6,} | multi {both:>5,} | total unit {len(agg):>6,}")


# ---------- A. Unit desa-hari (hanya baris yang match ke desa) ----------
dd = b[b["level"].isin(LEVEL_DESA)].copy()
print("=== A. DESA-HARI (baris yang match sampai desa) ===")
laporkan("desa-hari persis sama", dd, ["kab_gee", "kec_gee", "desa_gee", "tanggal"])

# multi-hazard dengan jendela waktu: banjir dan longsor di desa yang sama dalam +-k hari
bj = dd[dd["banjir"]][["kab_gee", "kec_gee", "desa_gee", "tanggal"]].drop_duplicates()
ls = dd[dd["longsor"]][["kab_gee", "kec_gee", "desa_gee", "tanggal"]].drop_duplicates()
m = bj.merge(ls, on=["kab_gee", "kec_gee", "desa_gee"], suffixes=("_b", "_l"))
m["selisih"] = (m["tanggal_b"] - m["tanggal_l"]).abs().dt.days
for k in (0, 3, 7, 14):
    n = m[m["selisih"] <= k].drop_duplicates(["kab_gee", "kec_gee", "desa_gee", "tanggal_b"])
    print(f"  hari banjir yang ada longsor di desa sama dalam +-{k:>2} hari: {len(n):,} dari {len(bj):,} hari banjir")

# ---------- B. Unit kecamatan-hari (semua baris yang punya kecamatan) ----------
kk = b[b["kec_gee"].notna() & (b["kec_gee"] != "")].copy()
print("\n=== B. KECAMATAN-HARI (desa-level + kecamatan-level) ===")
print(f"baris terpakai: {len(kk):,} dari {len(b):,}")
laporkan("kecamatan-hari persis sama", kk, ["kab_gee", "kec_gee", "tanggal"])

# ---------- C. Unit statis per desa / per kecamatan (riwayat 2016-2025) ----------
print("\n=== C. STATIS (riwayat 2016-2025 per wilayah) ===")
agg_d = dd.groupby(["kab_gee", "kec_gee", "desa_gee"])[["banjir", "longsor"]].any().reset_index()
bo, lo, both = kelas(agg_d)
tanpa = N_DESA_GEE - len(agg_d)
print(f"per desa: tanpa kejadian {tanpa:,} | banjir-only {bo:,} | longsor-only {lo:,} | multi {both:,}")
agg_k = kk.groupby(["kab_gee", "kec_gee"])[["banjir", "longsor"]].any().reset_index()
bo, lo, both = kelas(agg_k)
print(f"per kecamatan: kecamatan dengan kejadian {len(agg_k):,} | banjir-only {bo:,} | longsor-only {lo:,} | multi {both:,}")