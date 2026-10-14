"""
Cocokkan nama desa BPBD ke nama desa versi GEE (boundary ~2020).

Prasyarat:  pip install rapidfuzz
            python script_python\\bangun_fitur_desa.py  (sudah dijalankan)
Jalankan dari root project (C:\\multihazard_jateng):
    python script_python\\cocokkan_desa.py

Output (di data_olahan/):
    padanan_desa_bpbd_gee.csv   -> semua kombinasi (kab, kec, desa) BPBD + hasil match
    desa_tidak_cocok.csv        -> yang gagal match, diurutkan dari kejadian terbanyak

Cara perbaiki manual (opsional, cukup yang kejadiannya banyak):
    isi kolom kab_gee, kec_gee, desa_gee di desa_tidak_cocok.csv (nama persis seperti di GEE),
    simpan sebagai data_olahan/padanan_manual.csv, lalu jalankan ulang script ini.
"""
import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

OUT = Path("data_olahan")
MANUAL = OUT / "padanan_manual.csv"
AMBANG_KEC = 88
AMBANG_DESA = 88
AMBANG_GABUNGAN = 90


def norm(s):
    if pd.isna(s):
        return ""
    s = re.sub(r"[^A-Z0-9 ]", " ", str(s).upper())
    s = re.sub(r"\b(DESA|DS|KELURAHAN|KEL|KECAMATAN|KEC|KABUPATEN|KAB)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def ambil_nama_desa(s):
    """'Dsn. Kediran Rt. 6 Rw. 7 Desa Tegaron' -> ' Tegaron' (ambil teks setelah kata Desa/Kelurahan terakhir)."""
    if pd.isna(s):
        return s
    bagian = re.split(r"\b(?:DESA|DS|KELURAHAN|KEL)\b\.?", str(s), flags=re.I)
    return bagian[-1] if len(bagian) > 1 and bagian[-1].strip() else s


# ---------- Data ----------
gee = pd.read_csv(OUT / "fitur_statis_desa.csv")[["kabupaten", "kecamatan", "desa"]].copy()
gee["kab_n"] = gee["kabupaten"].map(norm)
gee["kec_n"] = gee["kecamatan"].map(norm)
gee["desa_n"] = gee["desa"].map(norm)
gee_by_kab = {k: g.reset_index(drop=True) for k, g in gee.groupby("kab_n")}

b = pd.read_csv(OUT / "bpbd_banjir_longsor_split_desa.csv", parse_dates=["tanggal"])
b = b[b["tanggal"].dt.year.between(2016, 2025)]

unik = (
    b.groupby(["kab_kota", "kecamatan", "desa"], dropna=False)
    .size()
    .reset_index(name="n_baris")
)
unik["kab_n"] = unik["kab_kota"].map(norm)
unik["kec_n"] = unik["kecamatan"].map(norm)
unik["desa_n"] = unik["desa"].map(ambil_nama_desa).map(norm)
print(f"Baris BPBD 2016-2025: {len(b):,} | kombinasi unik kab/kec/desa: {len(unik):,}")

# ---------- Padanan kabupaten ----------
kab_gee_list = list(gee_by_kab.keys())
kabmap = {}
for k in unik["kab_n"].unique():
    if k in gee_by_kab:
        kabmap[k] = k
    else:
        h = process.extractOne(k, kab_gee_list, scorer=fuzz.ratio, score_cutoff=80)
        kabmap[k] = h[0] if h else None
        print(f"  Kab tidak persis: '{k}' -> {kabmap[k]}  (CEK BENAR ATAU TIDAK)")

# ---------- Manual ----------
manual = {}
if MANUAL.exists():
    m = pd.read_csv(MANUAL).dropna(subset=["kab_gee"])
    for r in m.itertuples(index=False):
        manual[(r.kab_kota, r.kecamatan, r.desa)] = dict(
            kab_gee=r.kab_gee, kec_gee=r.kec_gee, desa_gee=r.desa_gee
        )
    print(f"Padanan manual dipakai: {len(manual)} baris")


def cari_kec(g, kec_n):
    daftar = g["kec_n"].unique()
    if kec_n in daftar:
        return kec_n, 100
    if not kec_n:
        return None, 0
    h = process.extractOne(kec_n, daftar, scorer=fuzz.ratio, score_cutoff=AMBANG_KEC)
    return (h[0], h[1]) if h else (None, 0)


# ---------- Matching ----------
hasil = []
for r in unik.itertuples(index=False):
    out = dict(
        kab_kota=r.kab_kota, kecamatan=r.kecamatan, desa=r.desa, n_baris=r.n_baris,
        kab_gee=None, kec_gee=None, desa_gee=None, level="tidak_cocok", skor=0,
    )
    kunci = (r.kab_kota, r.kecamatan, r.desa)
    if kunci in manual:
        out.update(manual[kunci])
        out.update(level="manual", skor=100)
        hasil.append(out)
        continue

    kab = kabmap.get(r.kab_n)
    if kab is None:
        out["level"] = "kab_tidak_ketemu"
        hasil.append(out)
        continue
    g = gee_by_kab[kab]
    kec, skor_kec = cari_kec(g, r.kec_n)

    # baris TIDAK PASTI: cukup level kecamatan
    if r.desa_n in ("", "TIDAK PASTI"):
        if kec:
            out.update(
                kab_gee=g["kabupaten"].iloc[0],
                kec_gee=g.loc[g["kec_n"] == kec, "kecamatan"].iloc[0],
                level="kecamatan", skor=skor_kec,
            )
        hasil.append(out)
        continue

    baris, level, skor = None, None, 0
    if kec:
        gk = g[g["kec_n"] == kec].reset_index(drop=True)
        e = gk[gk["desa_n"] == r.desa_n]
        if len(e):
            baris, level, skor = e.iloc[0], "exact", 100
        else:
            h = process.extractOne(r.desa_n, gk["desa_n"].tolist(), scorer=fuzz.ratio, score_cutoff=AMBANG_DESA)
            if h:
                baris, level, skor = gk.iloc[h[2]], "fuzzy_kec", h[1]
    if baris is None:
        e = g[g["desa_n"] == r.desa_n]
        if len(e) == 1:
            baris, level, skor = e.iloc[0], "desa_sama_kec_beda", 100
    if baris is None:
        gab = (g["kec_n"] + " " + g["desa_n"]).tolist()
        h = process.extractOne(r.kec_n + " " + r.desa_n, gab, scorer=fuzz.ratio, score_cutoff=AMBANG_GABUNGAN)
        if h:
            baris, level, skor = g.iloc[h[2]], "fuzzy_kab", h[1]
    if baris is None and kec is None:
        # kecamatan kosong / tidak ketemu: cocokkan nama desa saja, hanya kalau kandidat terbaiknya tunggal
        top = process.extract(r.desa_n, g["desa_n"].tolist(), scorer=fuzz.ratio,
                              score_cutoff=AMBANG_GABUNGAN, limit=2)
        if top and (len(top) == 1 or top[0][1] > top[1][1]):
            baris, level, skor = g.iloc[top[0][2]], "fuzzy_kab_tanpa_kec", top[0][1]

    if baris is not None:
        out.update(kab_gee=baris["kabupaten"], kec_gee=baris["kecamatan"], desa_gee=baris["desa"],
                   level=level, skor=skor)
    hasil.append(out)

df = pd.DataFrame(hasil)

# ---------- Ringkasan ----------
ringkas = df.groupby("level").agg(kombinasi=("n_baris", "size"), baris=("n_baris", "sum"))
ringkas["persen_baris"] = (ringkas["baris"] / ringkas["baris"].sum() * 100).round(1)
print("\nRingkasan hasil match:")
print(ringkas.sort_values("baris", ascending=False).to_string())

df.to_csv(OUT / "padanan_desa_bpbd_gee.csv", index=False)

gagal = df[df["level"].isin(["tidak_cocok", "kab_tidak_ketemu"])].sort_values("n_baris", ascending=False)
gagal.to_csv(OUT / "desa_tidak_cocok.csv", index=False)
print(f"\nTidak cocok: {len(gagal):,} kombinasi, {gagal['n_baris'].sum():,} baris kejadian")
print("Cek dulu baris 'fuzzy_*' di padanan_desa_bpbd_gee.csv (urutkan kolom skor naik) untuk mastiin gak ada yang salah tebak.")