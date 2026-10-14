r"""
cek_padanan_fuzzy.py

Validasi manual padanan nama desa BPBD -> GEE hasil cocokkan_desa.py
(level fuzzy_kec, fuzzy_kab, fuzzy_kab_tanpa_kec, desa_sama_kec_beda).

Langkah 1 (bikin sampel), jalankan dari root project (C:\multihazard_jateng):
    python script_python\cek_padanan_fuzzy.py
  -> data_olahan\sampel_validasi_padanan.csv
  Buka di Excel, isi kolom 'benar' lalu simpan tetap sebagai CSV:
    1     = desa BPBD dan desa GEE adalah desa yang sama (ejaan beda tidak masalah)
    0     = desa berbeda (salah pasang)
    kosong = tidak bisa dipastikan (jangan dipaksa)

Langkah 2 (rekap), jalankan perintah yang sama setelah kolom 'benar' terisi
  -> hasil\rekap_validasi_padanan.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data_olahan")
PADANAN = OUT / "padanan_desa_bpbd_gee.csv"
SAMPEL = OUT / "sampel_validasi_padanan.csv"
REKAP = Path("hasil") / "rekap_validasi_padanan.csv"
LEVEL_UJI = ["fuzzy_kec", "fuzzy_kab", "fuzzy_kab_tanpa_kec", "desa_sama_kec_beda"]
N_PER_LEVEL = 30
SEED = 42


def wilson(k, n, z=1.96):
    """Interval kepercayaan 95% Wilson untuk proporsi k/n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    pusat = (p + z**2 / (2 * n)) / d
    lebar = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (pusat - lebar, pusat + lebar)


def rekap(d):
    n = len(d)
    ok = int((d["benar"] == 1).sum())
    salah = int((d["benar"] == 0).sum())
    kosong = int(d["benar"].isna().sum())
    ter = ok + salah
    lo, hi = wilson(ok, ter)
    return {
        "n_sampel": n,
        "benar (1)": ok,
        "salah (0)": salah,
        "tidak_bisa_diverifikasi": kosong,
        "pct_benar_dari_terverifikasi": round(100 * ok / ter, 1) if ter else np.nan,
        "ci95_bawah": round(100 * lo, 1) if ter else np.nan,
        "ci95_atas": round(100 * hi, 1) if ter else np.nan,
    }


def baca_sampel():
    """Baca CSV dengan pemisah otomatis (Excel versi Indonesia sering menyimpan pakai titik koma)."""
    d = pd.read_csv(SAMPEL, encoding="utf-8-sig", sep=None, engine="python")
    d.columns = [str(c).strip() for c in d.columns]
    if "benar" not in d.columns:
        raise SystemExit(f"Kolom 'benar' tidak ketemu. Kolom yang terbaca: {list(d.columns)}")
    return d


def buat_sampel():
    p = pd.read_csv(PADANAN)
    total = p["n_baris"].sum()
    print("Sebaran baris BPBD per level padanan:")
    r = p.groupby("level")["n_baris"].agg(kombinasi="size", baris="sum")
    r["persen_baris"] = (r["baris"] / total * 100).round(1)
    print(r.sort_values("baris", ascending=False).to_string())

    potongan = []
    for lv in LEVEL_UJI:
        sub = p[p["level"] == lv]
        if len(sub):
            potongan.append(sub.sample(min(N_PER_LEVEL, len(sub)), random_state=SEED))
    s = pd.concat(potongan, ignore_index=True)
    kolom = ["level", "skor", "kab_kota", "kecamatan", "desa",
             "kab_gee", "kec_gee", "desa_gee", "n_baris"]
    s = s[kolom]
    s["benar"] = ""
    s.to_csv(SAMPEL, index=False, encoding="utf-8-sig")
    print(f"\nSampel {len(s)} kombinasi tersimpan: {SAMPEL}")
    print("Isi kolom 'benar' (1 / 0 / kosong), simpan sebagai CSV, lalu jalankan script ini lagi.")


def hitung_rekap():
    d = baca_sampel()
    d["benar"] = pd.to_numeric(d["benar"], errors="coerce")
    if not d["benar"].dropna().isin([0, 1]).all():
        raise SystemExit("Kolom 'benar' hanya boleh berisi 1, 0, atau kosong.")
    baris = {"SEMUA": rekap(d)}
    for lv, sub in d.groupby("level"):
        baris[lv] = rekap(sub)
    hasil = pd.DataFrame(baris).T
    REKAP.parent.mkdir(exist_ok=True)
    hasil.to_csv(REKAP)
    pd.options.display.width = 250
    print(hasil.to_string())
    print(f"\nTersimpan: {REKAP.resolve()}")
    print("Catatan: persentase hanya dari baris yang bisa diverifikasi. Laporkan juga porsi yang tidak bisa diverifikasi.")
    salah = d[d["benar"] == 0]
    if len(salah):
        print("\nPadanan yang salah (pertimbangkan masukkan ke padanan_manual.csv):")
        print(salah[["level", "skor", "kecamatan", "desa", "kec_gee", "desa_gee"]].to_string(index=False))


def main():
    if not SAMPEL.exists():
        buat_sampel()
        return
    d = baca_sampel()
    if pd.to_numeric(d["benar"], errors="coerce").notna().sum() == 0:
        print(f"{SAMPEL} sudah ada tapi kolom 'benar' masih kosong semua.")
        print("Isi dulu, atau hapus file itu kalau mau bikin sampel baru.")
        return
    hitung_rekap()


if __name__ == "__main__":
    main()