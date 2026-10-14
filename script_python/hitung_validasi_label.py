r"""
hitung_validasi_label.py

Merekap hasil validasi manual label di data_olahan\sampel_validasi_label.csv.

Cara mengisi kolom 'benar' (di Excel, simpan tetap sebagai CSV):
    1     = teks kejadian mendukung desa yang dipasangkan
    0     = teks kejadian bertentangan dengan desa yang dipasangkan
    kosong = tidak bisa diverifikasi dari teks (jangan dipaksa 1 atau 0)

Jalankan dari root project (C:\multihazard_jateng):
    python script_python\hitung_validasi_label.py

Output: tabel rekap di layar + hasil\rekap_validasi_label.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

PATH = Path("data_olahan") / "sampel_validasi_label.csv"
OUT = Path("hasil") / "rekap_validasi_label.csv"


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
    terverifikasi = ok + salah
    lo, hi = wilson(ok, terverifikasi)
    return {
        "n_sampel": n,
        "didukung_teks (1)": ok,
        "bertentangan (0)": salah,
        "tidak_bisa_diverifikasi": kosong,
        "pct_tidak_bisa_diverifikasi": round(100 * kosong / n, 1) if n else np.nan,
        "pct_benar_dari_terverifikasi": round(100 * ok / terverifikasi, 1) if terverifikasi else np.nan,
        "ci95_bawah": round(100 * lo, 1) if terverifikasi else np.nan,
        "ci95_atas": round(100 * hi, 1) if terverifikasi else np.nan,
    }


def main():
    d = pd.read_csv(PATH)
    if "benar" not in d.columns:
        raise SystemExit("Kolom 'benar' tidak ada di CSV.")
    d["benar"] = pd.to_numeric(d["benar"], errors="coerce")
    if not d["benar"].dropna().isin([0, 1]).all():
        raise SystemExit("Kolom 'benar' hanya boleh berisi 1, 0, atau kosong.")
    if d["benar"].notna().sum() == 0:
        raise SystemExit("Kolom 'benar' masih kosong semua. Isi dulu hasil verifikasi.")

    baris = {"SEMUA": rekap(d)}
    if "grp" in d.columns:  # A_daftar_desa, B_keterangan, B_kolom_desa (masing-masing 50 baris)
        for k, sub in d.groupby("grp"):
            baris[k] = rekap(sub)

    hasil = pd.DataFrame(baris).T
    OUT.parent.mkdir(exist_ok=True)
    hasil.to_csv(OUT)
    pd.options.display.width = 250
    print(hasil.to_string())
    print(f"\nTersimpan: {OUT.resolve()}")
    print("\nCatatan: persentase benar hanya dihitung dari baris yang bisa diverifikasi.")
    print("Laporkan juga porsi 'tidak bisa diverifikasi' supaya klaimnya tidak berlebihan.")


if __name__ == "__main__":
    main()