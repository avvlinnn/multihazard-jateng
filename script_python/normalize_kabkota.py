"""
Normalisasi kolom 'kab_kota' di data BPBD:
- Buang spasi nyempil (misal "Brebes ", "Kudus ")
- Samain casing kalau ada penulisan beda (misal "Kota semarang" vs "Kota Semarang")
- Ganti "Kabupaten Tegal" -> "Tegal" (biar konsisten sama kabupaten lain yang
  emang gak pake prefix "Kabupaten")

4 pasang ini SENGAJA dipertahanin sebagai entitas beda (bukan bug), soalnya
emang kabupaten & kota yang berbeda:
- Tegal (kab) vs Kota Tegal
- Magelang (kab) vs Kota Magelang
- Pekalongan (kab) vs Kota Pekalongan
- Semarang (kab) vs Kota Semarang

Jalanin ini SETELAH patch_tanggal_manual.py, SEBELUM split_multi_desa.py,
biar hasil normalisasinya ikut kebawa ke file split.

Cara pakai, dari root project:
    python script_python/normalize_kabkota.py
"""

import pandas as pd

INPUT_FILE = "data_olahan/bpbd_banjir_longsor_bersih.csv"
OUTPUT_FILE = "data_olahan/bpbd_banjir_longsor_bersih.csv"  # ditimpa

# Rename semantik tambahan, di luar urusan spasi/casing
RENAME_MANUAL = {
    "Kabupaten Tegal": "Tegal",
}


def main():
    df = pd.read_csv(INPUT_FILE)
    sebelum = df["kab_kota"].nunique()

    df["kab_kota"] = df["kab_kota"].astype(str).str.strip()

    # canonical casing: buat tiap nama (case-insensitive), pilih penulisan
    # yang paling sering muncul sebagai "yang bener"
    counts = df["kab_kota"].value_counts()
    canon = {}
    for val in counts.index:
        key = val.lower()
        if key not in canon or counts[val] > counts[canon[key]]:
            canon[key] = val
    mapping = {val: canon[val.lower()] for val in df["kab_kota"].unique()}
    mapping.update(RENAME_MANUAL)

    df["kab_kota"] = df["kab_kota"].map(mapping)
    sesudah = df["kab_kota"].nunique()

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Jumlah nama kab/kota unik: {sebelum} -> {sesudah}")
    print(f"Disimpan ke: {OUTPUT_FILE}")
    print("\nDaftar akhir:")
    for nama in sorted(df["kab_kota"].unique()):
        print(f"  - {nama}")


if __name__ == "__main__":
    main()