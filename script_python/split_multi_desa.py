"""
Split baris BPBD yang kolom kecamatan/desa-nya berisi lebih dari satu nilai
(dipisah koma, titik koma, atau baris baru) jadi satu baris per desa.

Contoh kasus: kolom Ds. = "Purwodadi, Jono" dan kolom KECAMATAN = "Purwodadi, Tawangharjo"
  -> dipecah jadi 2 baris:
     1. Ds. = "Purwodadi", KECAMATAN = "Purwodadi"
     2. Ds. = "Jono",      KECAMATAN = "Tawangharjo"
  (dipasangkan berdasarkan urutan index, karena polanya di data BPBD memang
   desa ke-i selalu berpasangan dengan kecamatan ke-i)
"""

import pandas as pd
import re

# ==== KONFIGURASI — cek dulu, sesuaikan kalau nama kolom di CSV kamu beda ====
INPUT_FILE = "data_olahan/bpbd_banjir_longsor_bersih.csv"
OUTPUT_FILE = "data_olahan/bpbd_banjir_longsor_split_desa.csv"
KOL_KECAMATAN = "kecamatan"
KOL_DESA = "desa"
# ==============================================================================

SEPARATOR_PATTERN = r"[,\n;]+"


def split_nilai(value):
    """Pecah satu sel jadi list nilai. Sel kosong/NaN jadi [None]."""
    if pd.isna(value):
        return [None]
    parts = [p.strip() for p in re.split(SEPARATOR_PATTERN, str(value)) if p.strip()]
    return parts if parts else [None]


def is_multi(value):
    return isinstance(value, str) and re.search(SEPARATOR_PATTERN, value) is not None


def expand_row(row):
    kec_list = split_nilai(row[KOL_KECAMATAN])
    desa_list = split_nilai(row[KOL_DESA])

    # cuma dianggap ambigu kalau DUA-duanya (kecamatan & desa) sama-sama
    # punya lebih dari 1 nilai DAN jumlahnya beda. Kalau cuma salah satu
    # yang >1 (misal 1 kecamatan buat banyak desa), itu aman -> broadcast.
    mismatch = len(kec_list) > 1 and len(desa_list) > 1 and len(kec_list) != len(desa_list)

    hasil = []
    if mismatch:
        # Gak bisa mastiin desa mana masuk kecamatan mana -> jangan nebak.
        # Turunin ke level kecamatan aja: satu baris per kecamatan, desa
        # ditandai "TIDAK PASTI" (daftar desa aslinya disimpen di kolom
        # terpisah buat referensi, siapa tau nanti kepake).
        for kec in kec_list:
            baris_baru = row.copy()
            baris_baru[KOL_KECAMATAN] = kec
            baris_baru[KOL_DESA] = "TIDAK PASTI"
            baris_baru["_daftar_desa_asli"] = " | ".join(d for d in desa_list if d)
            baris_baru["_flag_hasil_split"] = True
            baris_baru["_flag_perlu_cek_manual"] = True
            hasil.append(baris_baru)
        return hasil

    n = max(len(kec_list), len(desa_list))
    for i in range(n):
        baris_baru = row.copy()
        baris_baru[KOL_KECAMATAN] = kec_list[i] if i < len(kec_list) else kec_list[-1]
        baris_baru[KOL_DESA] = desa_list[i] if i < len(desa_list) else desa_list[-1]
        baris_baru["_daftar_desa_asli"] = ""
        baris_baru["_flag_hasil_split"] = True
        baris_baru["_flag_perlu_cek_manual"] = False
        hasil.append(baris_baru)
    return hasil


def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"Kolom yang terbaca: {list(df.columns)}")
    print(f"Total baris awal: {len(df)}")

    if KOL_KECAMATAN not in df.columns or KOL_DESA not in df.columns:
        print(f"\n⚠️  Kolom '{KOL_KECAMATAN}' atau '{KOL_DESA}' gak ketemu di CSV.")
        print("   Cek nama kolom di atas, terus edit KOL_KECAMATAN / KOL_DESA di bagian KONFIGURASI.")
        return

    semua_baris = []
    n_split = 0
    n_mismatch = 0

    for _, row in df.iterrows():
        kec_multi = is_multi(row[KOL_KECAMATAN])
        desa_multi = is_multi(row[KOL_DESA])

        if kec_multi or desa_multi:
            expanded = expand_row(row)
            n_split += 1
            if expanded[0]["_flag_perlu_cek_manual"]:
                n_mismatch += 1
            semua_baris.extend(expanded)
        else:
            baris_baru = row.copy()
            baris_baru["_daftar_desa_asli"] = ""
            baris_baru["_flag_hasil_split"] = False
            baris_baru["_flag_perlu_cek_manual"] = False
            semua_baris.append(baris_baru)

    hasil_df = pd.DataFrame(semua_baris)
    hasil_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nBaris asli yang di-split (multi desa/kecamatan): {n_split}")
    print(f"  - dari situ, yang jumlah desa & kecamatannya gak sama (diturunin ke level kecamatan, desa='TIDAK PASTI'): {n_mismatch}")
    print(f"Total baris setelah split: {len(hasil_df)}")
    print(f"Disimpan ke: {OUTPUT_FILE}")
    print("\nBaris dengan desa='TIDAK PASTI' itu tetap kepake buat data di level kecamatan,")
    print("cuma gak dipaksa nebak desa spesifiknya. Daftar desa aslinya masih ada di kolom")
    print("'_daftar_desa_asli' kalau suatu saat mau dicek ulang manual.")


if __name__ == "__main__":
    main()