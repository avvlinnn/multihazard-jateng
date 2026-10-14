"""
Nambal 9 baris di bpbd_gagal_parse_tanggal.csv yang tanggalnya gagal
diparse otomatis oleh clean_bpbd.py, terus gabungin hasilnya ke
bpbd_banjir_longsor_bersih.csv.

Tanggal yang dipakai di sini dicek manual satu-satu dari kolom 'keterangan':
- "2606/2022"                      -> 2022-06-26 (jelas disebut "26 Juni 2022")
- "15 -16 September..." (2021)     -> 2021-09-16 (kejadian utama/susulan jam 01.30 dini hari tgl 16)
- "10-11 Nov 2021..."   (Wonogiri)  -> 2021-11-10 (rentang tanggal, gak ada patokan jam -> pakai awal rentang)
- "17-18/11/2021..."    (Grobogan)  -> 2021-11-17 (raw string; keterangannya ternyata nyasar,
                                       nyeritain kejadian LAIN di Cilacap -> keterangan dikosongin)
- "1812/2017 pkl-08.00"             -> 2017-12-18 (jelas disebut "18 Desember 2017 pukul 08.00")
"""

import pandas as pd

INPUT_GAGAL = "data_olahan/bpbd_gagal_parse_tanggal.csv"
INPUT_BERSIH = "data_olahan/bpbd_banjir_longsor_bersih.csv"
OUTPUT_BERSIH = "data_olahan/bpbd_banjir_longsor_bersih.csv"  # ditimpa, sudah termasuk yang ditambal

# Pemetaan tanggal_raw -> tanggal hasil cek manual
PATCH_TANGGAL = {
    "2606/2022": "2022-06-26",
    "15 -16 September Pukul 01.00WIB": "2021-09-16",
    "10-11 Nov 2021 Pukul 17.00WIB": "2021-11-10",
    "17-18/11/2021 Pukul 22.00WIB": "2021-11-17",
    "1812/2017 pkl-08.00": "2017-12-18",
}

# Baris ini keterangannya nyasar dari kejadian lain (Cilacap, bukan Grobogan) -> dikosongin
KETERANGAN_TIDAK_VALID = ["17-18/11/2021 Pukul 22.00WIB"]


def main():
    gagal = pd.read_csv(INPUT_GAGAL)
    bersih = pd.read_csv(INPUT_BERSIH)

    tidak_ketemu = set(gagal["tanggal_raw"].unique()) - set(PATCH_TANGGAL.keys())
    if tidak_ketemu:
        print(f"⚠️  Ada tanggal_raw yang belum ada di PATCH_TANGGAL: {tidak_ketemu}")
        print("   Baris ini gak ikut ditambal, cek manual dulu.")

    gagal["tanggal"] = gagal["tanggal_raw"].map(PATCH_TANGGAL)
    gagal.loc[gagal["tanggal_raw"].isin(KETERANGAN_TIDAK_VALID), "keterangan"] = \
        "[keterangan asli tidak valid - nyasar dari kejadian lain, dikosongin]"

    ditambal = gagal[gagal["tanggal"].notna()].copy()
    kolom_bersih = ["tanggal", "tahun_sheet", "format_asal", "kab_kota",
                     "kecamatan", "desa", "jenis_bencana", "keterangan"]
    ditambal = ditambal[kolom_bersih]

    hasil = pd.concat([bersih, ditambal], ignore_index=True)
    hasil.to_csv(OUTPUT_BERSIH, index=False)

    print(f"Baris berhasil ditambal & digabung: {len(ditambal)}")
    print(f"Total baris di {OUTPUT_BERSIH} sekarang: {len(hasil)}")


if __name__ == "__main__":
    main()