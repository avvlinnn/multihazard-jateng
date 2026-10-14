# Multi-hazard Jawa Tengah

Kode untuk pencocokan nama desa, penyusunan label, ekstraksi fitur Google Earth Engine, pemodelan kerawanan, validasi, dan pembuatan grafik/peta di Jawa Tengah.

## Status data dan publikasi

Repositori ini sengaja tidak menyertakan data mentah BPBD, data berlabel turunannya, kredensial, `.env`, batas desa GeoJSON, lingkungan virtual, atau output besar. Label kelas berasal dari data BPBD yang diperoleh melalui permintaan resmi, sehingga CSV turunan per desa hanya boleh dipublikasikan setelah ada persetujuan BPBD.

Jika izin berbagi sudah jelas, jalankan `script_python/ekspor_dataset_per_desa.py`. Skrip itu membuat satu CSV untuk setiap desa di `data/turunan/`, dengan kunci desa (`id_desa`), `kelas` bernilai 0 sampai 3, dan 11 fitur model. Dataset turunan tersebut dapat dilisensikan dengan CC BY 4.0 jika pemilik datanya menyetujui.

Batas wilayah desa tidak disalin ke repositori ini. Gunakan sumber resmi yang dirujuk oleh proyek, lalu simpan file lokal di `data_mentah/batas_wilayah/`.

## Instalasi

Gunakan Python yang kompatibel dengan versi pada `requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Urutan menjalankan

Jalankan semua perintah dari root proyek. Data input dan output lokal yang dipakai skrip berada di folder yang sengaja diabaikan Git.

1. Jalankan skrip GEE di `script_gee/` untuk mengekspor curah hujan, topografi, jenis tanah, dan tutupan lahan per desa.
2. Jalankan `filter_desa_jateng.py` untuk memilih desa Jawa Tengah dari batas wilayah resmi.
3. Jalankan `bangun_fitur_desa.py` untuk menggabungkan fitur statis dan hujan.
4. Jalankan `clean_bpbd.py`, `normalize_kabkota.py`, `split_multi_desa.py`, dan `cocokkan_desa.py` untuk membersihkan serta mencocokkan nama desa BPBD. Jalankan `patch_tanggal_manual.py` bila diperlukan.
5. Jalankan `pulihkan_label_desa.py` dan `bangun_dataset_kerawanan.py` untuk membentuk dataset berlabel.
6. Periksa kolom dan distribusi dengan `cek_kolom.py` serta `cek_distribusi_kelas.py`.
7. Jalankan `model_kerawanan.py` untuk validasi silang, Random Forest, XGBoost, SMOTENC, confusion matrix, dan feature importance.
8. Jalankan `model_tambahan.py`, `hitung_validasi_label.py`, `analisis_spasial.py`, `peta_kerawanan.py`, dan `gambar_inggris.py` sesuai kebutuhan analisis.
9. Setelah izin distribusi data diperoleh, jalankan `ekspor_dataset_per_desa.py` dan tinjau hasilnya sebelum publikasi.

Contoh:

```powershell
python script_python\bangun_fitur_desa.py
python script_python\bangun_dataset_kerawanan.py
python script_python\model_kerawanan.py
```

## Reproduksibilitas

Eksperimen model menggunakan seed 42, 5-fold `StratifiedGroupKFold` dengan grup kabupaten-kecamatan, dan 11 fitur yang didefinisikan di `model_kerawanan.py`. Artefak hasil dapat dibuat ulang dari input yang sah dan skrip dalam repositori.

## Lisensi

Kode dilisensikan dengan MIT. Data pihak ketiga dan data BPBD tetap tunduk pada izin serta ketentuan pemiliknya; jangan menganggap lisensi kode sebagai izin untuk membagikan data.