// ============================================================
// Gee_CurahHujan_Jateng_CHIRPS_Desa.js
// Ekstraksi curah hujan HARIAN per DESA dari CHIRPS.
//
// VERSI PER BULAN (bukan per tahun) - biar tiap task lebih ringan
// dan lebih aman kalau lagi restricted mode. Satu kali run = satu
// bulan, jadi 12 export per tahun x 10 tahun (2016-2025) = 120 run
// total. Lebih banyak klik, tapi tiap task-nya jauh lebih ringan
// drpd 365 hari sekaligus.
//
// FIX PENTING: reduceRegions() pada image 1-band dengan reducer
// mean() ternyata ngasih nama properti hasil = 'mean', BUKAN nama
// band aslinya ('precipitation'). Ini penyebab asli export
// sebelumnya blank (sama kayak bug di jenis tanah & tutupan lahan).
// Fix pake reducer.setOutputs(['precipitation']).
//
// CARA PAKAI:
// 1. Ganti TAHUN dan BULAN di bawah (BULAN = 1 s/d 12)
// 2. Run script, cek dulu di Console: "Contoh 3 hasil" - pastiin
//    precipitation-nya KEISI (bukan 0 columns kayak kejadian kemarin)
// 3. Kalau udah keisi, buka tab Tasks, klik RUN di task yang muncul
// 4. Tunggu selesai, lalu ganti BULAN ke bulan berikutnya, run lagi
// 5. Ulangi sampe semua 12 bulan x 10 tahun selesai (120 kali)
//
// Rolling rainfall (3/7/14/30 hari) TETAP dihitung nanti di Python
// pas Fase 3, bukan di sini - sama kayak workflow versi kabupaten.
// ============================================================

var TAHUN = 2016;  // <-- GANTI INI
var BULAN = 1;      // <-- GANTI INI (1-12)

var desa = ee.FeatureCollection('projects/artikel-jateng/assets/desa_jawa_tengah_shp_noprj');
var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY');

var mulai = ee.Date.fromYMD(TAHUN, BULAN, 1);
var selesai = mulai.advance(1, 'month'); // otomatis handle akhir bulan & pergantian tahun (Des -> Jan)

var chirpsBulanIni = chirps.filterDate(mulai, selesai);

var hasil = chirpsBulanIni.map(function (img) {
  var tanggal = img.date().format('YYYY-MM-dd');
  var reduced = img.select('precipitation').reduceRegions({
    collection: desa,
    reducer: ee.Reducer.mean().setOutputs(['precipitation']), // <-- FIX: dulu ke-name 'mean', bukan 'precipitation'
    scale: 5566,
    tileScale: 4
  });
  return reduced.map(function (f) {
    return f.set('tanggal', tanggal).setGeometry(null);
  });
}).flatten();

var namaBulan = ee.String(mulai.format('YYYY-MM'));

Export.table.toDrive({
  collection: hasil,
  description: 'Gee_CurahHujan_Jateng_CHIRPS_Desa_' + TAHUN + '_' + (BULAN < 10 ? '0' + BULAN : BULAN),
  folder: 'multihazard_jateng_gee_curahhujan', // folder KHUSUS curah hujan, biar 120 file gak nyampur sama 3 layer lain
  fileNamePrefix: 'curahhujan_desa_jateng_' + TAHUN + '_' + (BULAN < 10 ? '0' + BULAN : BULAN),
  fileFormat: 'CSV',
  selectors: ['provinsi', 'kabupaten', 'kecamatan', 'desa', 'tanggal', 'precipitation']
});

print('Bulan yang diproses:', namaBulan);
print('Jumlah desa:', desa.size());
print('Jumlah hari (image) bulan ini:', chirpsBulanIni.size());
print('Contoh 3 hasil (cek precipitation-nya keisi apa kosong!):', hasil.limit(3));