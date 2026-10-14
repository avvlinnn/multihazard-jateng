// ============================================================
// Gee_TutupanLahan_Jateng_WorldCover_Desa.js
// Ekstraksi kelas tutupan lahan (kategorikal) per DESA dari
// ESA WorldCover v200, resolusi native 10m.
//
// FIX: sama kayak jenis tanah - reduceRegions() pada image 1-band
// dengan reducer mode() ngasih nama properti = 'mode', bukan nama
// band yang di-rename(). Fix pake reducer.setOutputs([...]).
// ============================================================

var desa = ee.FeatureCollection('projects/artikel-jateng/assets/desa_jawa_tengah_shp_noprj');

var tutupanLahan = ee.ImageCollection('ESA/WorldCover/v200')
  .first()
  .select('Map')
  .rename('tutupan_lahan');

var hasil = tutupanLahan.reduceRegions({
  collection: desa,
  reducer: ee.Reducer.mode().setOutputs(['tutupan_lahan']), // <-- FIX di sini
  scale: 10,
  tileScale: 8
});

var hasilBersih = hasil.map(function (f) {
  return f.setGeometry(null);
});

Export.table.toDrive({
  collection: hasilBersih,
  description: 'Gee_TutupanLahan_Jateng_WorldCover_Desa',
  folder: 'multihazard_jateng_gee',
  fileNamePrefix: 'tutupanlahan_desa_jateng',
  fileFormat: 'CSV',
  selectors: ['provinsi', 'kabupaten', 'kecamatan', 'desa', 'tutupan_lahan']
});

print('Jumlah desa:', desa.size());
print('Contoh 3 hasil (cek kolomnya harus "tutupan_lahan", bukan "mode"):', hasilBersih.limit(3));

// Catatan: resolusi 10m + 8.559 fitur ini yang paling berat dari
// 3 script "statis" (topografi & tanah lebih ringan). Kalau nge-timeout
// atau error "user memory limit exceeded", naikin tileScale ke 12-16.