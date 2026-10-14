// ============================================================
// Gee_JenisTanah_Jateng_OpenLandMap_Desa.js
// Ekstraksi kelas tekstur tanah (kategorikal) per DESA dari
// OpenLandMap, resolusi native 250m.
//
// FIX: reduceRegions() pada image 1-band dengan reducer mode()
// ternyata ngasih nama properti hasil = nama REDUCER ('mode'),
// BUKAN nama band yang udah di-rename(). Makanya export
// sebelumnya blank - selectors minta 'tekstur_tanah' tapi
// propertinya ketulis sebagai 'mode'. Fix pake
// reducer.setOutputs([...]) buat maksa nama properti yang benar.
// ============================================================

var desa = ee.FeatureCollection('projects/artikel-jateng/assets/desa_jawa_tengah_shp_noprj');

var tanah = ee.Image('OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02')
  .select('b0')
  .rename('tekstur_tanah');

var hasil = tanah.reduceRegions({
  collection: desa,
  reducer: ee.Reducer.mode().setOutputs(['tekstur_tanah']), // <-- FIX di sini
  scale: 250,
  tileScale: 4
});

var hasilBersih = hasil.map(function (f) {
  return f.setGeometry(null);
});

Export.table.toDrive({
  collection: hasilBersih,
  description: 'Gee_JenisTanah_Jateng_OpenLandMap_Desa',
  folder: 'multihazard_jateng_gee',
  fileNamePrefix: 'jenistanah_desa_jateng',
  fileFormat: 'CSV',
  selectors: ['provinsi', 'kabupaten', 'kecamatan', 'desa', 'tekstur_tanah']
});

print('Jumlah desa:', desa.size());
print('Contoh 3 hasil (cek kolomnya harus "tekstur_tanah", bukan "mode"):', hasilBersih.limit(3));