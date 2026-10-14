// ============================================================
// Gee_Topografi_Jateng_SRTM_Desa.js
// Ekstraksi elevasi & kemiringan lereng per DESA (bukan kabupaten
// lagi) dari SRTM 30m.
// ============================================================

// Batas desa Jawa Tengah (8.559 fitur)
var desa = ee.FeatureCollection('projects/artikel-jateng/assets/desa_jawa_tengah_shp_noprj');

var srtm = ee.Image('USGS/SRTMGL1_003');
var elevasi = srtm.select('elevation').rename('elevasi');
var kemiringan = ee.Terrain.slope(elevasi).rename('kemiringan');

var topo = elevasi.addBands(kemiringan);

var hasil = topo.reduceRegions({
  collection: desa,
  reducer: ee.Reducer.mean(),
  scale: 30,
  tileScale: 4
});

// buang geometry biar CSV-nya ringan (kita cuma butuh angkanya)
var hasilBersih = hasil.map(function (f) {
  return f.setGeometry(null);
});

Export.table.toDrive({
  collection: hasilBersih,
  description: 'Gee_Topografi_Jateng_SRTM_Desa',
  folder: 'multihazard_jateng_gee',
  fileNamePrefix: 'topografi_desa_jateng',
  fileFormat: 'CSV',
  selectors: ['provinsi', 'kabupaten', 'kecamatan', 'desa', 'elevasi', 'kemiringan']
});

print('Jumlah desa:', desa.size());
print('Contoh 3 hasil:', hasilBersih.limit(3));