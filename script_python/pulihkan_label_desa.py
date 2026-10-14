"""
Pulihkan label desa untuk kejadian BPBD yang tadinya cuma terlacak sampai kecamatan.

Dua sumber pemulihan, aturan sama: kalau cocoknya tidak tunggal, DIBUANG (tidak ditebak).
  A. Baris "TIDAK PASTI" (multi-desa & multi-kecamatan): pakai _daftar_desa_asli.
     Satu nama dari daftar dipasang ke kecamatan baris ini hanya kalau nama itu ada di
     kecamatan ini DAN tidak ada di kecamatan lain dari kejadian yang sama.
  B. Baris desa kosong / tidak_cocok: cari nama desa di kolom desa yang kotor dan di
     keterangan (kata sesudah desa/ds/kel/kelurahan; nama dusun/dukuh/kampung sengaja
     tidak dipakai karena sering bukan nama desa), cocokkan
     EXACT dengan daftar desa di kecamatan yang sama (kalau kecamatan tidak ketemu: harus
     unik di seluruh kabupaten). Dipakai hanya kalau hasilnya tepat 1 desa.

Prasyarat: cocokkan_desa.py sudah dijalankan.
Jalankan dari root project (C:\\multihazard_jateng):
    python script_python\\pulihkan_label_desa.py

Output (data_olahan/):
    label_tambahan_desa.csv     -> baris tambahan yang dibaca bangun_dataset_kerawanan.py
    sampel_validasi_label.csv   -> sampel acak untuk dicek manual (isi kolom 'benar': 1/0)
"""
import json
import re
from pathlib import Path

import pandas as pd

OUT = Path("data_olahan")
GEOJSON = Path("data_mentah/batas_wilayah/desa_jawa_tengah.geojson")
LEVEL_DESA = ["exact", "fuzzy_kec", "fuzzy_kab", "fuzzy_kab_tanpa_kec", "desa_sama_kec_beda", "manual"]
N_SAMPEL = 50  # per kelompok sumber


def norm(s):
    if pd.isna(s):
        return ""
    s = re.sub(r"[^A-Z0-9 ]", " ", str(s).upper())
    s = re.sub(r"\b(DESA|DS|KELURAHAN|KEL|KECAMATAN|KEC|KABUPATEN|KAB)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------- Daftar desa (GEE) ----------
with open(GEOJSON, encoding="utf-8") as f:
    gee = pd.DataFrame([x["properties"] for x in json.load(f)["features"]])
for c in ("kabupaten", "kecamatan", "desa"):
    gee[c + "_n"] = gee[c].map(norm)
kab_asli = gee.drop_duplicates("kabupaten_n").set_index("kabupaten_n")["kabupaten"].to_dict()
kec_asli = {(r.kabupaten, r.kecamatan_n): r.kecamatan for r in gee.drop_duplicates(["kabupaten", "kecamatan_n"]).itertuples()}
desa_di_kec = {k: set(g["desa_n"]) for k, g in gee.groupby(["kabupaten", "kecamatan"])}
desa_asli = {(r.kabupaten, r.kecamatan, r.desa_n): r.desa for r in gee.itertuples()}
kec_di_kab = {k: set(g["kecamatan_n"]) for k, g in gee.groupby("kabupaten")}
desa_di_kab = {}
for r in gee.itertuples():
    desa_di_kab.setdefault((r.kabupaten, r.desa_n), []).append((r.kecamatan, r.desa))

# ---------- Data BPBD + padanan ----------
b = pd.read_csv(OUT / "bpbd_banjir_longsor_split_desa.csv", parse_dates=["tanggal"])
b = b[b["tanggal"].dt.year.between(2016, 2025)].copy()
p = pd.read_csv(OUT / "padanan_desa_bpbd_gee.csv")
for d in (b, p):
    for c in ("kab_kota", "kecamatan", "desa"):
        d[c] = d[c].fillna("")
b = b.merge(p[["kab_kota", "kecamatan", "desa", "kab_gee", "kec_gee", "level"]],
            on=["kab_kota", "kecamatan", "desa"], how="left")
b = b.rename(columns={"_daftar_desa_asli": "daftar"})
j = b["jenis_bencana"].fillna("").str.lower()
b["banjir"] = j.str.contains("banjir")
b["longsor"] = j.str.contains("longsor")
b["keterangan"] = b["keterangan"].fillna("")
b["daftar"] = b["daftar"].fillna("")

tdk_pasti = b["desa"].str.upper().str.strip().eq("TIDAK PASTI")
sudah_desa = b["level"].isin(LEVEL_DESA)
poolA = b[(b["level"] == "kecamatan") & tdk_pasti]
poolB = b[~sudah_desa & ~b.index.isin(poolA.index)]
print(f"Baris BPBD 2016-2025: {len(b):,} | sudah level desa: {sudah_desa.sum():,}")
print(f"Pool A (TIDAK PASTI, kecamatan dikenal): {len(poolA):,} | Pool B (desa kosong/tidak cocok): {len(poolB):,}")

hasil = []

# ---------- A. TIDAK PASTI dari daftar desa asli ----------
kunci_ev = ["tanggal", "jenis_bencana", "daftar"]
kec_per_event = (b[b["kec_gee"].notna()].groupby(kunci_ev)["kec_gee"].agg(set)).to_dict()
statA = dict(baris_ok=0, baris_nol=0, pasangan=0, ambigu=0)
for r in poolA.itertuples():
    kab, kec = r.kab_gee, r.kec_gee
    kec_event = kec_per_event.get((r.tanggal, r.jenis_bencana, r.daftar), {kec})
    ada = 0
    for tok in r.daftar.split("|"):
        t = norm(tok)
        if not t or t not in desa_di_kec.get((kab, kec), ()):
            continue
        n_kec = sum(t in desa_di_kec.get((kab, k), ()) for k in kec_event)
        if n_kec != 1:
            statA["ambigu"] += 1
            continue
        ada += 1
        hasil.append(dict(tanggal=r.tanggal, kabupaten=kab, kecamatan=kec, desa=desa_asli[(kab, kec, t)],
                          banjir=r.banjir, longsor=r.longsor, sumber="A_daftar_desa",
                          bukti=f"'{tok.strip()}' di daftar: {r.daftar[:150]}"))
    statA["pasangan"] += ada
    statA["baris_ok" if ada else "baris_nol"] += 1

# ---------- B. Desa kosong / tidak cocok dari kolom desa & keterangan ----------
PEMICU = re.compile(r"\b(desa|ds|kelurahan|kel)\b\.?\s*:?\s*([A-Za-z][A-Za-z\-]*(?:\s+[A-Za-z][A-Za-z\-]*){0,2})", re.I)
STOP = {"RT", "RW", "KEC", "KECAMATAN", "KAB", "KABUPATEN", "DAN", "DI", "YANG", "DENGAN", "TERJADI", "AKIBAT",
        "PADA", "SEHINGGA", "DAERAH", "WILAYAH", "KOTA", "JALAN", "JL", "JLN", "KE", "DARI", "SEBAB", "DAMPAK", "UPAYA"}


def kandidat(teks):
    """Hasilkan (n-gram kata setelah kata pemicu, jenis pemicu), terpanjang dulu."""
    for m in PEMICU.finditer(teks):
        kata = []
        for w in m.group(2).split():
            if w.upper() in STOP:
                break
            kata.append(w)
        for n in range(len(kata), 0, -1):
            yield " ".join(kata[:n]), m.group(1).lower()


def cari(teks, kab, kec_n, polos=False):
    """Kembalikan set {(kec, desa, pemicu)} yang cocok exact. polos=True: seluruh teks dianggap nama desa."""
    ketemu = {}
    cands = []
    if polos:
        t = norm(teks)
        cands = [(t, "polos")] if t else []
    cands += [(norm(k), pm) for k, pm in kandidat(teks)]
    for t, pm in cands:
        if not t:
            continue
        if kec_n is not None:
            if t in desa_di_kec.get((kab, kec_n), ()):
                ketemu[(kec_n, desa_asli[(kab, kec_n, t)])] = pm
        else:
            h = desa_di_kab.get((kab, t), [])
            if len(h) == 1:
                ketemu[h[0]] = pm
    return ketemu


statB = dict(ok_desa=0, ok_ket=0, ambigu=0, nol=0, kab_gagal=0)
for r in poolB.itertuples():
    kab = kab_asli.get(norm(r.kab_kota))
    if kab is None:
        statB["kab_gagal"] += 1
        continue
    kec_n = norm(r.kecamatan)
    kec_n = kec_asli.get((kab, kec_n)) if kec_n in kec_di_kab[kab] else None
    # 1) kolom desa yang kotor; 2) keterangan
    for sumber, teks, polos in (("B_kolom_desa", r.desa if not tdk_pasti[r.Index] else "", True),
                                ("B_keterangan", r.keterangan, False)):
        if not teks.strip():
            continue
        k = cari(teks, kab, kec_n and norm(kec_n), polos)
        if len(k) == 1:
            (kc, ds), pm = next(iter(k.items()))
            i = teks.lower().find(norm(ds).split()[0].lower())
            bukti = teks[max(0, i - 40): i + 80].replace("\n", " ") if i >= 0 else teks[:120].replace("\n", " ")
            hasil.append(dict(tanggal=r.tanggal, kabupaten=kab, kecamatan=kc, desa=ds, banjir=r.banjir,
                              longsor=r.longsor, sumber=f"{sumber}({pm})", bukti=bukti))
            statB["ok_desa" if sumber == "B_kolom_desa" else "ok_ket"] += 1
            break
        if len(k) > 1:
            statB["ambigu"] += 1
            break  # ambigu: dibuang, jangan lanjut ke sumber lain
    else:
        statB["nol"] += 1

# ---------- Simpan + laporan ----------
tmb = pd.DataFrame(hasil)
tmb = tmb[tmb["banjir"] | tmb["longsor"]].drop_duplicates(["tanggal", "kabupaten", "kecamatan", "desa", "banjir", "longsor"])
tmb.to_csv(OUT / "label_tambahan_desa.csv", index=False)

print("\nPool A:", statA)
print("Pool B:", statB)
print(f"\nBaris label tambahan: {len(tmb):,} | desa unik: {tmb.groupby(['kabupaten','kecamatan','desa']).ngroups:,}")
print(tmb["sumber"].value_counts().to_string())

samp = pd.concat([g.sample(min(N_SAMPEL, len(g)), random_state=42)
                  for _, g in tmb.assign(grp=tmb["sumber"].str.split("(").str[0]).groupby("grp")])
samp["benar"] = ""
samp.to_csv(OUT / "sampel_validasi_label.csv", index=False, encoding="utf-8-sig")
print(f"\nSampel validasi: {len(samp)} baris -> {OUT / 'sampel_validasi_label.csv'} (isi kolom 'benar' 1/0)")