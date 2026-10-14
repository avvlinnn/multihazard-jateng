"""Export one reproducible feature-and-label CSV for each village.

Run only after permission has been obtained to redistribute BPBD-derived labels.
"""
from pathlib import Path
import re

import pandas as pd


INPUT = Path("data_olahan/dataset_kerawanan_desa.csv")
OUTPUT = Path("data/turunan")
KEY_COLUMNS = ["kabupaten", "kecamatan", "desa"]
FEATURES = [
    "elevasi", "kemiringan", "tekstur_tanah", "tutupan_lahan",
    "ch_tahunan_mean", "hari_hujan_thn", "hari_lebat_thn",
    "hari_sangat_lebat_thn", "ch_harian_max", "ch_3hari_max", "ch_7hari_max",
]
OUTPUT_COLUMNS = ["id_desa", "kabupaten", "kecamatan", "desa", "kelas", *FEATURES]


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def main() -> None:
    frame = pd.read_csv(INPUT)
    required = {*KEY_COLUMNS, "kelas", *FEATURES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {', '.join(missing)}")

    frame = frame.copy()
    frame["id_desa"] = frame[KEY_COLUMNS].astype(str).agg("|".join, axis=1)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for village_id, village in frame.groupby("id_desa", sort=True):
        filename = OUTPUT / f"desa_{slug(village_id.replace('|', '_'))}.csv"
        village[OUTPUT_COLUMNS].to_csv(filename, index=False)

    print(f"Diekspor {frame['id_desa'].nunique():,} file ke {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()