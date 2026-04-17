"""
Stage F - UCSF demographics.csv.

One row per entity (= wave cycle = admission-level key).

Columns:
  entity_id            {Patient_ID_GE}_{WaveCycleUID}
  patient_id_ge        str
  encounter_id         str
  age_years            int or ""; from Filtered_Encounters_New.Encounter_Age
  gender               ""   (UCSF data has no gender field in Filtered_* tables)
  ethnicity            ""   (not available)
  insurance            ""   (not available)
  admission_type       str; from Filtered_Encounters_New.Encounter_Admission_Type
  los_days             int; from offset xlsx (via Stage A parquet)
  admission_start_ms   int
  admission_end_ms     int
  wave_start_ms        int
  wave_end_ms          int
  has_ca               0/1

Age + admission_type are joined by `encounter_id` from
Filtered_Encounters_New/*.txt (dirty-comma CSV, repaired).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "workzone" / "ucsf"))
from readers.csv_repair import read_dirty_csv  # noqa: E402

CONFIG_PATH = REPO_ROOT / "workzone" / "configs" / "server_paths.yaml"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ENC_COLS = ["Encounter_ID", "Encounter_Age", "Encounter_Admission_Type"]


def load_encounter_table(raw_ehr_dir: Path,
                         wanted_ids: set[int]) -> dict[int, dict]:
    """Scan Filtered_Encounters_New/*.txt, return {encounter_id: {age, admit_type}}."""
    enc_dir = raw_ehr_dir / "Filtered_Encounters_New"
    shards = sorted(enc_dir.glob("*.txt"))
    log.info(f"scanning {len(shards)} encounter shards")
    out: dict[int, dict] = {}
    for i, p in enumerate(shards, 1):
        try:
            df = read_dirty_csv(str(p), quote_aware=True)
        except Exception as e:
            log.warning(f"  {p.name}: parse error {e}")
            continue
        if "Encounter_ID" not in df.columns:
            continue
        df = df.with_columns(pl.col("Encounter_ID").cast(pl.Int64, strict=False))
        df = df.filter(pl.col("Encounter_ID").is_in(list(wanted_ids)))
        if df.height == 0:
            continue
        keep = [c for c in ENC_COLS if c in df.columns]
        df = df.select(keep)
        for row in df.iter_rows(named=True):
            eid = row.get("Encounter_ID")
            if eid is None:
                continue
            out[int(eid)] = {
                "age": row.get("Encounter_Age"),
                "admit_type": row.get("Encounter_Admission_Type"),
            }
        if i % 50 == 0 or i == len(shards):
            log.info(f"  [{i}/{len(shards)}] matched {len(out)} encounters")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())["ucsf"]
    raw_ehr_dir = Path(cfg["raw_ehr_dir"])
    output_dir = Path(cfg["output_dir"])
    intermediate_dir = Path(cfg["intermediate_dir"])
    parquet = intermediate_dir / "valid_wave_window.parquet"

    t0 = time.time()
    df = (
        pl.read_parquet(parquet)
        .unique("entity_id", keep="first")
    )

    # Keep entities that completed Stage B (have meta.json)
    universe = {
        p.name for p in output_dir.iterdir()
        if p.is_dir() and (p / "meta.json").exists()
    }
    df = df.filter(pl.col("entity_id").is_in(list(universe)))
    log.info(f"entities in cohort: {df.height}")

    # Collect encounter IDs we need
    enc_ids: set[int] = set()
    for v in df["encounter_id"].to_list():
        try:
            enc_ids.add(int(v))
        except Exception:
            pass
    log.info(f"unique encounter_ids to look up: {len(enc_ids)}")

    enc_info = load_encounter_table(raw_ehr_dir, enc_ids)
    log.info(f"encounters matched: {len(enc_info)}/{len(enc_ids)}")

    # Pull wave_start/wave_end from each entity's time_ms.npy
    log.info("reading wave bounds from time_ms.npy ...")
    wave_bounds: dict[str, tuple[int, int]] = {}
    for eid in df["entity_id"].to_list():
        p = output_dir / eid / "time_ms.npy"
        if not p.exists():
            continue
        t = np.load(p, mmap_mode="r")
        if len(t):
            wave_bounds[eid] = (int(t[0]), int(t[-1]))

    out_path = output_dir / "demographics.csv"
    n_written = 0
    n_missing_age = 0
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "entity_id", "patient_id_ge", "encounter_id",
            "age_years", "gender", "ethnicity", "insurance",
            "admission_type",
            "los_days",
            "admission_start_ms", "admission_end_ms",
            "wave_start_ms", "wave_end_ms",
            "has_ca",
        ])
        for row in df.iter_rows(named=True):
            eid = row["entity_id"]
            enc_id = row.get("encounter_id")
            info = enc_info.get(int(enc_id)) if enc_id is not None else None
            age = info["age"] if info else None
            admit_type = info["admit_type"] if info else None
            if age is None or age == "":
                n_missing_age += 1
            ws, we = wave_bounds.get(eid, (None, None))
            w.writerow([
                eid,
                str(row["patient_id_ge"]),
                str(enc_id) if enc_id is not None else "",
                "" if age is None else age,
                "",  # gender not in UCSF filtered tables
                "",
                "",
                "" if admit_type is None else admit_type,
                int(row["encounter_los_days"]),
                int(row["admission_start_ms"]),
                int(row["admission_end_ms"]),
                "" if ws is None else ws,
                "" if we is None else we,
                int(row["has_ca"]),
            ])
            n_written += 1

    elapsed = time.time() - t0
    log.info(f"wrote {out_path}  rows={n_written}  missing_age={n_missing_age}  "
             f"elapsed={elapsed:.1f}s")

    summary = {
        "stage": "f_demographics",
        "ran_at_unix": int(time.time()),
        "elapsed_sec": round(elapsed, 1),
        "n_rows": n_written,
        "n_missing_age": n_missing_age,
        "output_csv": str(out_path),
    }
    (intermediate_dir / "stage_f_demographics_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
