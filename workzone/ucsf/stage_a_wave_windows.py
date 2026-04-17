"""
Stage A — UCSF wave-window table from CA cohort CSV.

Reads:
  - {ca_eventtime_csv}  (8,272 rows: Patient_ID_GE, WaveCycleUID, ValidStart/Stop, EventTime)
  - {offset_xlsx}       (27,903 rows: per-encounter offset/offset_GE day-shifts)

Writes:
  - {intermediate_dir}/valid_wave_window.parquet
  - {intermediate_dir}/stage_a_summary.json

One row per entity (= {Patient_ID_GE}_{WaveCycleUID}).  The CA cohort CSV
already provides episode boundaries (ValidStartTime / ValidStopTime) so no
MRN-Mapping.csv scanning is needed.

Per-encounter day-shifts are joined from the offset xlsx to enable EHR→waveform
time alignment in later stages.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "workzone" / "configs" / "server_paths.yaml"


def load_cohort(path: Path) -> pl.DataFrame:
    """Load CA cohort CSV and derive entity fields."""
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    needed = {"Patient_ID_GE", "WaveCycleUID", "Wynton_folder", "UnitBed",
              "ValidStartTime", "ValidStopTime", "EventTime"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"CA cohort CSV missing columns: {missing}")

    # Strip DE prefix if present (some rows have "DE12345", others "12345")
    df["patient_id_ge_raw"] = df["Patient_ID_GE"].str.strip()
    df["patient_id_ge"] = df["patient_id_ge_raw"].str.replace(r"^DE", "", regex=True)

    df["wave_cycle_uid"] = df["WaveCycleUID"].str.strip()
    df["entity_id"] = df["patient_id_ge"] + "_" + df["wave_cycle_uid"]
    df["wynton_folder"] = df["Wynton_folder"].str.strip()
    df["unit_bed"] = df["UnitBed"].str.strip()
    df["bed_subdir"] = df["unit_bed"] + "-" + df["patient_id_ge_raw"]

    # Parse episode boundaries
    df["valid_start"] = pd.to_datetime(df["ValidStartTime"], errors="coerce")
    df["valid_stop"] = pd.to_datetime(df["ValidStopTime"], errors="coerce")
    df["episode_start_ms"] = (df["valid_start"].astype("int64") // 1_000_000)
    df["episode_end_ms"] = (df["valid_stop"].astype("int64") // 1_000_000)
    df["episode_duration_sec"] = (df["valid_stop"] - df["valid_start"]).dt.total_seconds()

    # CA label: -1 means no event, otherwise it's an ISO timestamp
    df["has_ca"] = (df["EventTime"].str.strip() != "-1").astype(int)
    df["event_time_raw"] = df["EventTime"].str.strip()

    keep = ["entity_id", "patient_id_ge", "wave_cycle_uid", "wynton_folder",
            "unit_bed", "bed_subdir", "episode_start_ms", "episode_end_ms",
            "episode_duration_sec", "has_ca", "event_time_raw"]
    out = df[keep].copy()
    out = out.dropna(subset=["episode_start_ms", "episode_end_ms"])
    out = out[out["episode_duration_sec"] > 0]
    return pl.from_pandas(out)


def load_offset_xlsx(path: Path) -> pl.DataFrame:
    """Load offset xlsx, return one row per (Patient_ID_GE, Wynton_folder).

    Derives admission_start_ms / admission_end_ms in GE (waveform) time:
        admission_start_GE = Encounter_Start_time_EHR - (offset_GE - offset) days
        admission_end_GE   = admission_start_GE + Encounter_LOS days
    These bound the trajectory partitions (baseline/recent/future) in Stage E.
    """
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    keep = ["Patient_ID_GE", "Wynton_folder", "Encounter_ID", "Patient_ID",
            "Encounter_Start_time", "Encounter_LOS", "offset", "offset_GE"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise RuntimeError(f"offset xlsx missing columns: {missing}; got {list(df.columns)}")
    df = df[keep].copy()

    df["patient_id_ge"] = df["Patient_ID_GE"].astype(str).str.strip()
    df["wynton_folder"] = df["Wynton_folder"].astype(str).str.strip()
    df["Encounter_Start_time"] = pd.to_datetime(df["Encounter_Start_time"], errors="coerce")

    enc_start_ehr_ms = df["Encounter_Start_time"].astype("int64") // 1_000_000
    shift_ms = (df["offset_GE"].astype("int64") - df["offset"].astype("int64")) * 86_400_000
    df["admission_start_ms"] = enc_start_ehr_ms - shift_ms
    df["admission_end_ms"] = df["admission_start_ms"] + df["Encounter_LOS"].astype("int64") * 86_400_000

    df = df.rename(columns={
        "Encounter_ID": "encounter_id",
        "Patient_ID": "patient_id",
        "offset": "offset_days",
        "offset_GE": "offset_ge_days",
        "Encounter_LOS": "encounter_los_days",
    })
    return pl.from_pandas(df[["patient_id_ge", "wynton_folder", "encounter_id",
                              "patient_id", "offset_days", "offset_ge_days",
                              "encounter_los_days",
                              "admission_start_ms", "admission_end_ms"]])


def main():
    ap = argparse.ArgumentParser(description="Stage A: build entity wave-window table from CA cohort")
    ap.add_argument("--config", default=str(CONFIG_PATH))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())["ucsf"]
    ca_csv = Path(cfg["ca_eventtime_csv"])
    offset_xlsx = Path(cfg["offset_xlsx"])
    out_dir = Path(cfg["intermediate_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"ca_cohort_csv    = {ca_csv}")
    print(f"offset_xlsx      = {offset_xlsx}")
    print(f"intermediate_dir = {out_dir}")

    t0 = time.time()

    # 1. Load cohort
    cohort = load_cohort(ca_csv)
    print(f"cohort rows (valid episodes): {cohort.height}")
    print(f"  unique entities:  {cohort.select(pl.col('entity_id').n_unique()).item()}")
    print(f"  unique patients:  {cohort.select(pl.col('patient_id_ge').n_unique()).item()}")
    print(f"  CA-positive rows: {cohort.filter(pl.col('has_ca') == 1).height}")

    # 2. Load offsets
    offset_df = load_offset_xlsx(offset_xlsx)
    print(f"offset xlsx rows: {offset_df.height}")

    # 3. Join (left) on (patient_id_ge, wynton_folder)
    joined = cohort.join(offset_df, on=["patient_id_ge", "wynton_folder"], how="left")

    # Handle multi-encounter matches: annotate + deduplicate
    counts = joined.group_by("entity_id").agg(pl.len().alias("n_candidate_encounters"))
    joined = joined.join(counts, on="entity_id", how="left")

    n_unmatched = joined.filter(pl.col("encounter_id").is_null()).select(pl.col("entity_id").n_unique()).item()
    n_multi = joined.filter(pl.col("n_candidate_encounters") > 1).select(pl.col("entity_id").n_unique()).item()
    print(f"after offset join: {joined.height} rows")
    print(f"  unmatched entities (no offset):      {n_unmatched}")
    print(f"  entities with >1 candidate encounter: {n_multi}")

    # 4. Sort + write
    joined = joined.sort(["patient_id_ge", "episode_start_ms"])
    out_parquet = out_dir / "valid_wave_window.parquet"
    joined.write_parquet(out_parquet)
    print(f"wrote {out_parquet}")

    elapsed = time.time() - t0

    # 5. Summary
    median_sec = cohort.select(pl.col("episode_duration_sec").median()).item()
    summary = {
        "stage": "a_wave_windows",
        "source": "ca_cohort_csv",
        "ran_at_unix": int(time.time()),
        "elapsed_sec": round(elapsed, 2),
        "ca_cohort_csv": str(ca_csv),
        "offset_xlsx": str(offset_xlsx),
        "n_cohort_rows_raw": cohort.height,
        "n_entities": cohort.select(pl.col("entity_id").n_unique()).item(),
        "n_unique_patients": cohort.select(pl.col("patient_id_ge").n_unique()).item(),
        "n_ca_positive_rows": int(cohort.filter(pl.col("has_ca") == 1).height),
        "n_ca_positive_patients": int(
            cohort.filter(pl.col("has_ca") == 1)
            .select(pl.col("patient_id_ge").n_unique()).item()
        ),
        "n_rows_after_offset_join": joined.height,
        "n_entities_unmatched_offset": n_unmatched,
        "n_entities_multi_encounter": n_multi,
        "median_episode_hours": round(median_sec / 3600, 2) if median_sec else None,
        "output_parquet": str(out_parquet),
    }
    out_summary = out_dir / "stage_a_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_summary}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
