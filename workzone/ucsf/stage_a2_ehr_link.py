#!/usr/bin/env python3
"""
Stage A2 — EHR linkage for the all-raw store (`ucsf_all`): entity (wave cycle) -> encounter, patient, day offsets.

The all-raw Stage A (`stage_a_all_raw.py`) enumerates wave cycles from the waveform tree alone. To attach labs
(Stage D), the 4-partition trajectory (Stage E) and demographics (Stage F) — the same files the CA-cohort store has —
each entity needs the EHR side of the link: `Patient_ID` (EHR id), `Encounter_ID`, `offset` / `offset_GE` (per-encounter
day shifts, datasets/ucsf/ALIGNMENT.md) and the admission window on the entity's UTC-continuous grid.

Source: the per-encounter offset table (27,903 encounters; `Patient_ID_GE`, `Wynton_folder`, `Encounter_ID`,
`Patient_ID`, `Encounter_Start_time` (EHR calendar), `Encounter_LOS`, `offset`, `offset_GE`). Rule (identical to the
CA-store Stage A, `stage_a_wave_windows.py`):
  1. join on (patient_id_ge, wynton_folder); entities without a folder match fall back to a patient-only join
     (`link_rule = "patient_only"`);
  2. admission_start = ehr_wall_to_grid_ms(Encounter_Start_time, offset, offset_GE, episode_start); end = start + LOS;
  3. several candidate encounters -> the one whose admission window contains the wave-cycle start, else the nearest
     admission start (deterministic).

Writes {intermediate_dir}/ehr_link.parquet (one row per entity, same columns Stage D/E/F consume from the CA-store
`valid_wave_window.parquet`) and stage_a2_ehr_link_summary.json.

  python workzone/ucsf/stage_a2_ehr_link.py --dataset ucsf_all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "workzone" / "ucsf"))
from clock import ehr_wall_to_grid_ms  # noqa: E402
from stage_a_wave_windows import load_offset_xlsx  # noqa: E402

CONFIG_PATH = REPO_ROOT / "workzone" / "configs" / "server_paths.yaml"
LINK_COLS = ["patient_id_ge", "wynton_folder", "encounter_id", "patient_id", "offset_days", "offset_ge_days",
             "encounter_los_days", "encounter_start_ehr_ms"]


def resolve(joined: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """admission window on the grid + deterministic choice among candidate encounters."""
    def _adm(s):
        if s["encounter_start_ehr_ms"] is None or s["offset_days"] is None or s["offset_ge_days"] is None:
            return None
        return int(ehr_wall_to_grid_ms(int(s["encounter_start_ehr_ms"]), float(s["offset_days"]),
                                       float(s["offset_ge_days"]), int(s["episode_start_ms"])))
    joined = joined.with_columns(
        pl.struct(["encounter_start_ehr_ms", "offset_days", "offset_ge_days", "episode_start_ms"])
          .map_elements(_adm, return_dtype=pl.Int64).alias("admission_start_ms"))
    joined = joined.with_columns(
        (pl.col("admission_start_ms") + (pl.col("encounter_los_days").cast(pl.Float64) * 86_400_000).round().cast(pl.Int64)).alias("admission_end_ms"))
    counts = joined.group_by("entity_id").agg(pl.len().alias("n_candidate_encounters"))
    joined = joined.join(counts, on="entity_id", how="left")
    joined = joined.with_columns(
        ((pl.col("admission_start_ms") <= pl.col("episode_start_ms")) & (pl.col("episode_start_ms") < pl.col("admission_end_ms")))
        .fill_null(False).alias("_contains"),
        (pl.col("episode_start_ms") - pl.col("admission_start_ms")).abs().fill_null(2**62).alias("_dist"))
    joined = (joined.sort(["entity_id", "_contains", "_dist"], descending=[False, True, False])
                    .unique("entity_id", keep="first", maintain_order=True))
    n_contained = int(joined["_contains"].sum())
    return joined.drop(["_contains", "_dist"]), n_contained


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG_PATH)); ap.add_argument("--dataset", default="ucsf_all")
    args = ap.parse_args()
    cfg_all = yaml.safe_load(Path(args.config).read_text()); cfg = cfg_all[args.dataset]
    inter = Path(cfg["intermediate_dir"]); output_dir = Path(cfg["output_dir"])
    offset_path = Path(cfg.get("offset_table_parquet") or cfg_all["ucsf"]["offset_table_parquet"])
    t0 = time.time()

    ents = (pl.read_parquet(inter / "valid_wave_window.parquet")
              .select(["entity_id", "patient_id_ge", "wynton_folder", "episode_start_ms", "episode_end_ms"])
              .unique("entity_id", keep="first")
              .with_columns(pl.col("patient_id_ge").cast(pl.Utf8).str.strip_chars(), pl.col("wynton_folder").cast(pl.Utf8).str.strip_chars()))
    print(f"entities (Stage A): {ents.height}", flush=True)
    off = load_offset_xlsx(offset_path).with_columns(pl.col("patient_id_ge").str.replace(r"^DE", ""))
    print(f"offset table rows: {off.height}, patients {off['patient_id_ge'].n_unique()}", flush=True)

    # 1. folder-aware join, 2. patient-only fallback
    j1 = ents.join(off.select(LINK_COLS), on=["patient_id_ge", "wynton_folder"], how="inner").with_columns(pl.lit("patient+folder").alias("link_rule"))
    left = ents.filter(~pl.col("entity_id").is_in(j1["entity_id"].unique().to_list()))
    j2 = left.join(off.select([c for c in LINK_COLS if c != "wynton_folder"]), on="patient_id_ge", how="inner").with_columns(pl.lit("patient_only").alias("link_rule"))
    unl = left.filter(~pl.col("entity_id").is_in(j2["entity_id"].unique().to_list()))
    joined = pl.concat([j1, j2], how="diagonal")
    linked, n_contained = resolve(joined)
    # keep unlinked entities in the table (nulls) so downstream can report them
    out = pl.concat([linked, unl.with_columns(pl.lit("unlinked").alias("link_rule"))], how="diagonal").sort(["patient_id_ge", "episode_start_ms"])

    # has_ca from the manifest when present (patient-level CA flag of the study cohort / Code Blue)
    man_p = output_dir / "manifest.json"
    if man_p.exists():
        man = json.loads(man_p.read_text())
        hc = pl.DataFrame({"entity_id": [m["entity_id"] for m in man], "has_ca": [int(bool(m.get("has_ca"))) for m in man]})
        out = out.join(hc, on="entity_id", how="left").with_columns(pl.col("has_ca").fill_null(0))
    else:
        out = out.with_columns(pl.lit(0).alias("has_ca"))

    inter.mkdir(parents=True, exist_ok=True)
    out.write_parquet(inter / "ehr_link.parquet")
    rules = out.group_by("link_rule").len().to_dicts()
    summ = {"stage": "a2_ehr_link", "dataset": args.dataset, "ran_at": time.strftime("%Y-%m-%d %H:%M"), "elapsed_sec": round(time.time() - t0, 1),
            "offset_table": str(offset_path), "n_entities": out.height, "n_linked": int(out["encounter_id"].is_not_null().sum()),
            "by_link_rule": {r["link_rule"]: r["len"] for r in rules}, "n_admission_contains_wave_start": n_contained,
            "n_multi_candidate": int(out.filter(pl.col("n_candidate_encounters") > 1).height),
            "n_patients": out["patient_id_ge"].n_unique(), "n_patients_linked": out.filter(pl.col("encounter_id").is_not_null())["patient_id_ge"].n_unique(),
            "output": str(inter / "ehr_link.parquet")}
    (inter / "stage_a2_ehr_link_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
