"""
Stage G (hf events) — MIMIC-III-compatible `ehr_hf.npy` derived from the dense `vitals_hf.npy`.

Consumers built for MIMIC-III (phase-4 sparse-target path, PatientStore.ehr_hf) expect per-entity
`ehr_hf.npy` with EHR_EVENT_DTYPE (time_ms:int64, seg_idx:int32, var_id:uint16, value:float32) and
var_ids 150.. from var_registry.json. UCSF stores the same quantities densely at 0.5 Hz
(vitals_hf.npy [n_seg, 15, n_var]); this stage projects them onto the MIMIC convention:

  --cadence pair     (default, byte-compatible with mimic3/stage3b_extract_numerics.py)
      every 2 segments = 60 s = one tick; for pair (2k, 2k+1) the reading nearest the PAIR-END time
      t_q = time_ms[2k+1] + 30 s within +-30 s is written at seg_idx = 2k+1, time_ms = t_q.
      Nearest = last valid slot of segment 2k+1 (slot 14 -> t_q - 1 s), else earlier slots of that
      segment / earlier-first slots of segment 2k+2, by |slot centre - t_q|.
  --cadence segment  one value per 30 s segment at seg_idx = i, time_ms = time_ms[i] + 30 s (segment
      end), value = reading nearest the segment end within the segment (denser; UCSF-native).

Variables written: 150 HR_hf, 151 SpO2_hf, 152 RR_hf, 153-155 ABP s/d/m_hf, 156 PULSE_hf
(+ 113 PR_art with --include-pr-art). NBP (157-159) already lives in nbp_events.npy and is appended
when --include-nbp is given (its times are true cuff times, not ticks). Canonical files are untouched;
meta.json gains an `ehr_hf` section. Resume skips entities whose meta.ehr_hf.version matches.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from physio_data.schema import EHR_EVENT_DTYPE  # noqa: E402

CONFIG_PATH = REPO_ROOT / "workzone" / "configs" / "server_paths.yaml"
MAX_WORKERS = 22
EHR_HF_VERSION = 1
SEG_MS = 30_000; SLOT_MS = 2_000; SLOTS = 15
MIMIC_IDS = (150, 151, 152, 153, 154, 155, 156)


def nearest_valid(block: np.ndarray, centres_ms: np.ndarray, t_q: float, tol_ms: float) -> float:
    """block: values for candidate slots (1-D), centres_ms: their centre times. Nearest finite within tol."""
    ok = np.isfinite(block)
    if not ok.any():
        return np.nan
    d = np.abs(centres_ms - t_q); d[~ok] = np.inf
    j = int(np.argmin(d))
    return float(block[j]) if d[j] <= tol_ms else np.nan


def project_pair(hf: np.ndarray, time_ms: np.ndarray, col: int, tol_ms: float = 30_000.0):
    """MIMIC pair-end convention -> (seg_idx array, time array, values) for one variable column."""
    n_seg = hf.shape[0]
    q = np.arange(1, n_seg, 2)                       # 2k+1
    seg_out, t_out, v_out = [], [], []
    for s in q:
        t_q = time_ms[s] + SEG_MS                      # pair-end
        # candidates: whole segment s (before t_q) and, if it exists, segment s+1 (after t_q)
        cand_v = hf[s, :, col]
        cand_c = time_ms[s] + np.arange(SLOTS) * SLOT_MS + SLOT_MS / 2
        if s + 1 < n_seg:
            cand_v = np.concatenate([cand_v, hf[s + 1, :, col]])
            cand_c = np.concatenate([cand_c, time_ms[s + 1] + np.arange(SLOTS) * SLOT_MS + SLOT_MS / 2])
        v = nearest_valid(cand_v, cand_c, t_q, tol_ms)
        if np.isfinite(v):
            seg_out.append(int(s)); t_out.append(int(t_q)); v_out.append(v)
    return seg_out, t_out, v_out


def project_segment(hf: np.ndarray, time_ms: np.ndarray, col: int):
    """One value per segment: reading nearest the segment end (last finite slot)."""
    n_seg = hf.shape[0]
    seg_out, t_out, v_out = [], [], []
    block = hf[:, :, col]
    fin = np.isfinite(block)
    for i in np.flatnonzero(fin.any(axis=1)):
        j = int(np.flatnonzero(fin[i])[-1])
        seg_out.append(int(i)); t_out.append(int(time_ms[i] + SEG_MS)); v_out.append(float(block[i, j]))
    return seg_out, t_out, v_out


def process_entity(eid: str, output_dir: str, cadence: str, include_pr_art: bool, include_nbp: bool) -> dict:
    d = Path(output_dir) / eid
    st = {"entity_id": eid, "status": "pending", "n_events": 0}
    try:
        meta = json.loads((d / "meta.json").read_text())
        vh = meta.get("vitals_hf")
        if not vh or not (d / "vitals_hf.npy").exists():
            st["status"] = "no_vitals_hf"; return st
        hf = np.load(d / "vitals_hf.npy", mmap_mode="r")
        time_ms = np.load(d / "time_ms.npy")
        var_ids = list(vh["var_ids"])
        want = [v for v in var_ids if v in MIMIC_IDS or (include_pr_art and v == 113)]
        rows = []
        for vid in want:
            col = var_ids.index(vid)
            if cadence == "pair":
                segs, ts, vals = project_pair(np.asarray(hf), time_ms, col)
            else:
                segs, ts, vals = project_segment(np.asarray(hf), time_ms, col)
            rows.extend((t, s, vid, v) for t, s, v in zip(ts, segs, vals))
        if include_nbp and (d / "nbp_events.npy").exists():
            nbp = np.load(d / "nbp_events.npy")
            rows.extend((int(r["time_ms"]), int(r["seg_idx"]), int(r["var_id"]), float(r["value"])) for r in nbp)
        arr = np.array(rows, dtype=EHR_EVENT_DTYPE) if rows else np.empty(0, dtype=EHR_EVENT_DTYPE)
        arr.sort(order=["time_ms", "var_id"])
        np.save(d / "ehr_hf.npy", arr)
        counts = {int(v): int((arr["var_id"] == v).sum()) for v in sorted(set(arr["var_id"].tolist()))}
        meta["ehr_hf"] = {"version": EHR_HF_VERSION, "file": "ehr_hf.npy", "dtype": "EHR_EVENT_DTYPE",
                          "source": "vitals_hf.npy", "cadence": cadence,
                          "convention": ("mimic3 stage3b pair-end: seg_idx=2k+1, t=time_ms[2k+1]+30s, nearest reading within 30 s"
                                         if cadence == "pair" else "per 30 s segment: seg_idx=i, t=segment end, last valid slot"),
                          "var_ids": sorted(counts), "n_events": int(arr.size), "per_var_count": counts,
                          "includes_nbp_events": bool(include_nbp), "includes_pr_art": bool(include_pr_art)}
        (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        st.update(status="ok", n_events=int(arr.size)); return st
    except Exception as e:
        st.update(status="error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-500:]); return st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG_PATH)); ap.add_argument("--dataset", default="ucsf_all")
    ap.add_argument("--cadence", choices=["pair", "segment"], default="pair")
    ap.add_argument("--include-pr-art", action="store_true"); ap.add_argument("--include-nbp", action="store_true")
    ap.add_argument("--entities", default=""); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--entity-file", default="", help="one entity_id per line (merged with --entities)")
    ap.add_argument("--dst-switch-only", action="store_true", help="only entities whose meta.json has dst_switch_in_cycle")
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--batch-size", type=int, default=500)
    args = ap.parse_args()
    args.workers = min(args.workers, MAX_WORKERS)
    cfg = yaml.safe_load(Path(args.config).read_text())[args.dataset]
    out_dir = Path(cfg["output_dir"]); inter = Path(cfg["intermediate_dir"])
    man = json.loads((out_dir / "manifest.json").read_text())
    ids = [e["entity_id"] for e in man]
    want = {s.strip() for s in args.entities.split(",") if s.strip()}
    if args.entity_file:
        want |= {s.strip() for s in Path(args.entity_file).read_text().splitlines() if s.strip() and not s.startswith("#")}
    if want:
        ids = [e for e in ids if e in want]
    elif args.limit:
        ids = ids[:args.limit]
    if args.dst_switch_only:
        keep = []
        for e in ids:
            try:
                if json.loads((out_dir / e / "meta.json").read_text()).get("dst_switch_in_cycle"): keep.append(e)
            except Exception:
                pass
        print(f"dst-switch-only: {len(keep)} entities"); ids = keep
    if not args.no_resume:
        keep = []
        for e in ids:
            try:
                m = json.loads((out_dir / e / "meta.json").read_text())
                if m.get("ehr_hf", {}).get("version") == EHR_HF_VERSION and m["ehr_hf"].get("cadence") == args.cadence and (out_dir / e / "ehr_hf.npy").exists():
                    continue
            except Exception:
                pass
            keep.append(e)
        print(f"resume: {len(ids) - len(keep)} already have ehr_hf v{EHR_HF_VERSION} ({args.cadence})"); ids = keep
    print(f"dataset={args.dataset} entities={len(ids)} cadence={args.cadence} workers={args.workers} pr_art={args.include_pr_art} nbp={args.include_nbp}")
    statuses = []; t0 = time.time(); done = 0; total = len(ids); ctx = mp.get_context("spawn")
    for b0 in range(0, total, args.batch_size):
        batch = ids[b0:b0 + args.batch_size]
        try:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
                futs = {ex.submit(process_entity, e, str(out_dir), args.cadence, args.include_pr_art, args.include_nbp): e for e in batch}
                for fut in as_completed(futs):
                    try: s = fut.result()
                    except BrokenProcessPool: s = {"entity_id": futs[fut], "status": "worker_killed"}
                    except Exception as e: s = {"entity_id": futs[fut], "status": "error", "error": str(e)[:200]}
                    statuses.append(s); done += 1
                    if done % 1000 == 0 or done == total: print(f"  [{done}/{total}] elapsed={time.time()-t0:.0f}s last={s['status']}", flush=True)
        except BrokenProcessPool as e:
            fin = {s["entity_id"] for s in statuses}
            statuses.extend({"entity_id": e_, "status": "worker_killed"} for e_ in batch if e_ not in fin); print(f"  BATCH {b0}: pool broken ({e})")
    by = {}
    for s in statuses: by[s["status"]] = by.get(s["status"], 0) + 1
    summary = {"stage": "g_ehr_hf", "dataset": args.dataset, "ran_at_unix": int(time.time()), "elapsed_sec": round(time.time() - t0, 1),
               "n_entities_input": total, "by_status": by, "cadence": args.cadence, "include_pr_art": args.include_pr_art, "include_nbp": args.include_nbp,
               "n_events_total": int(sum(s.get("n_events", 0) for s in statuses)), "output_dir": str(out_dir)}
    inter.mkdir(parents=True, exist_ok=True)
    (inter / "stage_g_ehr_hf_summary.json").write_text(json.dumps(summary, indent=2))
    if statuses: pl.DataFrame(statuses, infer_schema_length=None).write_parquet(inter / "stage_g_ehr_hf_status.parquet")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
