"""
Stage C (hf) — dense machine-sampled vitals sidecar for UCSF entities.

For each entity with a Stage B meta.json:
  1. Read the `.vital` streams listed by Stage A (HR, SPO2-%, RESP, SPO2-R, AR*/FE* S/D/M/R,
     NBP-S/D/M, CUFF).
  2. Put every sample on the entity's 30 s segment grid at 2 s resolution:
        vitals_hf.npy   float32 [n_seg, 15, n_var]   NaN = no reading; slot k covers
                        [seg_start + 2k, seg_start + 2k + 2) seconds
        vitals_hf_abp_src.npy  uint8 [n_seg, 15]     which arterial line filled the ABP slots
                        (0 none, 1 AR1, 2 AR2, 3 AR3, 4 FE1, 5 FE3)
  3. NBP is a 2 s stream that repeats the last cuff reading, so it is stored as EVENTS at value
     changes (EHR_EVENT_DTYPE, var_id 157/158/159) with the event time moved to the end of the
     matching CUFF inflation burst when one is found:
        nbp_events.npy
  4. meta.json gains a `vitals_hf` section (var ids/names, per-file timing class and policy,
     counts) and an `nbp_events` section.

Timing of `.vital` files (RAW_FORMAT.md §2): class A has a header time and relative offsets;
class B has a zero header time and relative offsets; class C is B whose offsets switch to absolute
seconds-since-0001 part-way. Policies:
  --zero-anchor adibin_first | skip   (B/C relative part; adibin_first = entity episode_start_ms,
                                       i.e. the first .adibin of the same wave cycle)
  --abs-tail drop | reanchor          (C absolute part; reanchor assumes continuity with the
                                       relative part: abs_first == rel_last + 2 s)
Canonical Stage B files are never modified.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import struct
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from physio_data.schema import EHR_EVENT_DTYPE  # noqa: E402
sys.path.insert(0, str(REPO_ROOT / "workzone" / "ucsf"))
from clock import ge_wall_to_grid_ms, dst_switch_between  # noqa: E402

CONFIG_PATH = REPO_ROOT / "workzone" / "configs" / "server_paths.yaml"
VAR_REGISTRY_PATH = REPO_ROOT / "indices" / "var_registry.json"

MAX_WORKERS = 22
VH_FMT = "<16s8s8s4s iiiii d"; VH_SIZE = struct.calcsize(VH_FMT)
ABS_THRESHOLD = 6.0e10          # offsets above this are seconds since 0001-01-01
SLOT_SEC = 2; SEG_SEC = 30; SLOTS_PER_SEG = SEG_SEC // SLOT_SEC   # 15
VITALS_HF_VERSION = 2   # v2: UTC-continuous grid (header wall clock -> grid via clock.ge_wall_to_grid_ms)

ABP_LINES = ("AR1", "AR2", "AR3", "FE1", "FE3")
ABP_LINE_CODE = {ln: i + 1 for i, ln in enumerate(ABP_LINES)}
# (var_id, name, suffix or ABP kind). ABP kinds are resolved per line (AR1-S, AR2-S, ...).
HF_VARS = [
    (150, "HR_hf", "HR"),
    (151, "SpO2_hf", "SPO2-%"),
    (152, "RR_hf", "RESP"),
    (153, "ABPs_hf", "ABP:S"),
    (154, "ABPd_hf", "ABP:D"),
    (155, "ABPm_hf", "ABP:M"),
    (156, "PULSE_hf", "SPO2-R"),
    (113, "PR_art", "ABP:R"),
]
HF_VAR_IDS = [v[0] for v in HF_VARS]
HF_VAR_NAMES = [v[1] for v in HF_VARS]
NBP_VARS = [(157, "NBPs_hf", "NBP-S"), (158, "NBPd_hf", "NBP-D"), (159, "NBPm_hf", "NBP-M")]
CUFF_MIN_MMHG = 5.0            # CUFF samples above this are "inflating"
CUFF_BURST_GAP_S = 10          # samples closer than this belong to one inflation
CUFF_MATCH_BEFORE_S = 300      # an NBP change is matched to a burst ending within this window before it
CUFF_MATCH_AFTER_S = 10


def load_ranges(path: Path) -> dict[int, tuple[float, float]]:
    reg = json.loads(path.read_text())
    entries = reg["variables"] if isinstance(reg, dict) else reg
    out = {}
    for e in entries:
        if "physio_min" in e and "physio_max" in e:
            out[int(e["id"])] = (float(e["physio_min"]), float(e["physio_max"]))
    return out


def _cstr(b: bytes) -> str:
    return b.split(b"\0", 1)[0].decode("latin-1")


def read_vital(path: str):
    """-> (hdr dict, values f64, offsets f64) or None."""
    size = os.path.getsize(path)
    n = (size - VH_SIZE) // 32
    if n <= 0:
        return None
    with open(path, "rb") as f:
        hb = f.read(VH_SIZE)
    lab, uom, unit, bed, y, mo, d, h, mi, sec = struct.unpack(VH_FMT, hb)
    start_ms = None
    if y != 0:
        try:
            si = int(sec)
            start_ms = int(datetime(y, mo, d, h, mi, si, tzinfo=timezone.utc).timestamp() * 1000) + int(round((sec - si) * 1000))
        except Exception:
            start_ms = None
    mm = np.memmap(path, dtype=np.float64, mode="r", offset=VH_SIZE, shape=(n, 4))
    val = np.array(mm[:, 0]); off = np.array(mm[:, 1]); del mm
    return {"label": _cstr(lab), "uom": _cstr(uom), "start_ms": start_ms, "zero_time": start_ms is None, "n": int(n)}, val, off


def sample_times_ms(hdr: dict, off: np.ndarray, episode_start_ms: int, zero_anchor: str, abs_tail: str):
    """Absolute time (ms) per sample under the timing policy; NaN where the sample is unusable."""
    t = np.full(off.shape, np.nan)
    is_abs = off >= ABS_THRESHOLD
    rel = ~is_abs
    info = {"class": "A" if not hdr["zero_time"] else ("C" if is_abs.any() else "B"),
            "n": int(off.size), "n_rel": int(rel.sum()), "n_abs": int(is_abs.sum()), "policy": ""}
    anchor = hdr["start_ms"]
    if anchor is not None:
        anchor = int(ge_wall_to_grid_ms(anchor, episode_start_ms))   # header wall clock (GE calendar) -> UTC-continuous grid
    if anchor is None:
        if zero_anchor == "adibin_first":
            anchor = episode_start_ms; info["policy"] += "rel@adibin_first"
        else:
            info["policy"] += "rel_skipped"; anchor = None
    else:
        info["policy"] += "rel@header"
    if anchor is not None and rel.any():
        t[rel] = anchor + off[rel] * 1000.0
    if is_abs.any():
        ab = off[is_abs]
        degenerate = ab.size < 2 or float(np.median(np.diff(ab))) <= 0   # NBP abs tails repeat one offset
        if abs_tail == "reanchor" and anchor is not None and rel.any() and not degenerate:
            base = anchor + off[rel][-1] * 1000.0 + SLOT_SEC * 1000.0   # continuity: abs_first == rel_last + 2 s
            t[is_abs] = base + (ab - ab[0]) * 1000.0
            info["policy"] += "+abs_reanchored"
        else:
            info["policy"] += "+abs_dropped" + ("(degenerate)" if degenerate else "")
    return t, info


def to_slots(t_ms: np.ndarray, episode_start_ms: int, n_seg: int):
    """-> (flat slot index in [0, n_seg*15) or -1) per sample."""
    k = np.floor((t_ms - episode_start_ms) / (SLOT_SEC * 1000.0))
    k = np.where(np.isfinite(k), k, -1).astype(np.int64)
    k[(k < 0) | (k >= n_seg * SLOTS_PER_SEG)] = -1
    return k


def clean_values(val: np.ndarray, rng: tuple[float, float] | None):
    v = val.astype(np.float64)
    bad = ~np.isfinite(v) | (v <= -99999)
    if rng is not None:
        bad |= (v < rng[0]) | (v > rng[1])
    v[bad] = np.nan
    return v


def process_entity(row: dict, raw_dir: str, output_dir: str, ranges: dict, zero_anchor: str, abs_tail: str) -> dict:
    eid = row["entity_id"]; out_dir = Path(output_dir) / eid
    st = {"entity_id": eid, "status": "pending", "n_vital_files": 0, "n_files_used": 0,
          "n_slots_hr": 0, "n_slots_abp": 0, "n_nbp_events": 0}
    try:
        meta_path = out_dir / "meta.json"
        if not meta_path.exists():
            st["status"] = "no_meta"; return st
        meta = json.loads(meta_path.read_text())
        n_seg = int(meta["n_seg"]); ep_start = int(meta["episode_start_ms"])
        if int(meta.get("seg_duration_sec", SEG_SEC)) != SEG_SEC:
            st["status"] = "bad_seg_len"; return st
        files = row.get("vital_files") or []
        st["n_vital_files"] = len(files)
        by_suffix: dict[str, list[str]] = {}
        for rel in sorted(files):
            name = os.path.basename(rel)
            suffix = name[:-len(".vital")].rsplit("_", 1)[-1]
            by_suffix.setdefault(suffix, []).append(rel)   # a UID can have several files per suffix

        n_var = len(HF_VARS)
        hf = np.full((n_seg * SLOTS_PER_SEG, n_var), np.nan, dtype=np.float32)
        abp_src = np.zeros(n_seg * SLOTS_PER_SEG, dtype=np.uint8)
        file_info: dict[str, dict] = {}
        cache: dict[str, tuple] = {}

        def load(suffix):
            """All files of one suffix, concatenated in file order -> (val, t, k) or None."""
            if suffix in cache:
                return cache[suffix]
            parts = []
            for i, rel in enumerate(by_suffix.get(suffix, [])):
                r = read_vital(os.path.join(raw_dir, rel))
                if r is None:
                    continue
                hdr, val, off = r
                t, info = sample_times_ms(hdr, off, ep_start, zero_anchor, abs_tail)
                k = to_slots(t, ep_start, n_seg)
                info["n_in_window"] = int((k >= 0).sum()); info["uom"] = hdr["uom"]
                file_info[suffix if len(by_suffix[suffix]) == 1 else f"{suffix}#{i}"] = info
                parts.append((val, t, k))
            if not parts:
                cache[suffix] = None; return None
            cache[suffix] = tuple(np.concatenate([p[j] for p in parts]) for j in range(3))
            return cache[suffix]

        # simple streams
        for j, (vid, name, src) in enumerate(HF_VARS):
            if src.startswith("ABP:"):
                continue
            r = load(src)
            if r is None:
                continue
            val, t, k = r
            v = clean_values(val, ranges.get(vid)); ok = (k >= 0) & np.isfinite(v)
            hf[k[ok], j] = v[ok]                     # later samples overwrite (last wins)
            st["n_files_used"] += 1
        # arterial lines: coalesce by preference, never mix two lines in one slot
        abp_cols = {src.split(":")[1]: j for j, (vid, name, src) in enumerate(HF_VARS) if src.startswith("ABP:")}
        abp_vids = {src.split(":")[1]: vid for (vid, name, src) in HF_VARS if src.startswith("ABP:")}
        line_counts = {}
        for line in ABP_LINES:
            rS = load(f"{line}-S")
            if rS is None:
                continue
            valS, tS, kS = rS
            vS = clean_values(valS, ranges.get(abp_vids["S"]))
            okS = (kS >= 0) & np.isfinite(vS)
            free = okS & (abp_src[np.clip(kS, 0, None)] == 0)
            line_counts[line] = int(free.sum())
            if not free.any():
                continue
            slots = kS[free]
            abp_src[slots] = ABP_LINE_CODE[line]
            hf[slots, abp_cols["S"]] = vS[free]
            st["n_files_used"] += 1
            for kind in ("D", "M", "R"):
                r = load(f"{line}-{kind}")
                if r is None:
                    continue
                val, t, k = r
                v = clean_values(val, ranges.get(abp_vids[kind]))
                ok = (k >= 0) & np.isfinite(v) & (abp_src[np.clip(k, 0, None)] == ABP_LINE_CODE[line])
                hf[k[ok], abp_cols[kind]] = v[ok]
                st["n_files_used"] += 1

        # NBP change events (+ CUFF burst-end timing)
        bursts_end = np.empty(0)
        rC = load("CUFF")
        if rC is not None:
            valC, tC, kC = rC
            infl = np.isfinite(tC) & (valC > CUFF_MIN_MMHG)
            tt = np.sort(tC[infl])
            if tt.size:
                cut = np.flatnonzero(np.diff(tt) > CUFF_BURST_GAP_S * 1000.0)
                bursts_end = tt[np.r_[cut, tt.size - 1]]
        events = []
        n_corr = 0
        for vid, name, suffix in NBP_VARS:
            r = load(suffix)
            if r is None:
                continue
            val, t, k = r
            v = clean_values(val, ranges.get(vid)); ok = np.isfinite(t) & np.isfinite(v)
            if not ok.any():
                continue
            order = np.argsort(t[ok], kind="stable"); tv = t[ok][order]; vv = v[ok][order]
            chg = np.r_[True, np.diff(vv) != 0]
            te = tv[chg]; ve = vv[chg]
            if bursts_end.size:
                pos = np.searchsorted(bursts_end, te + CUFF_MATCH_AFTER_S * 1000.0, side="right") - 1
                cand = bursts_end[np.clip(pos, 0, None)]
                good = (pos >= 0) & (cand >= te - CUFF_MATCH_BEFORE_S * 1000.0)
                te = np.where(good, cand, te); n_corr += int(good.sum())
            seg = np.floor((te - ep_start) / (SEG_SEC * 1000.0)).astype(np.int64)
            inw = (seg >= 0) & (seg < n_seg)
            for a, b, c in zip(te[inw], seg[inw], ve[inw]):
                events.append((int(a), int(b), vid, float(c)))
            st["n_files_used"] += 1
        ev = np.array(events, dtype=EHR_EVENT_DTYPE) if events else np.empty(0, dtype=EHR_EVENT_DTYPE)
        ev.sort(order=["time_ms", "var_id"])

        hf3 = np.ascontiguousarray(hf.reshape(n_seg, SLOTS_PER_SEG, n_var))
        np.save(out_dir / "vitals_hf.npy", hf3)
        np.save(out_dir / "vitals_hf_abp_src.npy", np.ascontiguousarray(abp_src.reshape(n_seg, SLOTS_PER_SEG)))
        np.save(out_dir / "nbp_events.npy", ev)

        valid = np.isfinite(hf).sum(axis=0)
        meta["vitals_hf"] = {
            "version": VITALS_HF_VERSION, "file": "vitals_hf.npy", "dtype": "float32", "time_base": "utc_continuous",
            "shape": [n_seg, SLOTS_PER_SEG, n_var], "slot_sec": SLOT_SEC, "slots_per_seg": SLOTS_PER_SEG,
            "var_ids": HF_VAR_IDS, "var_names": HF_VAR_NAMES,
            "n_valid_per_var": {name: int(c) for name, c in zip(HF_VAR_NAMES, valid)},
            "valid_frac_per_var": {name: round(float(c) / hf.shape[0], 4) for name, c in zip(HF_VAR_NAMES, valid)},
            "abp_src_file": "vitals_hf_abp_src.npy", "abp_line_codes": ABP_LINE_CODE, "abp_slots_per_line": line_counts,
            "zero_anchor": zero_anchor, "abs_tail": abs_tail, "files": file_info,
            "source": "UCSF .vital monitor streams (2 s cadence)",
        }
        meta["nbp_events"] = {"file": "nbp_events.npy", "dtype": "EHR_EVENT_DTYPE", "var_ids": [v[0] for v in NBP_VARS],
                              "n_events": int(ev.size), "n_cuff_time_corrected": int(n_corr),
                              "rule": "value-change points of the 2 s NBP hold stream; time = end of matching CUFF burst when found"}
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        st.update(status="ok", n_slots_hr=int(valid[0]), n_slots_abp=int(valid[3]), n_nbp_events=int(ev.size),
                  n_seg=n_seg, classes=";".join(f"{s}:{i['class']}" for s, i in file_info.items()))
        return st
    except Exception as e:
        st["status"] = "error"; st["error"] = f"{type(e).__name__}: {e}"; st["traceback"] = traceback.format_exc()[-600:]
        return st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--dataset", default="ucsf_all")
    ap.add_argument("--zero-anchor", choices=["adibin_first", "skip"], default="adibin_first")
    ap.add_argument("--abs-tail", choices=["drop", "reanchor"], default="drop")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--entities", default="")
    ap.add_argument("--entity-file", default="", help="one entity_id per line (merged with --entities)")
    ap.add_argument("--dst-switch-only", action="store_true", help="only cycles straddling a DST switch on the GE calendar")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--batch-size", type=int, default=200)
    args = ap.parse_args()
    if args.workers > MAX_WORKERS:
        print(f"clamping workers {args.workers} -> {MAX_WORKERS}"); args.workers = MAX_WORKERS

    cfg = yaml.safe_load(Path(args.config).read_text())[args.dataset]
    raw_dir = cfg["raw_waveform_dir"]; output_dir = cfg["output_dir"]; inter = Path(cfg["intermediate_dir"])
    ranges = load_ranges(VAR_REGISTRY_PATH)
    df = pl.read_parquet(inter / "valid_wave_window.parquet").unique(subset=["entity_id"], keep="first")
    ids = [s.strip() for s in args.entities.split(",") if s.strip()]
    if args.entity_file:
        ids += [s.strip() for s in Path(args.entity_file).read_text().splitlines() if s.strip() and not s.startswith("#")]
    if ids:
        df = df.filter(pl.col("entity_id").is_in(ids))
    elif args.limit:
        df = df.head(args.limit)
    rows = df.select(["entity_id", "vital_files", "episode_start_ms", "episode_end_ms"]).to_dicts()
    if args.dst_switch_only:
        rows = [r for r in rows if dst_switch_between(int(r["episode_start_ms"]), int(r["episode_end_ms"]))]
        print(f"dst-switch-only: {len(rows)} cycles straddle a DST switch on the GE calendar")
    # only entities Stage B produced
    rows = [r for r in rows if (Path(output_dir) / r["entity_id"] / "meta.json").exists()]
    if not args.no_resume:
        keep = []
        for r in rows:
            d = Path(output_dir) / r["entity_id"]
            if (d / "vitals_hf.npy").exists():
                try:
                    if json.loads((d / "meta.json").read_text()).get("vitals_hf", {}).get("version") == VITALS_HF_VERSION:
                        continue
                except Exception:
                    pass
            keep.append(r)
        print(f"resume: {len(rows) - len(keep)} entities already have vitals_hf v{VITALS_HF_VERSION}")
        rows = keep
    print(f"dataset={args.dataset} entities={len(rows)} workers={args.workers} zero_anchor={args.zero_anchor} abs_tail={args.abs_tail}")

    statuses = []; t0 = time.time(); done = 0; total = len(rows)
    out_status = inter / "stage_c_hf_status.parquet"
    def flush():
        try: pl.DataFrame(statuses, infer_schema_length=None).write_parquet(out_status)
        except Exception as e: print(f"  (status parquet not written: {e})")
    ctx = mp.get_context("spawn")
    for b0 in range(0, total, args.batch_size):
        batch = rows[b0:b0 + args.batch_size]
        try:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
                futs = {ex.submit(process_entity, r, raw_dir, output_dir, ranges, args.zero_anchor, args.abs_tail): r["entity_id"] for r in batch}
                for fut in as_completed(futs):
                    eid = futs[fut]
                    try: s = fut.result()
                    except BrokenProcessPool: s = {"entity_id": eid, "status": "worker_killed"}
                    except Exception as e: s = {"entity_id": eid, "status": "error", "error": f"{type(e).__name__}: {str(e)[:200]}"}
                    statuses.append(s); done += 1
                    if done % 50 == 0 or done == total:
                        print(f"  [{done}/{total}] elapsed={time.time()-t0:.0f}s last={s['status']}", flush=True)
        except BrokenProcessPool as e:
            fin = {s["entity_id"] for s in statuses}
            for r in batch:
                if r["entity_id"] not in fin:
                    statuses.append({"entity_id": r["entity_id"], "status": "worker_killed"}); done += 1
            print(f"  BATCH {b0}: pool broken ({e})", flush=True)
        flush()
    by = {}
    for s in statuses:
        by[s["status"]] = by.get(s["status"], 0) + 1
    summary = {"stage": "c_vitals_hf", "dataset": args.dataset, "ran_at_unix": int(time.time()), "elapsed_sec": round(time.time() - t0, 1),
               "n_entities_input": total, "by_status": by, "zero_anchor": args.zero_anchor, "abs_tail": args.abs_tail,
               "n_nbp_events_total": int(sum(s.get("n_nbp_events", 0) for s in statuses)),
               "var_ids": HF_VAR_IDS, "var_names": HF_VAR_NAMES, "output_dir": output_dir}
    (inter / "stage_c_hf_summary.json").write_text(json.dumps(summary, indent=2))
    flush(); print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
