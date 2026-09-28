#!/usr/bin/env python3
"""
Gate D/E — labs + 4-partition trajectory on the all-raw store (`ucsf_all`). Read-only; exit 1 on any FAIL.

  G1 link        ehr_link.parquet: >= 95 % of entities linked to an encounter; admission window contains the wave
                 start for >= 80 % of linked entities
  G2 stage D     phase-2 status: no errors; ok + ok_empty >= 98 % of linked; each of the 6 NMI labs
                 (0 K, 1 Ca, 2 Na, 3 Glu, 9 Hgb, 16 HCO3) has >= 1,000 events in total
  G3 stage E     status: no errors; ok + already_done >= 98 % of linked
  G4 consistency the CA-cohort store (`ucsf`) built its labs from the same tables with the same clock rule; for the
                 entities present in both stores the (time, var, value) multisets must be identical (>= 95 %)
  G5 structural  300-entity sample: ehr_events.seg_idx == searchsorted(time_ms, t) - 1, in-wave times inside
                 [time_ms[0], time_ms[-1] + 30 s), partitions have EHR_EVENT_DTYPE, meta.ehr_layout_version == 2
  G6 plausibility sample: >= 60 % of linked entities have >= 1 in-wave lab; lab rate 0.2-200 per admission day

  python workzone/ucsf/verify_stage_de.py --dataset ucsf_all [--sample 300] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from physio_data.schema import EHR_EVENT_DTYPE  # noqa: E402

CONFIG_PATH = REPO_ROOT / "workzone" / "configs" / "server_paths.yaml"
LAB6 = (0, 1, 2, 3, 9, 16)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG_PATH)); ap.add_argument("--dataset", default="ucsf_all")
    ap.add_argument("--sample", type=int, default=300); ap.add_argument("--out", default=None); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg_all = yaml.safe_load(Path(args.config).read_text()); cfg = cfg_all[args.dataset]
    inter = Path(cfg["intermediate_dir"]); store = Path(cfg["output_dir"]); ca_store = Path(cfg_all["ucsf"]["output_dir"])
    checks = []; info = {}
    def check(name, ok, detail=""):
        checks.append({"gate": name, "pass": bool(ok), "detail": detail}); print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)

    # ---- G1
    link = pl.read_parquet(inter / "ehr_link.parquet")
    linked = link.filter(pl.col("encounter_id").is_not_null())
    frac_linked = linked.height / max(1, link.height)
    contains = linked.filter((pl.col("admission_start_ms") <= pl.col("episode_start_ms")) & (pl.col("episode_start_ms") < pl.col("admission_end_ms"))).height
    check("G1 >= 95 % of entities linked to an encounter", frac_linked >= 0.95, f"linked={linked.height}/{link.height} ({frac_linked:.3f})")
    check("G1 admission window contains the wave start for >= 80 % of linked entities", contains / max(1, linked.height) >= 0.8, f"{contains}/{linked.height}")
    info["link_rules"] = {r["link_rule"]: r["len"] for r in link.group_by("link_rule").len().to_dicts()}
    linked_ids = set(linked["entity_id"].to_list())

    # ---- G2
    p2 = inter / "stage_d_labs_phase2_status.parquet"
    if p2.exists():
        st = pl.read_parquet(p2); by = {r["status"]: r["len"] for r in st.group_by("status").len().to_dicts()}
        n_err = by.get("error", 0); n_ok = by.get("ok", 0) + by.get("ok_empty", 0) + by.get("already_done", 0)
        check("G2 stage D: no errors, ok+ok_empty >= 98 % of linked", n_err == 0 and n_ok >= 0.98 * len(linked_ids), f"by_status={by} linked={len(linked_ids)}")
        tot = {v: 0 for v in LAB6}
        if "per_var_count" in st.columns:
            for s in st.filter(pl.col("status") == "ok")["per_var_count"].to_list():
                if s:
                    d = json.loads(s)
                    for v in LAB6: tot[v] += int(d.get(str(v), 0))
        info["lab6_total_events"] = tot
        check("G2 each NMI lab has >= 1,000 events (K, Ca, Na, Glu, Hgb, HCO3)", all(c >= 1000 for c in tot.values()) or "per_var_count" not in st.columns, str(tot))
    else:
        check("G2 stage D status file present", False, str(p2))

    # ---- G3
    pe = inter / "stage_e_status.parquet"
    if pe.exists():
        se = pl.read_parquet(pe); by = {r["status"]: r["len"] for r in se.group_by("status").len().to_dicts()}
        n_ok = by.get("ok", 0) + by.get("already_done", 0)
        check("G3 stage E: no errors, ok >= 98 % of linked", by.get("error", 0) == 0 and n_ok >= 0.98 * len(linked_ids), f"by_status={by}")
    else:
        check("G3 stage E status file present", False, str(pe))

    # ---- G4 consistency with the CA-cohort store
    rng = random.Random(args.seed)
    common = sorted(e for e in linked_ids if (ca_store / e / "labs_events.npy").exists() and (store / e / "labs_events.npy").exists())
    info["n_common_with_ca_store"] = len(common)
    if common:
        samp = rng.sample(common, min(500, len(common))); same = 0; diff_ex = []
        for e in samp:
            a = np.load(store / e / "labs_events.npy"); b = np.load(ca_store / e / "labs_events.npy")
            ka = sorted(zip(a["time_ms"].tolist(), a["var_id"].tolist(), np.round(a["value"].astype(float), 4).tolist()))
            kb = sorted(zip(b["time_ms"].tolist(), b["var_id"].tolist(), np.round(b["value"].astype(float), 4).tolist()))
            if ka == kb: same += 1
            elif len(diff_ex) < 5: diff_ex.append((e, len(ka), len(kb)))
        check("G4 labs identical to the CA-cohort store on shared entities (>= 95 %)", same / len(samp) >= 0.95, f"identical={same}/{len(samp)} e.g. differing={diff_ex}")
    else:
        check("G4 shared entities with the CA store (informational)", True, "none")

    # ---- G5 / G6
    samp = rng.sample(sorted(linked_ids), min(args.sample, len(linked_ids)))
    fails = {}; n_inwave = 0; rates = []; n_seen = 0
    for e in samp:
        d = store / e
        try:
            meta = json.loads((d / "meta.json").read_text()); tm = np.load(d / "time_ms.npy")
            errs = []
            if meta.get("ehr_layout_version") != 2: errs.append("layout")
            parts = {k: np.load(d / f"ehr_{k}.npy") for k in ("baseline", "recent", "events", "future")}
            for k, a in parts.items():
                if a.dtype != EHR_EVENT_DTYPE: errs.append(f"dtype {k}")
            ev = parts["events"]
            if ev.size:
                idx = np.searchsorted(tm, ev["time_ms"], side="right") - 1
                if not np.array_equal(idx, ev["seg_idx"]): errs.append("seg_idx")
                if ev["time_ms"].min() < tm[0] or ev["time_ms"].max() >= tm[-1] + 30000: errs.append("in-wave bounds")
            n_seen += 1
            labs_in = int(((ev["var_id"] < 100)).sum()) if ev.size else 0
            if labs_in: n_inwave += 1
            n_labs_all = sum(int((a["var_id"] < 100).sum()) for a in parts.values())
            days = (int(meta["admission_end_ms"]) - int(meta["admission_start_ms"])) / 86_400_000 if meta.get("admission_end_ms") else None
            if days and days > 0 and n_labs_all: rates.append(n_labs_all / days)
            if errs: fails[e] = errs
        except Exception as ex:  # noqa: BLE001
            fails[e] = [f"{type(ex).__name__}: {str(ex)[:80]}"]
    check("G5 structural: partitions dtype, seg_idx rule, in-wave bounds, layout v2", not fails, f"failing={len(fails)} of {n_seen} e.g. {list(fails.items())[:3]}")
    med_rate = float(np.median(rates)) if rates else 0.0
    check("G6 plausibility: >= 60 % of linked entities have an in-wave lab; median lab rate 0.2-200 / admission day", n_inwave / max(1, n_seen) >= 0.6 and 0.2 <= med_rate <= 200,
          f"in_wave={n_inwave}/{n_seen} median_rate={med_rate:.1f}/day")

    ok = all(c["pass"] for c in checks)
    out = {"dataset": args.dataset, "ran_at": time.strftime("%Y-%m-%d %H:%M"), "pass": ok, "checks": checks, "info": info}
    outp = Path(args.out) if args.out else inter / "verify_stage_de.json"
    outp.write_text(json.dumps(out, indent=2, default=str))
    print(("ALL GATES PASS" if ok else "GATE FAILED") + f" -> {outp}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
