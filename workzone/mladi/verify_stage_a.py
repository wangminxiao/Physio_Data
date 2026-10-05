#!/usr/bin/env python3
"""Gate A for MLADI (exits non-zero on any failed check; writes verify_stage_a.json next to the inventory).

  entities    included count in [15,500, 17,500] (expected ~16.4 k: Pleth + >= 1 pretrain_wav_v2 row);
              entity ids unique; errors <= 1 % of files
  clock       among included entities with >= 3 exact NBP matches: verified + corrected >= 95 %,
              conflict <= 5 %
  split       no patient on two e1 sides
  disk        Stage B..G need (grid rows x bytes per segment) vs project free space: FAIL above 90 %,
              WARN above 60 % (free space from `my_quotas` when it runs; else reported as unknown)
"""
import argparse, collections, json, os, re, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import cfg  # noqa: E402

BYTES_PER_SEG = 1200 * 2 + 3600 * 2 + 30 * 11 * 4 + 30 + 8      # PLETH40 + II120 + vitals_hf + abp_src + time_ms


def main():
    C = cfg()
    ap = argparse.ArgumentParser(); ap.add_argument("--inv", default=os.path.join(C["intermediate_dir"], "stage_a", "inventory.jsonl"))
    ap.add_argument("--min-included", type=int, default=15500); ap.add_argument("--max-included", type=int, default=17500)
    a = ap.parse_args()
    R = [json.loads(l) for l in open(a.inv)]
    inc = [r for r in R if r.get("included")]
    res, fails, warns = {}, [], []
    ids = [r["entity_id"] for r in R]
    res["n_files"], res["n_included"] = len(R), len(inc)
    res["duplicate_ids"] = len(ids) - len(set(ids))
    res["error_frac"] = sum("error" in r for r in R) / max(1, len(R))
    if not a.min_included <= len(inc) <= a.max_included: fails.append(f"included {len(inc)} outside [{a.min_included}, {a.max_included}]")
    if res["duplicate_ids"]: fails.append("duplicate entity ids")
    if res["error_frac"] > 0.01: fails.append(f"errors {res['error_frac']:.2%}")
    chk = [r for r in inc if r.get("clock_check", {}).get("n_matches", 0) >= 3]
    cc = collections.Counter(r["clock_confidence"] for r in chk)
    res["clock_checkable"], res["clock_confidence_checkable"] = len(chk), dict(cc)
    res["clock_confidence_all"] = dict(collections.Counter(r.get("clock_confidence") for r in inc))
    if chk:
        good = (cc["verified"] + cc["corrected"]) / len(chk); bad = cc["conflict"] / len(chk)
        res["clock_good_frac"], res["clock_conflict_frac"] = good, bad
        if good < 0.95: fails.append(f"clock verified+corrected {good:.1%} < 95%")
        if bad > 0.05: fails.append(f"clock conflict {bad:.1%} > 5%")
    sides = collections.defaultdict(set)
    for r in R:
        if r.get("e1_split") in ("train", "val", "test"):
            sides[r["patient_id"]].add(r["e1_split"])
    res["e1_patients_two_sides"] = sum(len(v) > 1 for v in sides.values())
    if res["e1_patients_two_sides"]: fails.append("e1 patient on two sides")
    need_tb = sum(r.get("grid_rows", 0) for r in inc) * BYTES_PER_SEG / 1e12
    res["disk_need_tb"] = need_tb
    free_tb = None
    try:
        out = subprocess.run(["my_quotas"], capture_output=True, text=True, timeout=60).stdout
        blk = out[out.find("med250003p"):]
        q = re.search(r"Storage quota:\s*([\d.]+)TiB", blk); u = re.search(r"Storage used:\s*([\d.]+)TiB", blk)
        if q and u:
            free_tb = (float(q.group(1)) - float(u.group(1))) * 1.0995
    except Exception:
        pass
    res["disk_free_tb"] = free_tb
    if free_tb is None:
        warns.append("project free space unknown (my_quotas unavailable)")
    elif need_tb > 0.9 * free_tb:
        fails.append(f"disk need {need_tb:.2f} TB > 90% of free {free_tb:.2f} TB")
    elif need_tb > 0.6 * free_tb:
        warns.append(f"disk need {need_tb:.2f} TB > 60% of free {free_tb:.2f} TB")
    res["fails"], res["warns"] = fails, warns
    json.dump(res, open(os.path.join(os.path.dirname(a.inv), "verify_stage_a.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    print("GATE A:", "FAIL" if fails else "PASS", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
