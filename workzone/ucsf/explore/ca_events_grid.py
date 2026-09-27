"""
Cardiac-arrest (Code Blue, TypeCode CPA) event times on the ucsf_all entity grid.

For every CA patient: CodeTime (MATLAB datenum, TRUE local time) -> grid ms of the wave cycle that contains the
event (clock.real_wall_to_grid_ms with that cycle's origin), or, when no cycle contains it, the nearest cycle's
grid.  Writes {out}: {patient_id_ge: {"event_grid_ms", "entity", "inside_cycle", "dst_switch_in_cycle"}} and a
flat {pid: ms} file for ca_t0_detect.py --events-json.

Run on the lab node with the physio_data env after the DST fix (Stage B v2) has been applied:
  python workzone/ucsf/explore/ca_events_grid.py --out /projects/mwang80/staging/ca_events_grid.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import polars as pl, yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "workzone" / "ucsf"))
from clock import real_wall_to_grid_ms, matlab_datenum_to_wall_ms  # noqa: E402
sys.path.insert(0, str(REPO_ROOT / "workzone" / "ucsf"))
from stage_f_ca import load_codeblue  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "workzone" / "configs" / "server_paths.yaml"))
    ap.add_argument("--dataset", default="ucsf_all")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())[args.dataset]
    root = Path(cfg["output_dir"])
    codeblue = load_codeblue(Path(cfg["codeblue_parquet"]), Path(cfg["offset_table_parquet"]))
    off = pl.read_parquet(cfg["offset_table_parquet"]).select([
        pl.col("Patient_ID_GE").cast(pl.Utf8).str.strip_chars().str.replace(r"^DE", "").alias("pid"),
        pl.col("offset_GE").cast(pl.Float64).alias("og")]).unique("pid", keep="first")
    og = dict(zip(off["pid"].to_list(), off["og"].to_list()))
    man = json.loads((root / "manifest.json").read_text())
    by_pat: dict[str, list[dict]] = {}
    for q in man:
        by_pat.setdefault(str(q["patient_id_ge"]), []).append(q)
    out, flat, stats = {}, {}, {"inside": 0, "nearest": 0, "no_cycle": 0, "no_offset": 0}
    for pid, code_wall in codeblue.items():
        cyc = by_pat.get(pid, [])
        if not cyc or og.get(pid) is None:
            stats["no_cycle" if not cyc else "no_offset"] += 1
            continue
        cands = []
        for q in cyc:
            g = int(real_wall_to_grid_ms(code_wall, og[pid], int(q["wave_start_ms"])))
            inside = q["wave_start_ms"] <= g <= q["wave_end_ms"] + 30000
            dist = 0 if inside else min(abs(g - q["wave_start_ms"]), abs(g - q["wave_end_ms"]))
            cands.append((0 if inside else 1, dist, q, g, inside))
        cands.sort(key=lambda c: (c[0], c[1]))
        _, _, q, g, inside = cands[0]
        stats["inside" if inside else "nearest"] += 1
        meta = {}
        try:
            meta = json.loads((root / q["entity_id"] / "meta.json").read_text())
        except Exception:
            pass
        out[pid] = {"event_grid_ms": g, "entity": q["entity_id"], "inside_cycle": bool(inside),
                    "dst_switch_in_cycle": meta.get("dst_switch_in_cycle"), "time_base": meta.get("time_base")}
        flat[pid] = g
    Path(args.out).write_text(json.dumps(out, indent=1))
    Path(args.out).with_suffix(".flat.json").write_text(json.dumps(flat))
    print(f"CA patients with Code Blue: {len(codeblue)}; {stats}; wrote {args.out} (+ .flat.json)")


if __name__ == "__main__":
    main()
