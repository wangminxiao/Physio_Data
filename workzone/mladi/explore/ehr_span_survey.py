#!/usr/bin/env python3
"""MLADI: do the /ehr tables cover the waveform span? Per encounter (clock rule of
workzone/mladi/clock.py), each table's time span against the Pleth span: share of rows inside
[wave_start - 24 h, wave_end + 24 h], and hours from the waveform start of its median row. The 0c demo
found an encounter whose lab_results all lay 34-56 days before the waveform while its charted vitals
covered it. Counts how often each table is 'detached' (no row within 7 days of the waveform) while
low_rate is attached, and whether detachment goes with admission length (regDate .. dischDate).
"""
import collections, glob, json, os, random, sys
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clock  # noqa: E402

RAW = "/ocean/projects/med250003p/shared/mladi_extract_2023_waves"
TABLES = {"lab_results": "time", "low_rate": "date", "medications": "time", "infusions_and_outputs": "time",
          "diagnostic_codes": "time", "location": "beginDate", "patient": "time"}


def one(p):
    import h5py
    try:
        with h5py.File(p, "r") as f:
            if "data/waveforms/Pleth" not in f or "ehr" not in f:
                return None
            W, label = clock.parse_origin(json.loads(f.attrs[".meta"])["time_origin"])
            disch = None
            if "ehr/demographic" in f and "dischDate" in f["ehr/demographic"].dtype.names:
                x = float(f["ehr/demographic"][0]["dischDate"]); disch = x if np.isfinite(x) else None
            m = json.loads(f["data/waveforms/Pleth"].attrs[".meta"])["dwc_meta"]
            w0, w1 = clock.dwc_to_utc_ms([m["minTime"], m["maxTime"]], W)
            r = {"wave_h": float((w1 - w0) / 3.6e6), "tables": {}}
            if disch is not None and "regDate" in f["ehr/demographic"].dtype.names:
                reg = float(f["ehr/demographic"][0]["regDate"])
                if np.isfinite(reg):
                    a, b = clock.ehr_to_utc_ms([reg, disch], W, label, disch)
                    r["stay_d"] = float((b - a) / 8.64e7); r["wave_start_after_admit_d"] = float((w0 - a) / 8.64e7)
            for t, c in TABLES.items():
                k = f"ehr/{t}"
                if k not in f or c not in f[k].dtype.names:
                    continue
                tt = f[k][c][:].astype(float); tt = tt[np.isfinite(tt)]
                if not tt.size:
                    continue
                u = clock.ehr_to_utc_ms(tt, W, label, disch)
                inside = np.mean((u >= w0 - 8.64e7) & (u <= w1 + 8.64e7))
                near7 = np.any((u >= w0 - 7 * 8.64e7) & (u <= w1 + 7 * 8.64e7))
                r["tables"][t] = {"inside": float(inside), "near7": bool(near7),
                                  "med_h": float((np.median(u) - w0) / 3.6e6), "n": int(tt.size)}
            return r
    except Exception as ex:
        return {"error": f"{type(ex).__name__}: {ex}"}


def main():
    files = sorted(glob.glob(os.path.join(RAW, "*.h5")))
    random.seed(4)
    pick = random.sample(files, min(int(os.environ.get("N_ENC", 4000)), len(files)))
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 16))) as pool:
        R = [r for r in pool.map(one, pick, chunksize=4) if r]
    E = [r for r in R if "error" in r]; R = [r for r in R if "tables" in r]
    print(f"{len(R)} encounters with Pleth + ehr (errors {len(E)}: {collections.Counter(e['error'][:60] for e in E).most_common(3)})", flush=True)
    for t in TABLES:
        g = [r for r in R if t in r["tables"]]
        if not g:
            continue
        ins = np.array([r["tables"][t]["inside"] for r in g]); near = np.array([r["tables"][t]["near7"] for r in g])
        med = np.array([r["tables"][t]["med_h"] for r in g])
        print(f"  {t:22s} in {len(g):5d} enc | rows within wave+-24h: p25/p50/p75 {np.percentile(ins, [25, 50, 75]).round(2).tolist()} | "
              f"no row within 7 d of the waveform: {np.mean(~near):.1%} | median row, h from wave start p5/p50/p95 "
              f"{np.percentile(med, [5, 50, 95]).round(0).tolist()}", flush=True)
    lab_det = [r for r in R if "lab_results" in r["tables"] and "low_rate" in r["tables"]
               and not r["tables"]["lab_results"]["near7"] and r["tables"]["low_rate"]["near7"]]
    print(f"labs detached while charted vitals attached: {len(lab_det)} of "
          f"{sum('lab_results' in r['tables'] and 'low_rate' in r['tables'] for r in R)}", flush=True)
    if lab_det:
        print("  their lab median h from wave start: " + str(np.percentile([r['tables']['lab_results']['med_h'] for r in lab_det], [5, 50, 95]).round(0).tolist())
              + " | stay days p50 " + str(np.median([r.get('stay_d', np.nan) for r in lab_det])) + " | wave start after admission days p50 "
              + str(np.nanmedian([r.get('wave_start_after_admit_d', np.nan) for r in lab_det])), flush=True)
    att = [r for r in R if "stay_d" in r]
    if att:
        print(f"all: stay days p50 {np.median([r['stay_d'] for r in att]):.1f} | wave start after admission days p50/p95 "
              f"{np.percentile([r['wave_start_after_admit_d'] for r in att], [50, 95]).round(1).tolist()}", flush=True)


if __name__ == "__main__":
    main()
