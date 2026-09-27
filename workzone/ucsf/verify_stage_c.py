"""Verification gate after Stage C (vitals_hf). Exit 0 = PASS, 1 = FAIL. Writes {intermediate_dir}/verify_stage_c.json.

Checks: summary present; ok >= 95 % of processed, no_meta == 0; on 300 random ok entities: vitals_hf float32
C-contiguous (n_seg,15,8), abp_src uint8, nbp_events dtype/sorted/in-range, HR valid fraction median >= 0.8
among entities with an HR file, class shares reported, >= 90 % of class-C non-NBP files re-anchored, NBP
events per hour median in [0.3, 8]; physiological alignment on 40 random entities with II: ECG-derived HR vs
vitals_hf HR over 30 min from the middle -> median corr >= 0.6 and median best lag in [-2, 8] s, reported
separately for header-time (A) and zero-time (B/C) files.
"""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np, polars as pl, yaml
from scipy.signal import butter, filtfilt, find_peaks

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from physio_data.schema import EHR_EVENT_DTYPE  # noqa: E402
sys.path.insert(0, str(REPO_ROOT / "workzone" / "ucsf"))
from clock import utc_offset_ms, ge_wall_to_grid_ms  # noqa: E402
CONFIG_PATH = REPO_ROOT / "workzone" / "configs" / "server_paths.yaml"


def ecg_hr(ii, fs, n_slots, slot_s=2.0):
    x = np.where(np.isfinite(ii), ii, np.nanmean(ii))
    if not np.isfinite(x).all() or x.std() == 0: return np.full(n_slots, np.nan)
    b, a = butter(3, [5 / (fs / 2), 25 / (fs / 2)], btype="band"); y = filtfilt(b, a, x)
    e = np.convolve(y ** 2, np.ones(int(0.12 * fs)) / int(0.12 * fs), mode="same"); pk, _ = find_peaks(e, distance=int(0.3 * fs))
    if pk.size < 20: return np.full(n_slots, np.nan)
    h = e[pk]; tb = pk[h >= 0.3 * np.percentile(h, 90)] / fs; out = np.full(n_slots, np.nan)
    for k in range(n_slots):
        t = (k + 0.5) * slot_s; m = (tb >= t - 5) & (tb <= t + 5)
        if m.sum() >= 4: out[k] = 60.0 * (m.sum() - 1) / (tb[m][-1] - tb[m][0])
    return out


def best_lag(ref, num, max_slots=8):
    best = None
    for lag in range(-max_slots, max_slots + 1):
        a = ref[max(0, -lag):len(ref) - max(0, lag)]; b = num[max(0, lag):len(num) - max(0, -lag)]; ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 100 or a[ok].std() == 0 or b[ok].std() == 0: continue
        c = float(np.corrcoef(a[ok], b[ok])[0, 1])
        if best is None or c > best[1]: best = (lag * 2, c)
    return best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default=str(CONFIG_PATH)); ap.add_argument("--dataset", default="ucsf_all")
    ap.add_argument("--sample", type=int, default=300); ap.add_argument("--physio-sample", type=int, default=40)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())[args.dataset]
    inter = Path(cfg["intermediate_dir"]); out_dir = Path(cfg["output_dir"])
    checks = []; info = {}
    def check(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail}); print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    summ_p = inter / "stage_c_hf_summary.json"; st_p = inter / "stage_c_hf_status.parquet"
    check("summary present", summ_p.exists()); check("status present", st_p.exists())
    if not (summ_p.exists() and st_p.exists()):
        return finish(inter, checks, info)
    summ = json.loads(summ_p.read_text()); st = pl.read_parquet(st_p); by = summ["by_status"]; info["by_status"] = by
    processed = sum(by.values()); ok_n = by.get("ok", 0)
    check("ok >= 95 % of processed", ok_n >= 0.95 * max(processed, 1), f"ok={ok_n} processed={processed}")
    check("no_meta == 0", by.get("no_meta", 0) == 0, f"no_meta={by.get('no_meta', 0)}")
    ok_ids = st.filter(pl.col("status") == "ok")["entity_id"].to_list(); random.seed(2); sample = random.sample(ok_ids, min(args.sample, len(ok_ids)))
    fails = {}; hr_valid = []; cls = {"A": 0, "B": 0, "C": 0}; c_total = c_re = 0; nbp_rate = []; degenerate_non_nbp = 0
    for eid in sample:
        d = out_dir / eid; errs = []
        try:
            m = json.loads((d / "meta.json").read_text()); v = m["vitals_hf"]; n_seg = m["n_seg"]
            hf = np.load(d / "vitals_hf.npy", mmap_mode="r"); src = np.load(d / "vitals_hf_abp_src.npy", mmap_mode="r"); ev = np.load(d / "nbp_events.npy")
            if hf.dtype != np.float32 or hf.shape != (n_seg, 15, len(v["var_ids"])) or not hf.flags["C_CONTIGUOUS"]: errs.append("vitals_hf shape/dtype")
            if src.dtype != np.uint8 or src.shape != (n_seg, 15): errs.append("abp_src")
            if ev.dtype != EHR_EVENT_DTYPE: errs.append("nbp dtype")
            elif ev.size and (not np.all(np.diff(ev["time_ms"]) >= 0) or ev["seg_idx"].min() < 0 or ev["seg_idx"].max() >= n_seg): errs.append("nbp order/range")
            files = v["files"]
            if "HR" in files or any(k.startswith("HR#") for k in files): hr_valid.append(v["valid_frac_per_var"]["HR_hf"])
            for s, fi in files.items():
                cls[fi["class"]] = cls.get(fi["class"], 0) + 1
                if fi["class"] == "C" and not s.startswith("NBP"):
                    c_total += 1; c_re += "abs_reanchored" in fi["policy"]
                    if "degenerate" in fi["policy"]: degenerate_non_nbp += 1
            hours = n_seg * 30 / 3600
            if any(s.startswith("NBP-S") for s in files) and hours > 0: nbp_rate.append((ev["var_id"] == 157).sum() / hours)
        except Exception as e:
            errs.append(f"{type(e).__name__}: {str(e)[:80]}")
        if errs: fails[eid] = errs
    check(f"sampled entities structurally valid ({len(sample)})", not fails, f"failing={len(fails)} e.g. {list(fails.values())[:3]}")
    check("HR valid fraction median >= 0.8 (entities with HR file)", (np.median(hr_valid) >= 0.8) if hr_valid else False, f"median={np.median(hr_valid) if hr_valid else None:.3f} n={len(hr_valid)}")
    tot_cls = sum(cls.values()) or 1; info["class_shares"] = {k: round(v / tot_cls, 3) for k, v in cls.items()}
    check("class-C non-NBP files re-anchored >= 90 %", (c_re >= 0.9 * c_total) if c_total else True, f"{c_re}/{c_total} (degenerate non-NBP: {degenerate_non_nbp})")
    check("NBP events per hour median in [0.3, 8]", (0.3 <= np.median(nbp_rate) <= 8) if nbp_rate else True, f"median={np.median(nbp_rate) if nbp_rate else None:.2f}/h")
    # physiological alignment
    random.seed(3); phys = random.sample(ok_ids, min(args.physio_sample, len(ok_ids))); res = {"A": [], "BC": []}
    for eid in phys:
        d = out_dir / eid
        try:
            m = json.loads((d / "meta.json").read_text()); v = m["vitals_hf"]; n_seg = m["n_seg"]
            if n_seg < 90: continue
            hr_cls = next((fi["class"] for s, fi in v["files"].items() if s == "HR" or s.startswith("HR#")), None)
            if hr_cls is None: continue
            # class A: 30 min at the midpoint (as before); zero-time B/C files are anchored to the first .adibin, so
            # judge them on a 4-h window (robust correlation, catches drift) with a +-80 s lag search
            half = 30 if hr_cls == "A" else 240
            s0 = max(0, n_seg // 2 - half); nsg = min(2 * half, n_seg - s0)
            ii = np.asarray(np.load(d / "II120.npy", mmap_mode="r")[s0:s0 + nsg], dtype=np.float64).reshape(-1)
            if np.isnan(ii).mean() > 0.2: continue
            num = np.load(d / "vitals_hf.npy", mmap_mode="r")[s0:s0 + nsg, :, 0].reshape(-1).astype(np.float64)
            if np.isfinite(num).mean() < 0.5: continue
            b = best_lag(ecg_hr(ii, 120.0, nsg * 15), num, max_slots=8 if hr_cls == "A" else 40)
            if b: res["A" if hr_cls == "A" else "BC"].append(b)
        except Exception:
            continue
    allr = res["A"] + res["BC"]
    if allr:
        corr = np.median([c for _, c in allr]); lag = np.median([l for l, _ in allr])
        check("ECG-derived HR vs vitals_hf HR: median corr >= 0.6", corr >= 0.6, f"median corr={corr:.3f} n={len(allr)}")
        check("ECG-derived HR vs vitals_hf HR: median best lag in [-2, 8] s", -2 <= lag <= 8, f"median lag={lag:.0f} s")
        info["physio"] = {k: {"n": len(rs), "corr_median": round(float(np.median([c for _, c in rs])), 3) if rs else None, "lag_median_s": round(float(np.median([l for l, _ in rs])), 1) if rs else None} for k, rs in res.items()}
        if res["BC"]:
            bc_corr = float(np.median([c for _, c in res["BC"]])); bc_lag = float(np.median([abs(l) for l, _ in res["BC"]]))
            check("zero-time (B/C) entities: median corr >= 0.5 over 4 h", bc_corr >= 0.5, f"corr={bc_corr:.3f} |lag| median={bc_lag:.0f} s n={len(res['BC'])}")
    else:
        check("physiological alignment sample available", False, "no usable entity")
    # alignment AFTER a DST switch inside the cycle (the bug fixed by the UTC-continuous grid, Stage B v2 / C v2)
    strad = []
    for eid in ok_ids:
        try:
            m = json.loads((out_dir / eid / "meta.json").read_text())
        except Exception:
            continue
        if m.get("dst_switch_in_cycle") and m.get("time_base") == "utc_continuous":
            strad.append((eid, m))
        if len(strad) >= 200:
            break
    random.seed(4); random.shuffle(strad); after = []
    for eid, m in strad[:60]:
        try:
            d = out_dir / eid; ep_start = int(m["episode_start_ms"]); ep_end_wall = int(m.get("episode_end_wall_ms", m["episode_end_ms"])); n_seg = m["n_seg"]
            t = (ep_start // 3600000) * 3600000; sw = None
            while t < ep_end_wall:
                if utc_offset_ms(t) != utc_offset_ms(t + 3600000): sw = t + 3600000; break
                t += 3600000
            if sw is None: continue
            g = int(ge_wall_to_grid_ms(sw + 3600000, ep_start)); k0 = int((g - ep_start) // 30000) + 120; nsg = 60
            if k0 + nsg > n_seg: continue
            ii = np.asarray(np.load(d / "II120.npy", mmap_mode="r")[k0:k0 + nsg], dtype=np.float64).reshape(-1)
            num = np.load(d / "vitals_hf.npy", mmap_mode="r")[k0:k0 + nsg, :, 0].reshape(-1).astype(np.float64)
            if np.isnan(ii).mean() > 0.2 or np.isfinite(num).mean() < 0.5: continue
            b = best_lag(ecg_hr(ii, 120.0, nsg * 15), num, max_slots=40)
            if b: after.append(b)
        except Exception:
            continue
    if len(after) >= 3:
        lag_a = np.median([abs(l) for l, _ in after]); corr_a = np.median([c for _, c in after])
        check("DST-straddling cycles: ECG vs vitals_hf |lag| after the switch <= 8 s (median)", lag_a <= 8, f"median |lag|={lag_a:.0f} s corr={corr_a:.2f} n={len(after)} (of {len(strad)} straddling)")
        info["dst_switch_after"] = {"n": len(after), "lag_abs_median_s": float(lag_a), "corr_median": float(corr_a), "n_straddling_seen": len(strad)}
    else:
        check("DST-straddling cycles: alignment sample", len(strad) == 0, f"straddling={len(strad)} usable={len(after)} (pass only when none exist)")
    info.update(n_ok=ok_n, n_processed=processed, sample=len(sample), hr_valid_median=round(float(np.median(hr_valid)), 4) if hr_valid else None,
                n_nbp_events_total=summ.get("n_nbp_events_total"), zero_anchor=summ.get("zero_anchor"), abs_tail=summ.get("abs_tail"))
    return finish(inter, checks, info)


def finish(inter, checks, info):
    ok = all(c["pass"] for c in checks)
    (inter / "verify_stage_c.json").write_text(json.dumps({"stage": "c", "pass": ok, "ran_at_unix": int(time.time()), "checks": checks, "info": info}, indent=2))
    print(json.dumps(info, indent=2)); print("VERIFY STAGE C:", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
