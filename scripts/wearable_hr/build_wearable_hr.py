#!/usr/bin/env python3
"""Wearable PPG -> canonical PLETH40 + per-minute HR (var 150 HR_hf) sidecar.

Onboards three wrist/finger PPG datasets into the UNIPHY canonical format for an
HR-estimation task, matched EXACTLY to the mimic3 hf HR recipe:

  PLETH40.npy   [N_seg, 1200] f16   PPG resampled to 40 Hz, 30 s non-overlapping
  time_ms.npy   [N_seg] i64         t[i] = i*30000  (relative ms)
  ehr_events.npy                    empty (no clinical EHR in these datasets)
  ehr_hf.npy                        HR (var 150), one value per MINUTE at the
                                    pair-end seg_idx = 2k+1, value = median of the
                                    reference HR over minute [k*60, (k+1)*60) s
  meta.json

Reference HR source per dataset:
  dalia : label (ECG-derived HR @0.5 Hz) provided in S*.pkl        -> ready
  gyro  : bpmECG @ timeECG provided in Subject_*.mat               -> ready
  wesad : chest ECG @700 Hz -> R-peaks (neurokit2) -> per-beat HR  -> derived+cleaned

Run ON the data host (bedanalysis). Test with --limit 2 first.
  python build_wearable_hr.py --dataset gyro  --limit 2
  python build_wearable_hr.py --dataset dalia --limit 2
  python build_wearable_hr.py --dataset wesad --limit 2   # needs neurokit2
"""
import argparse, json, os, glob, zipfile, io
import numpy as np
from fractions import Fraction
from scipy.signal import resample_poly, butter, filtfilt, find_peaks

EHR_EVENT_DTYPE = np.dtype([('time_ms', 'int64'), ('seg_idx', 'int32'),
                            ('var_id', 'uint16'), ('value', 'float32')])
HR_VAR = 150                    # HR_hf, physio range (10, 300) — matches mimic3 hf
FS_OUT, SEG_S = 40, 30          # PLETH40, 30 s window
SPS = FS_OUT * SEG_S            # 1200 samples/segment
HR_MIN, HR_MAX = 30.0, 220.0    # physiologic clip for the reference HR

DATA_ROOT = "/labs/hulab/mxwang/data/Wav_Agent"
OUT_ROOT = "/labs/hulab/mxwang/data/Wav_Agent/processed"


# --------------------------- per-dataset loaders ---------------------------
# each returns list of (subject_id, ppg 1D float32, ppg_fs, hr_val 1D, hr_t_sec 1D)

def load_gyro(limit):
    import scipy.io as sio
    out = []
    for f in sorted(glob.glob(f"{DATA_ROOT}/Gyro-Acc-PPG/Subject_*.mat"),
                    key=lambda p: int(p.split("_")[-1].split(".")[0]))[:limit or None]:
        sid = "gyro_" + os.path.basename(f).split("_")[1].split(".")[0]
        m = sio.loadmat(f)
        ppg = m['sigPPG'][0].astype('float32')                 # ch0, 50 Hz
        hr = m['bpmECG'].ravel().astype('float32')
        ht = m['timeECG'].ravel().astype('float32')            # seconds
        out.append((sid, ppg, 50.0, hr, ht))
    return out


def load_dalia(limit):
    import pickle
    zp = f"{DATA_ROOT}/PPG-DaLiA/data.zip"
    with zipfile.ZipFile(zp) as z:
        names = sorted([n for n in z.namelist() if n.endswith('.pkl')],
                       key=lambda n: int(n.split('/S')[1].split('/')[0]))
        out = []
        for n in names[:limit or None]:
            sid = "dalia_S" + n.split('/S')[1].split('/')[0]
            d = pickle.load(io.BytesIO(z.read(n)), encoding='latin1')
            ppg = np.asarray(d['signal']['wrist']['BVP']).ravel().astype('float32')  # 64 Hz
            hr = np.asarray(d['label']).ravel().astype('float32')                    # 0.5 Hz
            ht = 4.0 + 2.0 * np.arange(len(hr))       # HR on 8 s window, 2 s shift -> center
            out.append((sid, ppg, 64.0, hr, ht))
    return out


def _hr_from_ecg(ecg, fs):
    """R-peak -> per-beat HR. neurokit2 if available, else scipy Pan-Tompkins-lite."""
    try:
        import neurokit2 as nk
        _, info = nk.ecg_peaks(nk.ecg_clean(ecg, sampling_rate=fs), sampling_rate=fs,
                               correct_artifacts=True)
        pk = np.asarray(info['ECG_R_Peaks'])
    except Exception:
        b, a = butter(3, [5 / (fs / 2), 15 / (fs / 2)], 'band')
        x = filtfilt(b, a, ecg)
        w = int(0.08 * fs)
        x = np.convolve(x ** 2, np.ones(w) / w, 'same')
        pk, _ = find_peaks(x, distance=int(0.30 * fs), height=np.mean(x) + 0.5 * np.std(x))
    rt = pk / fs
    rr = np.diff(rt)
    hr = 60.0 / rr
    ht = rt[1:]
    ok = (hr >= HR_MIN) & (hr <= HR_MAX)
    return hr[ok], ht[ok]


def load_wesad(limit):
    import pickle
    out = []
    for d0 in sorted(glob.glob(f"{DATA_ROOT}/WESAD/WESAD/S*"),
                     key=lambda p: int(os.path.basename(p)[1:]))[:limit or None]:
        sid = "wesad_" + os.path.basename(d0)
        d = pickle.load(open(f"{d0}/{os.path.basename(d0)}.pkl", 'rb'), encoding='latin1')
        ppg = np.asarray(d['signal']['wrist']['BVP']).ravel().astype('float32')     # 64 Hz
        ecg = np.asarray(d['signal']['chest']['ECG']).ravel().astype('float32')     # 700 Hz
        hr, ht = _hr_from_ecg(ecg, 700.0)
        out.append((sid, ppg, 64.0, hr, ht))
    return out


LOADERS = {"gyro": load_gyro, "dalia": load_dalia, "wesad": load_wesad}


# --------------------------- canonical transform ---------------------------
def resample_40(ppg, fs):
    fr = Fraction(FS_OUT, 1) / Fraction(fs).limit_denominator(1000)
    return resample_poly(ppg, fr.numerator, fr.denominator).astype('float32')


def build_subject(sid, ppg, fs, hr, ht, out_dir):
    p40 = resample_40(ppg, fs)
    n_seg = len(p40) // SPS
    if n_seg < 2:
        return None
    wav = np.ascontiguousarray(p40[:n_seg * SPS].reshape(n_seg, SPS), dtype=np.float16)
    time_ms = (np.arange(n_seg, dtype=np.int64) * SEG_S * 1000)

    # per-minute HR at pair-end seg_idx = 2k+1 ; value = median over [k*60,(k+1)*60) s
    hr = np.asarray(hr, float); ht = np.asarray(ht, float)
    hr = np.clip(hr, HR_MIN, HR_MAX)
    rows = []
    for k in range(n_seg // 2):
        t0, t1 = k * 60.0, (k + 1) * 60.0
        m = (ht >= t0) & (ht < t1)
        if not m.any():
            continue
        seg = 2 * k + 1
        rows.append((int(time_ms[seg]), seg, HR_VAR, float(np.median(hr[m]))))
    ev = np.array(rows, dtype=EHR_EVENT_DTYPE)

    # assertions (skill spec)
    assert wav.dtype == np.float16 and wav.flags['C_CONTIGUOUS']
    assert wav.shape[0] == len(time_ms) and np.all(np.diff(time_ms) > 0)
    assert np.all((ev['seg_idx'] >= 0) & (ev['seg_idx'] < n_seg))
    assert np.all(np.diff(ev['time_ms']) >= 0)

    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/PLETH40.npy", wav)
    np.save(f"{out_dir}/time_ms.npy", time_ms)
    np.save(f"{out_dir}/ehr_events.npy", np.empty(0, dtype=EHR_EVENT_DTYPE))
    np.save(f"{out_dir}/ehr_hf.npy", ev)
    json.dump({"patient_id": sid, "n_segments": int(n_seg),
               "segment_duration_sec": SEG_S,
               "channels": {"PLETH40": {"rate_hz": FS_OUT, "samples_per_seg": SPS}},
               "n_hr_events": int(len(ev)), "source": "wearable_ppg"},
              open(f"{out_dir}/meta.json", "w"), indent=2)
    return dict(sid=sid, n_seg=int(n_seg), n_hr=int(len(ev)),
                hr_lo=float(ev['value'].min()) if len(ev) else None,
                hr_hi=float(ev['value'].max()) if len(ev) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(LOADERS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-root", default=OUT_ROOT)
    a = ap.parse_args()
    subs = LOADERS[a.dataset](a.limit)
    out_base = f"{a.out_root}/{a.dataset}"
    print(f"[{a.dataset}] {len(subs)} subjects -> {out_base}")
    for sid, ppg, fs, hr, ht in subs:
        r = build_subject(sid, ppg, fs, hr, ht, f"{out_base}/{sid}")
        if r is None:
            print(f"  SKIP {sid} (<2 segments)"); continue
        print(f"  OK {r['sid']:14s} n_seg={r['n_seg']:5d} n_hr={r['n_hr']:4d} "
              f"HR=[{r['hr_lo']:.0f},{r['hr_hi']:.0f}]")


if __name__ == "__main__":
    main()
