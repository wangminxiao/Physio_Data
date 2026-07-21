#!/usr/bin/env python3
"""Stage 3b (numerics): high-frequency monitor vitals -> non-destructive sidecar.

Alignment (per user spec): the WAVEFORM segment grid drives the sample times.
mimic3 segments are 30 s non-overlapping, so every 2 segments = 60 s = one
1/min tick. For each 2-segment pair (2k, 2k+1) we take the vital value at the
PAIR-END time  t_q = time_ms[2k+1] + 30 s  and place it at seg_idx = 2k+1.
The value = the numerics reading nearest t_q (within TOL); with the monitor at
1/min or 1/sec the nearest reading is <=30 s away, so every pair-end that has
monitor data gets a label. Written to per-entity sidecar `ehr_hf.npy`.

Source : <subject>/*n.hea|*n.dat   (WFDB format-16, invalid sentinel -32768)
Reads  : entity time_ms.npy + meta.json['source_path','segment_duration_sec']
Writes : entity ehr_hf.npy  (same EHR_EVENT_DTYPE)  -- canonical files UNTOUCHED.

High-freq var_ids (distinct from CHARTEVENTS vitals 100/101/102/110/111/112):
  150 HR_hf  151 SpO2_hf  152 RR_hf  153 ABPs_hf  154 ABPd_hf  155 ABPm_hf
Add these to var_registry.json before consuming.

Consumer hook (NOT done here): PatientStore.ehr_hf() + merge into events at the
densify call site in dataset.py when target_var_ids intersect 150..155.
"""
import os, json, glob, argparse
from datetime import datetime
import numpy as np
import multiprocessing as mp

EHR_EVENT_DTYPE = np.dtype([('time_ms', 'int64'), ('seg_idx', 'int32'),
                            ('var_id', 'uint16'), ('value', 'float32')])

VARSPEC = {150: (10, 300), 151: (20, 100), 152: (1, 70),      # id -> (physio_min, max)
           153: (40, 300), 154: (20, 200), 155: (30, 250)}
MISSING = -32768                 # WFDB format-16 invalid sample
TOL_MS  = 30_000                 # a monitor reading counts if within +/-30 s of the pair-end


def chan_to_var(name: str):
    n = name.upper().lstrip('%').replace('-', '').replace('_', '')
    if n == 'HR':                    return 150
    if n == 'SPO2':                  return 151     # main SpO2 only (not L/R/AP/dSpO2)
    if n == 'RESP':                  return 152     # numerics RESPIRATION RATE
    if n in ('ABPSYS', 'ARTSYS'):    return 153     # ART == invasive arterial alias
    if n in ('ABPDIAS', 'ARTDIAS'):  return 154
    if n in ('ABPMEAN', 'ARTMEAN'):  return 155
    return None


def parse_num_header(hea):
    """(fs_hz, base_ms, [(dat_name, gain, adc_zero, var_id)])  or None."""
    try:
        lines = open(hea).read().replace('\r', '').splitlines()
    except OSError:
        return None
    if not lines:
        return None
    h = lines[0].split()
    if len(h) < 6:
        return None
    try:
        fs = float(h[2].split('/')[0])
    except ValueError:
        return None
    dt = None
    for fmt in ("%H:%M:%S.%f %d/%m/%Y", "%H:%M:%S %d/%m/%Y"):
        try:
            dt = datetime.strptime(f"{h[4]} {h[5]}", fmt); break
        except ValueError:
            continue
    if dt is None:
        return None
    base_ms = int(dt.timestamp() * 1000)             # MATCH stage3 (.timestamp())
    sigs = []
    for l in lines[1:]:
        if not l or l.startswith('#'):
            continue
        f = l.split()
        if not f or not f[0].endswith('.dat'):
            continue
        vid = chan_to_var(''.join(f[8:]))
        g = f[2].split('/')[0]
        if '(' in g:
            gain = float(g.split('(')[0]); zero = int(g.split('(')[1].rstrip(')'))
        else:
            gain = float(g); zero = int(f[4])
        sigs.append((f[0], gain, zero, vid))
    return fs, base_ms, sigs


def decode_record(hea):
    """Yield (var_id, times_ms, values) for target channels of one numerics record."""
    parsed = parse_num_header(hea)
    if parsed is None:
        return
    fs, base_ms, sigs = parsed
    if not sigs or all(v is None for *_, v in sigs):
        return
    nsig = len(sigs)
    dat = os.path.join(os.path.dirname(hea), sigs[0][0])     # numerics = single .dat
    try:
        raw = np.fromfile(dat, dtype='<i2')
    except OSError:
        return
    N = raw.size // nsig
    if N == 0:
        return
    raw = raw[:N * nsig].reshape(N, nsig)
    times = (base_ms + (np.arange(N) * (1000.0 / fs))).astype(np.int64)
    for j, (_, gain, zero, vid) in enumerate(sigs):
        if vid is None:
            continue
        col = raw[:, j].astype(np.float64)
        val = np.where(col == MISSING, np.nan, (col - zero) / gain)
        yield vid, times, val


def nearest_within(sample_t, sample_v, query_t, tol_ms):
    """Value of the sample nearest each query time, else NaN. sample_t sorted asc."""
    out = np.full(len(query_t), np.nan)
    if len(sample_t) == 0:
        return out
    pos = np.searchsorted(sample_t, query_t)
    lo = np.clip(pos - 1, 0, len(sample_t) - 1)
    hi = np.clip(pos, 0, len(sample_t) - 1)
    dl = np.abs(query_t - sample_t[lo])
    dh = np.abs(sample_t[hi] - query_t)
    use_lo = dl <= dh
    nn = np.where(use_lo, lo, hi)
    dist = np.where(use_lo, dl, dh)
    v = sample_v[nn]
    return np.where(dist <= tol_ms, v, np.nan)


def process_entity(args):
    pid, root, write = args
    edir = os.path.join(root, pid)
    tpath, mpath = os.path.join(edir, 'time_ms.npy'), os.path.join(edir, 'meta.json')
    if not (os.path.exists(tpath) and os.path.exists(mpath)):
        return (pid, 'skip_nodir', 0, None)
    time_ms = np.load(tpath)
    n_seg = len(time_ms)
    meta = json.load(open(mpath))
    seg_dur_ms = int(meta.get('segment_duration_sec', 30)) * 1000
    subj_dir = meta.get('source_path')
    if not subj_dir or not os.path.isdir(subj_dir):
        return (pid, 'skip_nosrc', 0, None)

    # query grid = end time of every 2-segment pair, placed at the pair's 2nd seg
    q_seg = np.arange(1, n_seg, 2, dtype=np.int64)
    if len(q_seg) == 0:
        if write: np.save(os.path.join(edir, 'ehr_hf.npy'), np.empty(0, EHR_EVENT_DTYPE))
        return (pid, 'ok', 0, None)
    q_t = time_ms[q_seg] + seg_dur_ms

    # collect all valid, in-range samples per var across the subject's numerics records
    per_var = {vid: ([], []) for vid in VARSPEC}
    for hea in sorted(glob.glob(os.path.join(subj_dir, '*n.hea'))):
        for vid, times, vals in decode_record(hea):
            lo, hi = VARSPEC[vid]
            m = ~np.isnan(vals) & (vals >= lo) & (vals <= hi)
            if m.any():
                per_var[vid][0].append(times[m]); per_var[vid][1].append(vals[m])

    rows = []
    sample_diag = None
    for vid, (tl, vl) in per_var.items():
        if not tl:
            continue
        st = np.concatenate(tl); sv = np.concatenate(vl)
        o = np.argsort(st, kind='stable'); st, sv = st[o], sv[o]
        st, ui = np.unique(st, return_index=True); sv = sv[ui]   # dedup identical stamps
        qv = nearest_within(st, sv, q_t, TOL_MS)
        good = ~np.isnan(qv)
        for s, v in zip(q_seg[good], qv[good]):
            rows.append((int(time_ms[s] + seg_dur_ms), int(s), int(vid), float(v)))
        if sample_diag is None and good.any():           # diag: 1st matched pair
            k = int(np.argmax(good))
            sample_diag = (vid, int(q_seg[k]), int(q_t[k]), float(qv[k]))

    arr = np.array(rows, dtype=EHR_EVENT_DTYPE) if rows else np.empty(0, EHR_EVENT_DTYPE)
    arr.sort(order='time_ms')
    if write:
        np.save(os.path.join(edir, 'ehr_hf.npy'), arr)
        meta['n_hf_events'] = int(len(arr))
        meta['hf_vars'] = sorted({int(r[2]) for r in rows})
        json.dump(meta, open(mpath, 'w'), indent=2)
    return (pid, 'ok', len(arr), (n_seg, len(q_seg), sample_diag))


def list_entities(root, split_file, entity_ids, limit):
    if entity_ids:
        ids = entity_ids
    elif split_file:
        d = json.load(open(split_file))
        ids = sorted({e for k, v in d.items() if isinstance(v, list) for e in v})
    else:
        ids = sorted(n for n in os.listdir(root)
                     if '_' in n and os.path.isdir(os.path.join(root, n)))
    return ids[:limit] if limit else ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--split-file', default=None, help='restrict to a task splits.json cohort')
    ap.add_argument('--entity-ids', nargs='+', default=None)
    ap.add_argument('--limit', type=int, default=0, help='first N only (TEST FIRST)')
    ap.add_argument('--workers', type=int, default=min(24, max(1, (os.cpu_count() or 2) // 2)))
    ap.add_argument('--dry-run', action='store_true', help='compute + print, write nothing')
    args = ap.parse_args()

    ents = list_entities(args.root, args.split_file, args.entity_ids, args.limit)
    write = not args.dry_run
    print(f"[stage3b-numerics] {len(ents)} entities  workers={args.workers}  "
          f"{'DRY-RUN (no write)' if args.dry_run else 'WRITE ehr_hf.npy'}", flush=True)
    tasks = [(pid, args.root, write) for pid in ents]

    ok = skip = 0; total = 0; hist = {}
    with mp.Pool(args.workers) as pool:
        for i, (pid, status, n, diag) in enumerate(pool.imap_unordered(process_entity, tasks, chunksize=8), 1):
            hist[status] = hist.get(status, 0) + 1
            if status == 'ok':
                ok += 1; total += n
                if args.dry_run and diag and diag[2] and i <= 8:
                    ns, nq, sd = diag
                    print(f"  {pid}: n_seg={ns} pairs={nq} hf_events={n}  "
                          f"sample[var{sd[0]} seg{sd[1]} t={sd[2]} val={sd[3]:.1f}]", flush=True)
            else:
                skip += 1
            if not args.dry_run and i % 500 == 0:
                print(f"  {i}/{len(ents)} ok={ok} skip={skip} events={total}", flush=True)
    print(f"[done] ok={ok} skip={skip} total_hf_events={total}  avg={total/max(ok,1):.0f}/entity  "
          f"status={hist}", flush=True)


if __name__ == '__main__':
    main()
