# Clock-alignment fix plan — MIMIC-III and MOVER stores (with verification gates)

Status 2026-09-27 (evening): **executed**. Both chains ran on the lab node with every gate passing (MIMIC verify job 50351,
MOVER verify 50349); results are recorded at the end of this document (§6) and in `datasets/{mimic3,mover,mover_epic}/API.md`.
MC_MED needs no fix (see §4).

## 0. Findings being fixed

| Store | Defect | Direction / size | Share affected | Root cause |
|---|---|---|---|---|
| `mimic3` (5,623 entities) | `time_ms.npy`, `ehr_hf.npy` (numerics) and sepsis block times are on a *local-epoch* base; `ehr_events / ehr_baseline / ehr_recent / ehr_future / ehr_actions`, admission windows are on a *wall-clock* base | EHR events sit **4 h (EDT surrogate dates) or 5 h (EST) earlier** than the waveform | every entity | `datetime.timestamp()` on naive surrogate times (`stage3_extract_waveforms.py:545`, `stage3b_extract_numerics.py:102`, `post_sepsis_cohort.py:309`) vs pandas `.timestamp()` / `astype(int64)` on the chart side |
| `mover` (SIS, 6,994) and very likely `mover_epic` (1,820) | waveform time base vs EHR differs by **±60 min** | winter (PST) cases: charted events 60 min *later* than the waveform in 30/47; summer (PDT): 60 min *earlier* in 19/73; otherwise aligned | ≈40 % of cases | XML `…Z` timestamps are exported with a fixed UTC offset (≈70 % of devices −7 h year-round, ≈30 % −8 h), no DST; the EHR is converted Pacific→UTC with DST correctly |

Evidence (all read-only, `workzone/common/explore/`): MIMIC `time_ms[0]` − raw master-header base time = +4.00/+5.00 h
(23/40 entities matched to a header); charted HR vs ECG-derived HR best lag +220…+300 min in 27/30 entities; numerics vs
ECG 0 min (12/12), numerics vs charted +235…+300 min (12/12). MOVER: charted HR (1–2 min cadence) vs PPG pulse rate
lag bands by month (120 entities) and `OR_start(EHR) − wave_start` = 38 min (lag 0), 101 min (−60 band), −31 min (+60 band).

Convention to converge on (same as UCSF, `datasets/ucsf/ALIGNMENT.md`): **`time_ms` = wall-clock ms of the source
calendar, continuous in real elapsed time; every EHR timestamp converted to that base by one shared, tested
function; `meta.json` records `time_base` and `clock_shift_ms`.**

## 1. MIMIC-III fix

### 1.1 Decision: shift the waveform side, keep the EHR side

The chart/lab side already equals wall-clock ms (pandas semantics, identical to the UCSF convention). The waveform
side is a uniform per-entity offset (`wav_start.timestamp()` is evaluated once per entity, then `+ block start_sec`),
so the fix is an exact per-entity shift of `time_ms` — no re-extraction of 234 GB of waveforms.

```
delta_ms(entity) = − utcoffset(America/New_York, wav_start_surrogate_naive)   # = −4 h (EDT) or −5 h (EST)
time_ms_new      = time_ms + delta_ms
```

Python's `datetime.timestamp()` extrapolates today's DST rule to the 2100–2200 surrogate years; the same rule
(`zoneinfo`) reproduces the offset exactly (verified: 4.00 / 5.00 h).

### 1.2 Code changes (`workzone/mimic3/`)

| File | Change |
|---|---|
| `stage3_extract_waveforms.py` | `block_start_ms = wall_ms(wav_start) + start_sec*1000` where `wall_ms(dt) = int(dt.replace(tzinfo=timezone.utc).timestamp()*1000)`; write `meta.time_base = "wall_clock"`, `meta.clock_shift_ms = 0`, `meta.recording_start_ms` on the new base |
| `stage3b_extract_numerics.py` | same `wall_ms()` for the numerics record base time (`base_ms`) — numerics records are per record, so recompute rather than shift |
| `post_sepsis_cohort.py` | same `wall_ms()` for `block_start_ms` |
| new `workzone/mimic3/fix_clock_migrate.py` | in-place migration for the existing store: per entity read `meta.recording_start_ms`, derive the surrogate naive start, compute `delta_ms`, shift `time_ms.npy`, rewrite `meta` (`recording_start_ms`, `wave_end_pad_ms`, `admission_overlap_hours`, `time_base = "wall_clock"`, `clock_shift_ms = delta_ms`, `clock_fix_version = 1`); idempotent (skips entities already carrying `time_base`) |
| `workzone/common/clock_utils.py` (new, shared) | `wall_ms(naive_dt)`, `wall_ms_array(datetime64)`, `assert_wall_clock(meta)`; MIMIC/MOVER/MC_MED stages import it so the two conversions can never diverge again |
| `verify_mimic3_clock.py` (new gate) | the checks of §1.4 |

### 1.3 Execution order (lab node, `xhu40-b.q`; each step gated)

| Step | Job | What it regenerates | Est. time |
|---|---|---|---|
| M0 | snapshot | copy `time_ms.npy`, `meta.json`, `ehr_events.npy`, `ehr_hf.npy` of 300 random entities to `/projects/mwang80/staging/mimic3_before/` (for the end-to-end check) | 5 min |
| M1 | `fix_clock_migrate.py` | `time_ms.npy`, `meta.json` for all 5,623 entities | 10 min |
| M2 | `stage3b_extract_numerics.py --no-resume` | `ehr_hf.npy` on the new base (numerics var 150–159) | ~1 h |
| M3 | `stage3b_extract_actions.py` + `stage3b_actions_v2.py` | `ehr_actions.npy` (chart-side times unchanged; `seg_idx` recomputed against the new `time_ms`) | 30 min |
| M4 | `post_sepsis_cohort.py` | sepsis onset / block times on the new base | 20 min |
| M5 | `stage3c_ehr_trajectory.py` | `ehr_baseline / recent / events / future` (partition bounds and `seg_idx` from the new `time_ms`) | 1–2 h |
| M6 | `post_sepsis_trajectory.py`, `post_demographics.py`, `post_demographics_extra.py` | sepsis extra events, `demographics.csv` | 30 min |
| M7 | `stage4_manifest_splits.py --keep-splits` (add the flag as in UCSF) | `manifest.json` refreshed, `pretrain_splits.json` / `downstream_splits.json` membership unchanged | 20 min |
| M8 | task builders | `tasks/{lab_est_full, vital_est_full, lab6_any_min2, lab_est_per_target_min2}` (`build_estimation_task.py`), `tasks/abp_hf` (`build_abp_hf_task.py`), `tasks/sepsis`, `tasks/{cardio,gas,hgb,kidney}_traj` — cohort membership may change slightly because events now fall inside the wave window differently; report before/after counts | 30 min |
| M9 | rsync to `/projects/xhu40-cdsfm/physio_data/mimic3/` (only `time_ms.npy`, `meta.json`, `ehr_*.npy`, `demographics.csv`, `manifest.json`, splits, `tasks/`) | | 30 min |

Splits are unchanged by design (same entities, same patients); only per-entity event counts and task cohorts move.

### 1.4 Verification gates (MIMIC)

| Gate | Check | Pass criterion |
|---|---|---|
| G1 code-level, all entities | `time_ms[0]` vs the raw master header (`pXXXXXX-YYYY-MM-DD-hh-mm.hea`) base time as wall ms, for the header whose record contains the first block | difference = 0 s (±1 s) for ≥ 99 % of entities; the remainder listed with reason |
| G2 numerics ↔ waveform | ECG-derived HR (II120) vs `ehr_hf` HR (var 150), 30-min and 4-h windows, 40 entities | median best lag in [−2, 8] s, median corr ≥ 0.6 (the UCSF gate C thresholds) |
| G3 chart ↔ waveform | charted HR (var 100) vs ECG-derived HR, lag scan −8…+8 h, ≥ 100 entities with ≥ 12 charted points | median |best lag| ≤ 30 min and ≥ 90 % within ±60 min (hourly charting quantisation); **before the fix the same test gives +220…+300 min** |
| G4 cuff BP ↔ ABP | charted NIBP (var 104–106) vs `ehr_hf` ABP (153–155), lag scan −8…+8 h | peak at 0 (±15 min); the +4/+5 h peak reported in the pasted note must disappear |
| G5 structural | existing `stage4` validation: `time_ms` monotonic, `len == n_seg`; `ehr_events.seg_idx == searchsorted(time_ms, time) − 1` and in `[0, n_seg)`; partitions disjoint (`validate_partition`); `ehr_hf` times inside `[time_ms[0], time_ms[-1] + 30 s]` | 0 failures |
| G6 end-to-end vs snapshot | for the 300 snapshot entities: `time_ms_new − time_ms_old == delta_ms` exactly; multiset of `(var_id, value, time)` in `ehr_events` unchanged (only `seg_idx` differs) | 300/300 |
| G7 splits | `pretrain_splits.json` membership identical to the snapshot; task cohort deltas reported | membership identical |
| G8 downstream | re-run the MIMIC `vitalBP` config with cuff labels (UNIPHY_Plus_v2) | within-R² for ICU cuff BP moves from ≈0 to the value obtained with the manual +4/+5 h shift (≈ +0.1 in the note) |

## 2. MOVER (SIS + EPIC) fix

### 2.1 Decision: correct the waveform clock per case, measured from the data

There is no rule that recovers the device offset from metadata, so each case gets a measured shift:

```
for each entity with charted HR (var 100, ≥ 60 points inside the recording):
    lag_ppg = argmin over L ∈ {−75…+75 min, 5-min step} of MAE( charted HR(t) , PPG pulse rate in [t+L−5, t+L+5 min] )
    lag_ecg = same with ECG-derived HR (when II is usable)
    decision:
        both references available and |lag_ppg − lag_ecg| ≤ 10 min           → shift = −round_to_60(lag)      confidence "two_refs"
        one reference, MAE(best) < 0.75·MAE(0) and best ∈ {−60, +60} ± 10    → shift = −best                  confidence "one_ref"
        best within ±10 min                                                    → shift = 0                      confidence "aligned"
        otherwise                                                              → shift = 0, flag clock_unverified
time_ms_new = time_ms + shift·60 000       (shift ∈ {−60, 0, +60} min; other values are rejected and flagged)
```

Rounding to ±60 is deliberate: the defect is a whole-hour device offset; sub-hour residuals are charting latency.
Cases without vitals (rare in the OR) keep `shift = 0` and `clock_unverified = true` so downstream tasks can exclude
them. `meta.json` gets `time_base = "utc_ms"`, `clock_shift_min`, `clock_shift_method`, `clock_shift_mae_before/after`.

### 2.2 Code changes

| File | Change |
|---|---|
| new `workzone/mover/stage_b2_clock.py` (shared by `mover` and `mover_epic` via `--dataset`) | the measurement + shift above; writes `workzone/outputs/{dataset}/clock_shift.parquet` (entity, month, dst, lag_ppg, lag_ecg, shift, confidence, MAE before/after) and updates `time_ms.npy` + `meta.json`; idempotent |
| `stage_b_wave.py` (SIS + EPIC) | document that `…Z` is *not* DST-aware UTC; keep parsing as is (the correction is data-driven), add `meta.time_base = "utc_ms_uncorrected"` for fresh extractions so B2 knows it must run |
| `stage_e_assemble.py` | unchanged code; rerun (recomputes `seg_idx` and partitions) |
| `mover_combine` | rerun its task builders (entity dirs are symlinks into `mover` / `mover_epic`, so the shifted `time_ms` is picked up automatically) |
| `verify_mover_clock.py` (new gate) | §2.4 |

### 2.3 Execution order

| Step | Job | Est. time |
|---|---|---|
| V0 | snapshot `time_ms.npy`, `meta.json`, `ehr_events.npy` of 300 random SIS + 100 EPIC entities to staging | 5 min |
| V1 | `stage_b2_clock.py --dataset mover` then `--dataset mover_epic` (16 workers; PPG + ECG rate per case) | 1.5 h + 30 min |
| V2 | gate `verify_mover_clock.py` (§2.4 G1–G3) | 20 min |
| V3 | `stage_e_assemble.py --no-resume` for both stores; `stage3b_actions.py` (mover_combine actions) | 15 min |
| V4 | `stage_f_manifest.py --keep-splits`, `stage_f_demographics.py` for both; task builders (`lab_est_full`, `vital_est_full`, `lab6_any_min2`, `lab_est_per_target_min2`) for `mover`, `mover_epic`, `mover_combine` | 30 min |
| V5 | rsync `mover_combine` (and `mover`, `mover_epic` if the copies are wanted) to `/projects/xhu40-cdsfm/physio_data/` | 30 min |

### 2.4 Verification gates (MOVER)

| Gate | Check | Pass criterion |
|---|---|---|
| G1 measurement quality | `clock_shift.parquet`: share of cases with `two_refs` / `one_ref` / `aligned` / `unverified`; distribution of shifts by month | ≥ 85 % of cases with vitals decided (not `unverified`); shifts ∈ {−60, 0, +60} only |
| G2 seasonality gone | re-run the lag scan on the corrected `time_ms` (120 random SIS cases, same script as the finding) | ≥ 95 % of decided cases at lag 0 ± 10 min; DST cross-tab shows no ±60 band (before: 30/47 winter, 19/73 summer) |
| G3 independent corroboration | `OR_start(EHR) − wave_start` per case | unimodal around +40 min (p25–p75 within ±30 min of the mode); the 101-min and −31-min side lobes disappear |
| G4 attribution unchanged | per case, the waveform still lies inside the same anesthesia/OR window it was attributed to (`or_start_ms`, `or_end_ms` ± 2 h) | 100 % (cases that would change attribution are listed and excluded from the shift, flagged) |
| G5 structural | `stage_e` `validate_partition`; `ehr_events.seg_idx` consistent with the new `time_ms`; `n_seg` unchanged | 0 failures |
| G6 end-to-end vs snapshot | `time_ms_new − time_ms_old == shift`; event multiset unchanged | 400/400 |
| G7 downstream | MOVER `vitalBP` config (UNIPHY_Plus_v2) NIBP labels: within-subject error before vs after; MOVER lab_est unchanged within noise | within-R² for NIBP does not decrease; expected to increase (≈40 % of labels were 60 min off) |

### 2.5 v4 — season-constrained binary test for the cases the free scan left unverified (2026-09-27)

The device hypothesis (fixed UTC offset, no DST) fixes the **sign** of any shift by season: in standard time a −7 h
device stamps the waveform 60 min early (→ +60), in DST a −8 h device stamps it 60 min late (→ −60). Checked on the v3
decisions (`workzone/mover/explore/unverified_analysis.py`): SIS 99.2 % of the 2,366 non-zero shifts obey it (high 99.3 %,
medium 99.2 %), EPIC 98.7 % of 668 (high 100 %). Left unverified by v3: SIS 1,719 (708 without charted HR, 1,011 with;
methods none 1,180 / unclear 472 / conflict 40 / disputed 27), EPIC 374 (59 without HR). Their charted-HR series are short
(median 67 points, cases median 2 h), so the 31-lag free scan (≥ 30 points at ≥ 15 lags) mostly had no usable reference.

`stage_b2_clock.py --season-binary` replaces the free scan by a two-hypothesis test per reference (PPG pulse rate, ECG rate):
H0 "aligned" = best of lags −10…+10 min, H1 "the one hour the season allows" = best of lags ±50…70 min; per hypothesis the
MAE (≥ 15 charted points) and the Pearson r between charted HR and the rate medians. A reference votes when its
MAE-preferred hypothesis has MAE ≤ 10 bpm (PPG) / 8 bpm (ECG) and a margin — MAE < 0.8 × the other, or r ≥ 0.5 with an r
gain ≥ 0.15 — and the two criteria do not contradict. Two votes agreeing → high, one → medium, disagreement or no vote →
unverified. `meta.clock_fix_version = 2`, `clock_season_prior`, `clock_shift_v3_method` recorded; the report rows are
merged into `clock_shift.parquet` (v3 copy kept as `clock_shift_v3.parquet`).

Gates: `--validate-decided` replays the test on every v3-decided case from its pre-fix clock (`time_ms − clock_shift_min`)
and fails the chain below 98 % agreement / 60 % coverage (smoke: 40/40 agree); then `--only-unverified` re-tests only the
unverified cases (smoke: 13 of 30 decided — 11 PPG, 2 PPG+ECG; 14 undecided, 2 conflict). Cases still unverified stay in
the store and in `pretrain_splits.json` (waveform-only pretraining does not need the EHR clock) but are **excluded from
every `tasks/` list**: `build_estimation_task.py` spec key `exclude_meta = {clock_shift_confidence: [unverified]}`
(`--exclude-clock-unverified`, recorded in `build_summary.json`; the excluded ids are listed in `splits.json`).

Chain (`workzone/mover/slurm/mover_fix_clock.sbatch`): b2_validate → b2_unverified → verify_b2 → assemble → relink →
actions → manifest → tasks → verify → copy.

### 2.6 BeeGFS copies (corrected 2026-09-27)

The 2026-09-27 `copy` step did **not** update `/projects/xhu40-cdsfm/physio_data/mover_combine`: its entity dirs are
symlinks on the lab node and `rsync -rlt` cannot replace the real directories on BeeGFS with links (job ran 2 s; files
there still dated 2026-06-20). Fixed copy step: `rsync -rLt` (dereference) of the small files into `mover_combine`
(waveforms and the FM sidecars — PPG_GPT_emory/ucsf, ECG_GPT, pyPPG, ecgNK, super-beat — already live there and are
unaffected: segments did not change), and `mover/`, `mover_epic/` created as per-entity symlinks `../mover_combine/<id>`
(every SIS/EPIC entity is a combine entity; no 38 GB duplicate) plus their own `manifest.json`, `pretrain_splits.json`,
`demographics*.csv`, `tasks/`. Training configs that translate `/opt/localdata100tb/physio_data/mover*` to BeeGFS then work.

## 3. Shared guard so this cannot recur

* `workzone/common/clock_utils.py`: the only place that turns a naive datetime into epoch ms (`wall_ms`), plus
  `utc_ms(aware_dt)`; pandas/polars paths use `.dt.replace_time_zone("UTC").dt.timestamp("ms")` explicitly.
* Every store's `meta.json` carries `time_base` (`wall_clock` for MIMIC/UCSF, `utc_ms` for MOVER/MC_MED) and
  `clock_shift_ms`; `stage_f_manifest` copies it into the manifest.
* A standing gate in every pipeline: **charted HR vs ECG-derived HR lag scan** (`workzone/common/explore/tz_probe_two_refs.py`
  as the template): median |lag| ≤ 10 min on ≥ 40 entities. UCSF already has it (gate C), MC_MED passes it, MIMIC and MOVER
  will after the fix.
* `skill/SKILL.md` "Time base" paragraph (already added) tells future pipelines to do this on day one.

## 4. MC_MED — no change

Numerics (1-min, UTC ISO) vs waveform (`.hea` base_datetime as UTC): ECG-derived HR best lag 0 min in 36/38 sampled
visits (the other two have unusable ECG), all 40 sampled waveforms inside `[arrival, departure]`. The PPG-pulse-rate
reference alone produces spurious multi-hour "shifts" on ED recordings and must not be used without the ECG reference.

## 5. Order of work and approvals

1. Code + unit tests locally (clock_utils, migrate script on a synthetic entity, B2 on the 120 MOVER probe cases).
2. Smoke on the lab node: MIMIC migration on 20 entities to a scratch copy → G1/G2/G3 on those; MOVER B2 on 200 cases to
   scratch → G1–G3.
3. Submit the MIMIC chain M0→M9 and the MOVER chain V0→V5 as `afterok` chains (one job per step, gate jobs between);
   both need the user's go-ahead (real `sbatch`).
4. Re-run the ICML configs that consume the changed stores (`mimic3`, `mover`, `mover_epic`, `mover_combine`
   vitalBP / lab_est) and refresh `datasets/{mimic3,mover,mover_epic}/API.md` with the new convention and counts.

## 6. Outcome (2026-09-27)

| Store | Result |
|---|---|
| `mimic3` | 5,623 entities migrated (−4 h ×… / −5 h; exact per-entity shift), numerics/actions/sepsis/trajectories/demographics/manifest/tasks rebuilt; all gates PASS (G3 charted HR vs ECG median 0 min, G4 NIBP vs numerics +5 min, G7 partitions consistent); splits unchanged; BeeGFS copy synced (2.0 GB of changed files). Two latent bugs fixed (stage3c datetime resolution, stage4 window pad). `*_traj` tasks: two-stage builder re-implemented (`workzone/common/build_trajectory_task.py`, rule verified exactly on all four recorded cohorts) — `traj → verify → copy` jobs awaiting submission (§7). |
| `mover` (SIS) | 6,993 cases: v3 shifted −60×1,175 / +60×1,191, unchanged 3,919, unverified 1,719 (708 without charted HR); gates PASS; Stage E, manifest (splits identical), tasks rebuilt. **v4 season-binary re-test of the unverified cases + task exclusion: jobs written, awaiting submission (§7).** |
| `mover_epic` | 1,819 cases: v3 −60×357 / +60×311, unchanged 1,092, unverified 374 (59 without HR); gates PASS (EPIC envelope [−30, 90]); Stage E, manifest, tasks rebuilt; v4 re-test pending (§7). |
| `mover_combine` | 8,812 dangling symlinks re-pointed; tasks rebuilt. **BeeGFS copy was NOT refreshed by the 2026-09-27 copy job (symlink dirs, see §2.6) — fixed copy step pending (§7).** No `mover` / `mover_epic` BeeGFS copies yet (§2.6 layout pending). |
| `mcmed` | unchanged (verified aligned). |

Follow-ups: grep every pipeline for `astype("int64") // 10**6` on datetimes; fill in the v4 / trajectory numbers below once
the §7 jobs have run.

## 7. Pending jobs (2026-09-27 — Claude's `sbatch` was blocked by the permission classifier; the user submits)

All from dream node00 (mirror already holds the code; each job rsyncs mirror → lab-node clone first):

```bash
# MOVER: v4 validation (fails the chain below 98 % agreement) → unverified re-test → gates → Stage E → manifest → tasks (excl. unverified) → gates → BeeGFS copies
cd /projects/mwang80/staging/Physio_Data/workzone/mover/logs && bash ../slurm/submit_fix_clock_from.sh b2_validate
# MIMIC: rebuild cardio/gas/hgb/kidney_traj (two-stage, --keep-splits) → gates → BeeGFS copy
cd /projects/mwang80/staging/Physio_Data/workzone/mimic3/logs && bash ../slurm/submit_fix_clock_from.sh traj
```

Afterwards (training, dream b.q via `scripts/slurm_run.sh`; archive the old result folder first, e.g. `mv X X_old_preclockfix_20260927`):
`mimic3_vital_est_wav_BP`, `mimic3_vital_est_emb_BP_pcs_pcp` (data ready now), `mover_vital_est_wav_BP`, `mover_epic_vital_est_wav_BP`
(after the MOVER `copy` job), `mimic3_{cardio,gas,hgb,kidney}_traj` (after the MIMIC `traj` job). Stale but not scheduled here:
162 top-level folders in `/projects/mwang80/uniphy_v2_out` (cross_eval_*_emory_to_mimic3_*, baselines, vmamba/uniphy traj
variants) and 475 `NMI/` folders (mimic3 / mover_combine matrix, ablations, fusion) — the user decides which to re-run.
