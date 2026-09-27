# UCSF — aligning waveform, monitor vitals and EHR on one clock

How every timestamp in the UCSF data relates to real time, why the naive rule was wrong, what the
entity time grid now means, and the exact conversions the pipelines use. Verified on data on
2026-09-26 (evidence in §4); the code lives in `workzone/ucsf/clock.py` and is the only place that
converts times.

## 1. Summary

| Source | File / table | Shift (per encounter) | Shift arithmetic | Wall clock it shows |
|---|---|---|---|---|
| `.adibin` waveform header | `DE{pid}_{ts14}_{uid}.adibin` (Y/M/D/h/m/s fields) | `offset_GE` days | **absolute**: `to_LA(UTC(real) − offset_GE d)` | GE calendar, DST of the *shifted* date |
| `.vital` numeric header | `…_{suffix}.vital` (Y/M/D/h/m/s; zero for class B/C) | `offset_GE` days | absolute (same as adibin) | GE calendar |
| MRN-Mapping / Alarms / ValidWaveTime ADT columns | `BedTransfer_In/Out`, `WaveStartTime`, `ValidStartTime`… | `offset_GE` days | **naive wall clock**: `real − offset_GE d` | GE calendar, DST of the *real* date |
| EHR tables | `Filtered_Lab_New`, `FLOWSHEETVALUEFACT`, encounters, orders… | `offset` days | naive wall clock: `real − offset d` | EHR calendar, DST of the real date |
| Code Blue list | `SAUCSFCodeBlue_FirstEvent_2013_2018_final.xlsx` `CodeTime` (MATLAB datenum) | none | — | real local time (not de-identified) |
| ValidWaveTime `EventTime` | CA-cohort CSV | `offset_GE` days **plus a spurious local→UTC conversion** | naive + 7/8 h | do not use |

`offset_GE − offset = 12 days` for all 27,903 encounters in the offset table.

Consequences:

* The naive rule `T_ge = T_ehr − 12 d` is exact only when the real date and the GE-shifted date have
  the same DST state; otherwise it is off by ±60 min. **45 % of encounters (12,605 / 27,903)** are in
  the mismatched case.
* Inside one wave cycle the monitor streams are continuous in real elapsed time, but every `.adibin`
  chunk carries its own wall-clock header. When a DST switch on the GE calendar falls inside a cycle,
  placing chunks by wall clock opens a 1-h gap (spring) or overlaps 1 h (fall, the later chunk
  overwrote the earlier one) and desynchronises the `.vital` streams from the waveform by ±60 min.
  663 of the 38,778 Stage-B cycles (610 of the 37,985 valid `ucsf_all` entities) straddle such a switch.
* The `ValidWaveTime` `EventTime` column carries both errors (naive shift and a +7/+8 h UTC
  conversion); Code Blue times must come from the xlsx.

## 2. Definitions

* **wall ms** — a naive local wall-clock datetime encoded as milliseconds since 1970-01-01 00:00 *as
  if it were UTC*. This is the encoding of `.adibin`/`.vital` headers, of `time_ms.npy`, and of every
  EHR timestamp in this repo. **utc ms** — true epoch milliseconds.
* **real calendar** — the patient's real local time (America/Los_Angeles).
* **GE calendar** — real time shifted by `offset_GE` days (monitor side). **EHR calendar** — real time
  shifted by `offset` days (EHR side). Both shifts are per encounter; the offset table
  (`encounter_date_offset_table_ver_Apr2024.xlsx`, converted to
  `workzone/outputs/ucsf_all/encounter_offset_table.parquet`) links `Encounter_ID` ↔
  `(Patient_ID_GE, Wynton_folder)` ↔ `offset`, `offset_GE`.
* `UTC(wall)` — interpret a wall clock as America/Los_Angeles local time and take the epoch instant
  (DST state of that wall date); `to_LA(utc)` — the inverse. The fall-back ambiguous hour resolves
  to its first occurrence (PDT).

## 3. The entity time grid (Stage B v2, `time_base = "utc_continuous"`)

For every entity (= one wave cycle) `time_ms[0]` is the wall clock of the cycle origin on the GE
calendar (`episode_start_ms`: the first `.adibin` header in `ucsf_all`; the ValidWaveTime
`ValidStartTime` in the older `ucsf` store), and every later grid time is

```
grid(t) = episode_start_ms + ( UTC(t) − UTC(episode_start_ms) )        # t = any GE-calendar wall clock
```

i.e. the origin's wall clock plus **real elapsed milliseconds**. Within a cycle the grid never jumps.
Each `.adibin` chunk and each class-A `.vital` file is placed at `grid(header)`; zero-time `.vital`
files (class B/C) are anchored to the origin as before. `meta.json` records

| field | meaning |
|---|---|
| `time_base` | `"utc_continuous"` (Stage B v2) — legacy stores say `"wall_clock"` in the manifest |
| `grid_utc_offset_min` | LA UTC offset at the origin (−420 PDT / −480 PST) |
| `dst_switch_in_cycle` | a DST switch on the GE calendar lies between origin and cycle end |
| `episode_end_wall_ms` | the last chunk's end as wall clock (differs from `episode_end_ms` by ±60 min when `dst_switch_in_cycle`) |
| `stage_b_version` | 2 |
| `vitals_hf.time_base`, `vitals.time_base` | `"utc_continuous"` |

The grid is a *de-identified* clock: `grid_to_real_wall_ms` (re-identifying) exists only for
validation and is never written to a store.

## 4. Evidence

All probes are in `workzone/ucsf/explore/clock_*.py`; results are aggregates only.

| Test | Sample | Result |
|---|---|---|
| Charted HR (flowsheet key 38524 PULSE) vs monitor HR (`vitals_hf`), lags −60/0/+60 | 597 cycles with a clear winner (EHR months 2015-03, 2015-11) | 595 at the lag predicted by "EHR naive / GE absolute"; the "both absolute" hypothesis fails (its ±60 cells sit at 0) |
| Charted SpO2 (key 2) vs monitor SpO2 | 164 | 164 / 164 |
| Charted NBP `sys/dia` (key 32710) vs cuff readings in `nbp_events.npy` (both within 2 mmHg, ±10 min) | 901 cycles, 4 EHR months | 884 (98 %) at the predicted lag; class-A `.vital` cycles 736 / 736 |
| ADT `bed_transferin_time` − first `.adibin` header (offset table) | 17,837 encounters | median −3 / −2 min when DST states agree; **−63 min** for (real PST, GE PDT); **+58 min** for (real PDT, GE PST); 84–88 % within ±15 min of those values |
| Code Blue times vs ECG collapse (t0′ detector B marker) | 111 events with a collapse | with the absolute rule the collapse sits within ±15 min of the code time in 67 events (38 with the naive rule); the ±60 min side peaks vanish |
| Lag across a DST switch **inside** a cycle (flowsheet HR vs `vitals_hf`) | 8 straddling cycles | identical before and after the GE-calendar switch → the `.vital` stream is continuous |
| ECG-derived HR vs `vitals_hf` HR after an in-cycle switch (legacy grid) | 14 straddling cycles | ±60 min in 6 → chunks placed by wall clock, `.vital` continuous |
| `.adibin` chunk boundaries at the switch (legacy grid) | 610 straddling cycles | single chunk spans the switch 433, continuous 60, **+60 min gap 43** (spring), **−60 min overlap 47** (fall), other 27 |

Residual anomaly: 17 class-B/C cycles from the `2016-01-deid` / `2016-02-deid` batches measure a
−60 min lag where the model predicts +60 (2 % of probed cycles; cause unknown). When an exact
anchor matters, measure the lag for that cycle (charted HR/NBP vs monitor, `clock_ca_lag_v3.py`).

## 5. Conversions (`workzone/ucsf/clock.py`)

```python
from clock import (ge_wall_to_grid_ms, real_wall_to_grid_ms, ehr_wall_to_grid_ms,
                   adt_wall_to_grid_ms, matlab_datenum_to_wall_ms, dst_switch_between)

# 1. monitor headers (GE calendar wall clock) -> grid          [.adibin / .vital, Stage B / C]
g = ge_wall_to_grid_ms(header_wall_ms, episode_start_ms)

# 2. true local time -> grid                                    [Code Blue CodeTime, any re-identified time]
g = real_wall_to_grid_ms(matlab_datenum_to_wall_ms(CodeTime), offset_GE, episode_start_ms)
#      = episode_start + (UTC(real) − offset_GE d − UTC(episode_start))

# 3. EHR-table time (shifted by `offset` days, naive) -> grid   [labs, flowsheet, orders, encounter times]
g = ehr_wall_to_grid_ms(t_ehr_wall_ms, offset, offset_GE, episode_start_ms)
#      real = t_ehr + offset d (wall arithmetic), then rule 2

# 4. ADT / MRN-Mapping / ValidWaveTime times (shifted by `offset_GE` days, naive) -> grid
g = adt_wall_to_grid_ms(t_adt_wall_ms, offset_GE, episode_start_ms)

# legacy rule, for migration checks only:  t_ehr − (offset_GE − offset) d
```

Closed forms (`off_X` = LA UTC offset in force at X):

```
grid(real)  = real − offset_GE d + ( off_GE(origin) − off_real(real) )
grid(t_ehr) = t_ehr − 12 d       + ( off_GE(origin) − off_real(t_ehr + offset d) )
```

so the correction to the legacy rule is 0 when the origin (GE calendar) and the real instant share a
DST state, else ±60 min. All functions accept scalars or `numpy` int64 arrays (vectorised per hour
bucket).

## 6. What was affected and what changed

| Store / artefact | Before | After |
|---|---|---|
| `ucsf_all` waveform + `vitals_hf` + `nbp_events` (37,985 entities) | wall-clock placement; straddling cycles had a 1-h gap or a lost hour and ±60 min vitals desync after the switch | Stage B v2 / C v2 / G rerun on the 663 straddling cycles (2026-09-26/27, jobs 50142/50170/50184; gates B, C, F pass; 662 valid in the manifest); other entities unchanged (their grid is identical); manifest refreshed with `--keep-splits` (train/val/test 20,773 / 8,696 / 8,516 unchanged; total 2,488,195 wave-hours, +3.4 h net from recovered fall-back hours minus removed spring gaps) |
| `ucsf_all` `manifest.json` | — | `time_base`, `dst_switch_in_cycle`, `grid_utc_offset_min`, `stage_b_version`; splits kept (`--keep-splits`) |
| `ucsf` (CA-cohort store, 5,027 entities with Stage B output) `labs_events.npy`, `ehr_*.npy`, admission windows | naive `−12 d` rule → ±60 min in ~45 % of encounters | Rerun 2026-09-27 (jobs 50187–50192, 50216): Stage A (admission windows; multi-encounter matches resolved deterministically — the encounter whose admission window contains the wave start), Stage B v2 / C on the 85 straddling cycles with waveform, Stage D phase 2 + Stage E for all 5,027 entities with `ehr_wall_to_grid_ms` (`meta.labs.time_rule = "utc_continuous_v2"`); end-to-end check on a 300-entity pre-fix snapshot: 287 match the predicted shift exactly (0 min 154, +60 min 74, −60 min 72), the other 13 changed encounter. Manifest re-validated with the all-NaN rule (4,913 valid; job 50241); split membership kept, the 114 all-NaN entities dropped (train/val/test 3,435 / 726 / 752); BeeGFS copy `/projects/xhu40-cdsfm/physio_data/ucsf` re-synced (job 50242, 5,028 dirs) |
| `ucsf` `tasks/ca_prediction` | `event_time_ms` from ValidWaveTime `EventTime` (7–8 h late, ±60 min) | Code Blue `CodeTime` (CPA) via `real_wall_to_grid_ms`; new fields `event_time_source`, `event_in_wave`, `has_ca_csv`. Rebuilt 2026-09-27: 5,027 entities, 313 positive (187 patients), 4,714 negative (3,407 patients); patient-grouped 70/15/15 split, seed 42 |
| `.vital` from the monitor (var 100–199) | unaffected (monitor clock) | unaffected |
| CA cohort analyses (`ca_events_*.json`) | v2 = absolute rule at the event | `/projects/mwang80/staging/ca_events_grid.json` (`workzone/ucsf/explore/ca_events_grid.py`): Code Blue times on the fixed grid for 235 CPA patients (193 inside a cycle); equals v2 except −60 min for the 4 straddling events. **43 of these CPA patients are absent from the ValidWaveTime cohort** and were never held out (30 in pretrain train) — decision pending |

## 7. Verification recipes

* **Gate C** (`verify_stage_c.py`) now includes: for cycles with `dst_switch_in_cycle`, ECG-derived HR
  vs `vitals_hf` HR one hour *after* the switch must have median |lag| ≤ 8 s.
* **Independent EHR check** (any new EHR table): pick 200–1,000 cycles, convert the table's times
  with rule 3, compare against the monitor at lags −60/0/+60 min (HR: mean absolute error of the
  charted value vs the 5-min monitor median; NBP: exact `sys/dia` match within ±10 min). The right
  lag must win in > 95 % of cycles with a clear margin. Scripts: `clock_fs_lag_probe.py`,
  `clock_fs_nbp_probe.py`.
* **ADT check**: `bed_transferin_time` vs first header, grouped by (DST real, DST GE):
  `clock_fs_rowkeys.py`.
* **In-cycle switch**: `clock_ge_switch_ecg_test.py` (ECG vs vitals before/after),
  `clock_adibin_jump_check2.py` (chunk boundaries).

## 8. Checklist for a new EHR-side table

1. Which shift does the table carry? EHR tables: `offset`; ADT-derived: `offset_GE`; re-identified
   lists (Code Blue): none. If unsure, run the recipe in §7 with both.
2. Convert with `ehr_wall_to_grid_ms` / `adt_wall_to_grid_ms` / `real_wall_to_grid_ms`, never by
   subtracting days.
3. Use the entity's own `episode_start_ms` as origin; never mix cycles.
4. Record `time_rule` in `meta.json` and the source in the task/cohort JSON.
5. Re-run the lag check on a sample before trusting sub-hour horizons.

Related: `datasets/ucsf/API.md` (store layout), `datasets/ucsf/explore/RAW_FORMAT.md` (raw headers and
`.vital` timing classes A/B/C), `workzone/ucsf/slurm/README.md` (jobs).
