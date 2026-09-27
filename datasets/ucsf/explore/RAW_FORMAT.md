# UCSF raw file formats — what is actually stored, at what rate, native or upsampled

Verified 2026-09-09 on DREAM node `xhu40-n01` against `/mnt/localdata/storage/UCSF` (the old
`/labs/hulab/UCSF`). Evidence comes from four read-only probes
(`workzone/ucsf/explore/raw_format_probe_v1..v4.py`, jobs 47207/47219/47234/47240): a header scan of
9,492 `.adibin` files in 603 bed directories (5 patients per cohort folder), data-level tests on 24–40
patients, and 513 `.vital` files. Only aggregates were printed.

> **Clock conventions (2026-09-26)**: header wall clocks of `.adibin`/`.vital` files are the real time shifted in
> ABSOLUTE (UTC) time by `offset_GE` days; EHR/ADT tables are shifted on the wall clock. Inside a cycle the streams
> are continuous in real elapsed time, so the entity grid is UTC-continuous. Full write-up, evidence and formulas:
> [`../ALIGNMENT.md`](../ALIGNMENT.md); code: `workzone/ucsf/clock.py`.

## 0. Layout and naming

```
{YYYY-MM}-deid/DE{Patient_ID_GE}/{bed_subdir}/
    DE{pid}_{YYYYMMDDHHMMSS}_{WaveCycleUID}.adibin            waveform block (token pattern DE+15_D14_D5)
    DE{pid}_{YYYYMMDDHHMMSS}_{WaveCycleUID}_{suffix}.vital    one numeric stream per suffix
{YYYY-MM}-deid/DE{Patient_ID_GE}/MRN-Mapping.csv              bed/wave-cycle table
{YYYY-MM}-deid/DE{Patient_ID_GE}/Alarms.csv                   alarm log
```

| Fact | Value |
|---|---|
| Cohort folders / DE dirs / unique Patient_ID_GE | 70 / 26,021 / 22,175 |
| `MRN-Mapping.csv` present | 26,012 dirs; 167,438 rows; **49,501 unique (pid, WaveCycleUID)** |
| WaveCycleUIDs per bed dir | 1 in 514 of 603 dirs, 0 in 69 (no adibin), 2 in 19, 3 in 1 |
| `.adibin` files per UID | p50 9, p90 37, max 286 |
| 14-digit timestamp in the filename | **equals the header start time exactly** (adibin n=758, vital n=419: delta 0 s). It is `00000000000000` when the `.vital` header time is zero. |
| Last token of the `.adibin` name | WaveCycleUID (4–5 digits); the same token sits before the suffix in `.vital` names |
| `MRN-Mapping.csv` columns | `MRN_ADT, UnitBed, BedTransfer_In, BedTransfer_Out, MRN_WaveCycleTable, WaveCycleUID, WaveStartTime, WaveStopTime` (a second variant adds `WaveCycleStart/WaveCycleStop`, has leading spaces in names and a trailing empty column). Timestamps `MM/DD/YYYY hh:mm:ss AM/PM`. ~10 % short rows. |
| `Alarms.csv` columns | `MRN_Given, MRN_InAlarmTable, WavecycleuID, Message, AlarmStartTime, AlarmEndTime, Duration, UnitBed, TransferIn, TransferOut, DataSource` |
| MRN-Mapping window vs adibin coverage | start delta p10/p50/p90 = −3,595 / **6** / 85,625 s; stop delta = −95,298 / −4,613 / 176 s. The mapping window is the bed stay; the waveform usually covers a subset of it. Every UID with adibin has a mapping row. |

## 1. `.adibin` — CFWB v1 binary (ADInstruments Chart binary as written by BedMaster)

**Byte layout** (little-endian, no padding)

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | magic `CFWB` |
| 4 | 4 | Version (int32) = 1 |
| 8 | 8 | secsPerTick (float64) = 1/240 in **all** 9,492 files |
| 16 | 20 | Year, Month, Day, Hour, Minute (int32 ×5) — file start, GE-shifted calendar |
| 36 | 8 | Second (float64) |
| 44 | 8 | trigger (float64) |
| 52 | 16 | NChannels, SamplesPerChannel, TimeChannel (=0 always), DataFormat (=3 int16 always) |
| 68 | 96 × NChannels | per channel: Title 32 s, Units 32 s, scale, offset, RangeHigh, RangeLow (float64 ×4) |
| 68 + 96 n | 2 × n × N | samples, **sample-major interleaved** `(N_samples, N_channels)` int16 |

Physical value = `scale × (raw + offset)`; offset is 0 everywhere. Gap sentinels `−32767 / −32768`
(< 0.2 % of samples in the probes). RangeHigh/RangeLow are either (1, 0) or (0, 100) — metadata noise,
not a real range. Units label is `Uncalib` in ~12 % of files with identical scale; ignore the label.
File duration: p10 60 s, p50 312 s, p90 12.7 h, p99 53.7 h, max 218 h; 27 zero-sample files.
Consecutive files of one UID are contiguous (gap p50 0 s, p90 2 s, no overlaps).

**Channels** (presence = % of 9,492 files)

| Title | Presence | Header unit / scale | Real physical unit | Raw p1 / p50 / p99 (LSB) | Native rate | How 240 Hz was produced (evidence) |
|---|---|---|---|---|---|---|
| I, II, III, V, AVR, AVL, AVF | 100 % (V 99 %) | `mV`, 2.44 | **µV per LSB** (II → −266 … 908 µV; R waves ≈ 0.9 mV). Header says mV but 2.44 µV/LSB is the GE ECG resolution. | II: −109 / −2 / 372 | **240 Hz native** | None. Phase test contrast ≈ 1 %; rebuilding from every 2nd sample leaves 4–5.5 LSB RMS error vs 0.5 for rounding. |
| SPO2 (pleth / PPG) | 80 % | `%`, 1.0 | arbitrary ADC units (≈ 390 … 1,671), not percent | 390 / 1,014 / 1,671 | **60 Hz native** | ×4 by **three equal linear steps plus one repeated sample per native period** (first-difference pattern `a, a, 0, a`; block ratios 1/3, 2/3, 2/3, 1). Neither pure hold nor pure interpolation; no information above 30 Hz. Values step by 2 LSB. |
| RR (respiration impedance waveform) | 96 % | `Imp`, 0.1 | arbitrary | −300 / −10 / 380 | **60 Hz native** | ×4 **sample-and-hold** (hold-from-every-4th error 0.00 LSB; dup pattern 100/100/100/16 %). |
| AR1 / AR2 / AR3 (arterial pressure waveform) | 29 % / 6 % / <1 % | `mmHg`, 0.2 | mmHg (AR1 → 60 … 127 mmHg) | 299 / 370 / 636 | **120 Hz native** | ×2 **linear interpolation** (rebuild-from-every-2nd error 0.25 LSB = rounding; from every 4th 1.2 LSB). |
| CVP1–4, PA2/PA4, SP2/3, LA4, FE*, ICP1/2 | CVP2 8 %, ICP2 7 %, PA2 4 %, ICP1 4 %, rest < 1 % | `mmHg`, 0.2 | mmHg | CVP2 0 / 21 / 37 | 120 Hz (same pressure module; lin2 error 0.25 LSB) | ×2 linear interpolation. CVP/ICP are very flat (1 LSB steps). |
| `` (blank title) | 12 % | `mmHg`/`Uncalib`, 0.2 | unlabeled pressure channel | — | — | Keep out unless a label can be recovered. |
| V2–V5 | rare | `Uncalib`, 0.2 | scale suggests a pressure-type channel despite the ECG-like name | — | — | Ignore. |

Implication: any target rate ≤ 60 Hz for PPG (our `PLETH40`) and ≤ 120 Hz for arterial waveforms is
loss-free with respect to the source; ECG is genuinely 240 Hz.

## 2. `.vital` — numeric monitor streams (one file per suffix per wave cycle)

**Byte layout**

| Offset | Size | Field |
|---|---|---|
| 0 | 16 | Label (= suffix, e.g. `HR`, `NBP-S`) |
| 16 | 8 | Uom (`Bpm`, `mmHg`, `%`, `BrMin`, `Deg C`, `mm`) |
| 24 | 8 | Unit = ICU unit name (`13ICU`, `9ICU`, `8NICU`, `11NICU`, …), not a physical unit |
| 32 | 4 | Bed |
| 36 | 20 | Year, Month, Day, Hour, Minute (int32 ×5) — **zero in 37 % of files** |
| 56 | 8 | Second (float64) |
| 64 | 32 × N | samples: `value, offset_sec, low, high` as float64 ×4 |

`low`/`high` are alarm-limit placeholders, constant per file (`−999999 / 32768`; `3276.8` for
0.1-scaled channels such as TMP and ST; `999999` for CUFF; `0 / 0` for the `-R` pulse-rate files).
`value` never carried the `−999999` sentinel in the probes. `offset_sec` is an **integer**.

**Cadence**: every suffix is a **2.000 s grid** (≥ 98 % of consecutive offsets differ by exactly 2 s;
missing samples are simply absent, producing gaps up to hours). This is one monitor reading per 2 s,
not an upsampled or aligned stream. The intermittent quantities are visible only through value
changes:

| Suffix | Uom | Presence (400 DE dirs) | Value changes every | Typical range (p1–p99) | Notes / registry id |
|---|---|---|---|---|---|
| HR | Bpm | 97 % | 2 s | 57–94 | ECG heart rate → 100 / hf 150 |
| SPO2-% | % | 97 % | ~8 s | 91–100 | → 101 / hf 151 |
| SPO2-R | Bpm | 97 % | ~4 s | 56–93 | oximeter pulse rate → 115 / hf 156 |
| RESP | BrMin | 97 % | 2 s | 9–50 | impedance respiration rate → 102 / hf 152 |
| PVC | Bpm | 97 % | 60 s | 0–1.5 | PVC per minute → 114 / hf 164 |
| NBP-S / NBP-D / NBP-M | mmHg | 96.5 % | **31–55 min** (2-s stream that repeats the last cuff reading) | 99–160 / 51–75 / 71–105 | cuff BP → 104/105/106 / hf 157–159. Extract as events at value changes (or use CUFF bursts as measurement markers). |
| CUFF | mmHg | 97 % | 2 s, but only during inflation (bursty, dt p95 140 s) | 0–160 | cuff pressure during a measurement; marks cuff cycles |
| AR1-S / -D / -M / -R | mmHg / Bpm | 59.5 % (AR2 20 %, AR3 rare) | 2 s | 91–151 / 39–76 / 56–112 / 55–101 | arterial line → 110/111/112/113 / hf 153–155 |
| FE1 / FE3 -S/-D/-M/-R | mmHg / Bpm | rare | 2 s | — | femoral line; treat like AR |
| CVP1–4 | mmHg | CVP2 16 %, others < 5 % | 4 s | 5–18 | → 107 / hf 160 |
| PA2 / PA4 -S/-D/-M | mmHg | 6.5 % | 4 s | — | pulmonary artery → hf 161–163 |
| TMP-1 / TMP-2 | Deg C | 37 % / 16 % | 4 s | 34.5–37.9 | → 103 |
| ST-I/II/III/V1/V2/V3 | mm | 23 % (V2/V3 14 %) | 12 s | ±1 | ST deviation → 120–122 (I, II, V) |
| ICP1/2, CPP1/2, SP2/3 | mmHg | ≤ 3.5 % | 2–4 s | — | neuro ICU |

**Time base — three file classes** (513-file sample; header-time-zero share confirmed at 37 % in the
wide scan, uniform across suffixes):

| Class | Share | Header time | `offset_sec` content | How to place samples in time |
|---|---|---|---|---|
| A `hdrT + rel` | 64.5 % | present | seconds from header time, starting at 0 | `t = header_time + offset`. Same GE-shifted calendar as `.adibin`: HR starts 4–16 s after the first adibin sample of the UID and ends 44 s before its last (medians). |
| B `zeroT + rel` | 10 % | zero (filename timestamp also zero) | relative, starting at 0 | Anchor = **first `.adibin` start of the same WaveCycleUID**: HR start delta 5 s median (p10 0, p90 42 s); ends −362 s median. MRN-Mapping `WaveStartTime` is a worse anchor (±1 h tails). |
| C `zeroT + mixed` | 20–25 % (HR files: 20 % of 1,253 random; only cohorts 2016–2018) | zero | first 35–76 % of the file relative (as B), then it **switches to absolute seconds since 0001-01-01** | Relative part as B. The absolute part lands **52–285 days (median 144) after** the end of the relative part when converted, i.e. it is on a different calendar (most likely the real, un-shifted dates) — never use it as a date. **Re-anchoring by continuity (`abs_first = rel_last + 2 s`) is validated**: over 248 class-C HR files the re-anchored tail ends within 60 s of the waveform end in 60 % of files (median +12 s, p90 +58 s, i.e. it practically never overshoots) and the rest end early exactly as class-A streams do; raw values are continuous across the seam and ECG-derived HR agrees with the re-anchored tail (job 47327/47328). Exception: the NBP-S/D/M absolute tails repeat a single offset (degenerate) and are dropped. |

The current canonical Stage C (`workzone/ucsf/stage_c_vital.py`) computes `vital_start_ms` from the
header and returns `None` when the year is 0, so **class B and C files were silently skipped** in the
5,027-entity store; class C absolute tails were clipped away. Vitals coverage there is therefore
incomplete for roughly a third of the files.

Zero-time anchoring check (smoke parquet, 377 wave cycles from the class-C target list plus 240 others):
with anchor = first `.adibin` of the wave cycle, the in-window fraction of the relative part is 1.00 at
the 10th percentile for every suffix (HR first_off p50 0 s, p90 128 s); 3 of 176 NBP files carry
negative or constant offsets and fall outside the window (dropped).

**Header size**: the `.vital` header is 64 bytes (`struct.calcsize("<16s8s8s4siiiiid")`). Hard-coding
56 shifts every column by one and silently turns the value column into "offsets".

## 3. Alignment summary

* `.adibin` and class-A `.vital` share one clock (GE-shifted, naive local): coverage of a UID matches
  to within 0–10 s at the start and −2 … −58 s at the end.
* Filename timestamp == header time for both file types, so the filename can be used to sort and
  bucket files without opening them; a zero timestamp flags class B/C.
* `MRN-Mapping.csv` windows are bed stays, wider than the recordings; use file headers to define the
  waveform window and the mapping only for UID ↔ bed ↔ patient linkage.

## 3b. Clock synchronisation between `.adibin` and `.vital` (physiological cross-check, job 47248)

`workzone/ucsf/explore/clock_sync_probe.py`: 14 class-A patients, 15-min chunk from the middle of the
longest lead-II file, R-peak heart rate on a 2-s grid vs `HR.vital`; AR1 waveform per-2-s systolic /
diastolic / mean vs `AR1-S/D/M.vital` (5 patients). Lag scanned ±40 s; positive lag = numeric lags the
waveform.

| Pair | Best lag | Corr at best lag (median) | MAE at best lag | MAE at lag 0 | Notes |
|---|---|---|---|---|---|
| ECG-derived HR vs `HR.vital` | **+2 to +4 s in 12 of 14** (median 4 s) | 0.80 (0.96–0.99 on clean ECG) | 1.0 bpm | 1.9 bpm | the 2–4 s is the monitor's beat-averaging delay, not a clock offset; 2 outliers are noisy/paced segments |
| AR1 waveform sys vs `AR1-S` | 0 to +2 s | 0.86–0.97 | 1.7–3.5 mmHg | 3.1–3.9 | |
| AR1 waveform dia vs `AR1-D` | 0 s | 0.79–0.92 | 1.2–2.4 mmHg | same | |
| AR1 waveform mean vs `AR1-M` | +2 s | 0.75–0.92 | 1.0–2.9 mmHg | 1.3–3.4 | |

Conclusion: class-A `.vital` files and `.adibin` files run on the same clock to within one 2-s grid
step; the remaining lag is the monitor's own numeric averaging. `.adibin` files start on whole seconds
(header `Second` fractional part is 0 in every probed file) and `.vital` offsets are integers, so the
two grids are commensurate. Class B/C files carry no time of their own; anchoring them at the first
`.adibin` of the same UID is an inference with a ~5 s median error that cannot be verified per file.

**MIMIC-III correspondence** (what plays the role of the MIMIC numerics `n.dat` "hf" records):

| MIMIC-III | UCSF equivalent | Cadence |
|---|---|---|
| numerics `ABPSys / ABPDias / ABPMean` (→ 153/154/155 in `ehr_hf.npy`) | `.vital` **`AR1-S / AR1-D / AR1-M`** (plus AR2/AR3, femoral FE1/FE3) | 2 s (MIMIC: 1 s or 1 min) |
| numerics `NBPSys / NBPDias / NBPMean` (157–159) | `.vital` `NBP-S / NBP-D / NBP-M`, taken at value changes | one reading per cuff cycle |
| numerics `HR`, `%SpO2`, `RESP`, `PULSE` (150–152, 156) | `.vital` `HR`, `SPO2-%`, `RESP`, `SPO2-R` | 2 s |
| ABP **waveform** record (125 Hz) | `.adibin` channel `AR1` (240 Hz stream, 120 Hz native) | — |
| PLETH / II waveform records | `.adibin` `SPO2` (60 Hz native) / `II` (240 Hz) | — |

## 4. What this means for a paired PPG + ECG + vitals dataset

| Stream | Store at | Why |
|---|---|---|
| PPG | ≤ 60 Hz (40 Hz keeps the canonical `PLETH40`) | source is 60 Hz; the 240 Hz stream adds nothing |
| ECG (any of 7 leads) | 120 or 240 Hz | native 240 Hz |
| ABP waveform (optional) | ≤ 120 Hz | native 120 Hz |
| RR waveform (optional) | ≤ 60 Hz | native 60 Hz hold |
| HR, RR, SpO2, ABP s/d/m, PR | 0.5 Hz grid (15 per 30 s segment) | that is the true monitor cadence |
| NBP s/d/m | events at value changes (~every 30–60 min), optionally with CUFF-burst timestamps | the 2-s stream is a hold of the last reading |
