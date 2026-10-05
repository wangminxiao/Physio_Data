# MLADI — Step 0a research notes

Pitt MLADI: bedside-monitor recordings exported by **Philips IntelliVue Data Warehouse
Connect (DWC)**, one HDF5 per encounter, with the encounter's EHR tables inside the same file.
The data live on PSC Bridges-2 only (`/ocean/projects/med250003p/shared/`); nothing leaves PSC.
Everything below was measured on PSC between 2026-09-28 and 2026-10-04 (Physio_HNET scripts
named in brackets) unless marked *to verify*.

## Files and entities

- Raw: `mladi_extract_2023_waves/<base>.h5`, **16,422 files**, `base = YYYYMMDD_<encounterID>_<patientID>`
  (patient = last field; `Physio_HNET/scripts/data/build_mladi_labels.py::extract_patient_id`).
  One file = one encounter; a patient can have several.
- Earlier derived set (Physio_HNET pretraining): `pretrain_wav_v2/<base>_II_120Hz_<n>_3600_mmap.npy`,
  `<base>_Pleth_40Hz_<n>_1200_mmap.npy` (float16, 30-s rows, no overlap), `<base>__meta.json`
  (`seg_list = [block, seg_idx, start_s]`), `<base>__numerics.npz`. Built by
  `/jet/home/mwang11/workspace/data_preparing_v2.py`: both channels interpolated from the H5 time
  columns onto one uniform grid per block (blocks split where BOTH channels are absent > 5 s), Butterworth
  order-4 `filtfilt` (Pleth 0.5–12 Hz at 125 Hz, II 0.5–50 Hz at 500 Hz), `resample_poly` down, rows
  kept only where the Pleth row is not flat (std ≥ 1e-4), values clipped to ±1000.
- Split already in use (`data_cache/e1_mladi/mladi_encounter_index.json`): **patient-level 70/15/15**,
  seed 42, over 13,168 "kept" encounters (≥ NBP acuity threshold); 3,254 are `unassigned`.

## HDF5 layout (from `mladi_raw_check.py`, `mladi_pat_device.py`)

- `/data/waveforms/<label>`: compound `(time f8, value f4)`; attribute `.meta` = JSON with
  `dwc_meta` {`id`, `basePhysioId`, `physioId`, `label`, `samplePeriod` (ms), `unitLabel`, `clipLow`,
  `clipHigh`, `minTime`, `maxTime`, ECG `lowEdgeFrequency`/`highEdgeFrequency`, `ecgLeadPlacement`}.
  Channels seen (encounters carrying each, of 12,636 with paired labels): II 12,636, Pleth 12,636,
  Resp 12,635, V 12,616, aVR 12,490, III 7,378, I 6,792, **ART 4,426**, **ABP 1,216**, MCL 792,
  CVP 466, ICP 428, PlethT 414, aVF 246, CO2 206, UAP 188, aVL 184, PAP 136.
  Rates: II 500 Hz (250 Hz in ~0.7 %), Pleth / ART 125 Hz.
- `/data/numerics/<label.sublabel>` (e.g. `NBP.NBPs`, `HR.HR`, `SpO₂.SpO₂`, `RR.RR`, `Perf.Perf`):
  same clock as the waveforms; ~1 Hz for continuous parameters, NBP at each cuff reading
  (`Physio_HNET/scripts/data/extract_mladi_numerics.py`). The full key list is *to verify* (0b).
- `/ehr/<table>` (structured arrays): `demographic` (race, age, sex, facility, unit, regDate,
  dischDate, ethnicity, encntrType, dischDisp), `patient` (time, category, pacedMode,
  resuscitationStatus, admitState, clinicalUnit, gender), `location` (beginDate, endDate, facility,
  unit), `lab_results` (time, orderedAs, eventDisp, resultVal, resultStat, resultUnit, eventTag,
  normalcy*, validDate), `medications` (orderedAs, catalogDisp, time, resultVal, dose, doseUnit, route,
  volumeDose…), `infusions_and_outputs` (time, name, detail, volume, unit), `low_rate` (date,
  eventName, resultVal, resultUnit, resultStat, eventTag), `diagnostic_codes`, `csce`,
  `culture_sensitivity`. `demographic.facility` and `.unit` are masked ("0").
- `/dwc/alerts`.

## Known issues (measured)

1. **Time stamps are synthesized from sample counts**: Pleth `dt` is exactly 8.000 ms, II 2.000 ms, in
   every file looked at; the H5 time columns carry no re-sync information.
2. **ECG → Pleth system delay ~1.2 s** (R-peak → Pleth foot median 1200 ms, IQR 1183–1217), the same
   in every year 2019–2025; > one inter-beat interval, so "nearest foot" pairs the wrong beat.
3. **Pleth re-sync sawtooth**: ramp 28–30 ms/min, reset −25…−30 ms every 64–68 s, on 99.5 % of streams;
   only on the Pleth path (R → ART foot has none). Matches Philips' frontend note and Ruffolo et al. 2025.
4. **Pleth sensor gaps move the delay level** (ART → Pleth leg, |jump| > 25 ms in 22 % of gaps vs 2 %
   in continuous recording); monitor restarts do not.
5. **Invalid-value codes**: Pleth carries large negative sentinels (~−2.7e8, −1.7e7) in stretches; raw
   Pleth otherwise lies in [0.25, 0.75] (`clipLow 0`, `clipHigh 1`, autoscaled).
6. **Hidden gaps**: some `seg_list` blocks contain missing rows (Pleth-flat rows dropped inside a block),
   ~0.4 % of 5-min windows.
7. ECG filter setting varies (highEdge 150 vs 40 Hz, lowEdge 0.05 vs 0.5 Hz, ~2.6 %), shifting R timing ~8 ms.
8. Bridges-2 L40S nodes are AVX2-only; the shared env's AVX-512 extensions need
   `Physio_HNET/scripts/psc/l40s_shim` there (not relevant to extraction, which is CPU-only).

## To establish in 0b / 0c

- Full `data/numerics` key list per encounter; which carry ART/ABP numerics; cadence.
- EHR time columns: type and origin relative to the waveform clock (the waveform time is seconds on a
  de-identified axis, e.g. minTime ≈ 2.7e6 s); `regDate`/`dischDate` vs waveform span; whether one
  offset per encounter maps EHR time onto the waveform axis.
- Charted vitals in `low_rate` vs monitor numerics (alignment check at lags −60/0/+60 min, as
  the skill requires).
- Whether `time_ms` can be the waveform clock × 1000 directly (continuous real elapsed time within
  an entity) or needs an origin.
