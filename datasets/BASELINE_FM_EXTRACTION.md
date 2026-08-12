# physio-data variants for UNIPHY baseline FMs

**physio-data prepares data for UNIPHY.** Swapping the FM in UNIPHY means the FM
wants its input at a specific **`(PPG_SR, ECG_SR, seg_len)`** — so we re-run the
extraction to produce a matching **canonical variant**. The current
`(40, 120, 30 s)` variant already exists (used by MOMENT / PPG-GPT / ECG-GPT /
physio-hnet). This doc lists the **new variants** the released baseline FMs need.

Scope of this prep: **waveform + ehr + actions only.** Embeddings and the UNIPHY
`fm/<name>/adapter.py` are downstream (UNIPHY-side, via `gen_embeddings.py`) and
**out of scope here.**

## What a "variant" is

A variant = one canonical re-run at a given **`seg_len`**, holding the channels
(`name @ SR`) that the FMs at that seg_len consume, with **waveform + ehr + actions
all aligned to that seg_len**:

- **waveform** — `stage_b_wave.py` re-run at the target SR + seg_len.
- **ehr** — `stage_c/d` → `stage_e_assemble` re-aligned: `seg_idx = searchsorted(new time_ms, event_time)`. Event `time_ms` / `var_id` / `value` unchanged; only `seg_idx` differs.
- **actions** — `stage3b_actions.py` re-aligned the same way (var_id 200–299).

`seg_len` sets the segmentation (⇒ `time_ms`, `N_seg`, all `seg_idx`). So **different
seg_len ⇒ different variant dir** (dim-0 differs; schema requires all channels share
dim 0). Channels at the **same** seg_len coexist in one dir.

Output on bedanalysis: `/opt/localdata100tb/physio_data/<variant>/{entity}/…`.

## Variants to produce (grouped by seg_len)

| Variant (`seg_len`) | Channels (`name @ SR`) | FMs served | Status |
|---|---|---|---|
| **30 s** | `PLETH40`, `II120` | MOMENT, PPG-GPT, ECG-GPT, physio-hnet | ✅ exists |
| **10 s** | `PLETH125`, `PLETH50`, `II500` | PaPaGei, AnyPPG, HeartGPT-PPG, ECGFounder-1 | ⬜ new |
| **240 s** | `PLETH50` | Pulse-PPG, SIGMA | ⬜ new |
| **5 s** | `II100` | HeartGPT-ECG | ⬜ new |

Per-FM requirement (what each channel is for):

| FM | channel | SR | seg_len |
|---|---|---:|---:|
| PaPaGei-S / AnyPPG | PLETH | 125 | 10 s |
| HeartGPT-PPG | PLETH | 50 | 10 s |
| Pulse-PPG / SIGMA | PLETH | 50 | 240 s |
| ECGFounder-1 | II | 500 | 10 s |
| HeartGPT-ECG | II | 100 | 5 s |

3 new variants × 4 datasets = 12 canonical re-runs. `PLETH50` is derivable from
`PLETH125` by downsample — store it explicitly (physio-data delivers each SR) or let
the frozen adapter downsample; either is fine since resample there is offline.

## Per-dataset source-rate caps (SR is `min(source, target)`)

`samples_per_seg = SR × seg_len`. Where source < target, the stored channel is
source-capped and that FM runs upsampled = OOD (flag it; don't fabricate).

| Dataset | PLETH src | II src | 10 s: PLETH125 / II500 | 240 s: PLETH50 | 5 s: II100 |
|---|---:|---:|---|---|---|
| **MC-MED** | 125 | 500 | 125 / 500 ✅ | 50 ✅ | 100 ✅ |
| **VitalDB** | 500 | 500 | 125 / 500 ✅ | 50 ✅ | 100 ✅ |
| **MIMIC-III** | 125 | 125 | 125 / **125** ⚠️ (ECGFounder 4× up) | 50 ✅ | 100 ✅ |
| **MOVER** | 100 | 300 | **100** ⚠️ / **300** ⚠️ + lead? | 50 ✅ | 100 ✅ |

**Report rule:** ECGFounder only on MC-MED + VitalDB (native 500, true Lead II);
on MIMIC-III / MOVER either drop it or flag OOD. All PPG baselines run on all 4
(MOVER PaPaGei/AnyPPG mild upsample from 100).

`samples_per_seg` per channel (MC-MED/VitalDB rates): `PLETH125`@10 s = 1250 · `PLETH50`@10 s = 500 · `II500`@10 s = 5000 · `PLETH50`@240 s = 12000 · `II100`@5 s = 500.

## Producing a variant (per dataset)

Re-run the existing stages with the variant's `(SR…, seg_len)`; nothing new invented:

```
stage_b_wave.py     --channels PLETH@125,PLETH@50,II@500  --seg_len 10   → PLETH125/PLETH50/II500 + time_ms
stage_c_vitals.py / stage_d_labs.py / stage_e_assemble.py --seg_len 10   → ehr_{baseline,recent,events,future} (seg_idx @10s)
stage3b_actions.py  --seg_len 10                                          → ehr_actions (seg_idx @10s)
```

Suggested change to keep the pipeline reusable: **`stage_b_wave.py` takes a channel
list `(name, target_SR)` and a `--seg_len` instead of hardcoded PLETH40/II120/30 s;**
the ehr/action stages take `--seg_len` and re-align via `searchsorted`.

## Out of scope (UNIPHY-side, downstream)

- `uniphy/model/fm/<baseline>/adapter.py` (`normalize` + `embed`, reusing
  `Physio_HNET/baselines/*_adapter.py::_preprocess`/`_forward`).
- `scripts/gen_embeddings.py --fm <NAME>` → cached `<FM>.npy` (frozen path only).
- Trainable path reads the same raw channels on-the-fly (no cache).

## Caveats

1. **MIMIC-III ECG is 125 Hz native** — `II500` there is actually `II125`; ECGFounder = 4× upsample.
2. **MOVER**: PLETH 100 / II 300 below targets; `ECG1` lead unverified; EPIC ~40 % DATADOWN empty XML.
3. **SIGMA** is fixed-window (240 s patch grid) — the 240 s variant is mandatory for it; it cannot use 30 s.
4. **ECGFounder-1 is Lead-I-trained**; feeding Lead II is a uniform mild single-lead shift on all datasets.
5. **SIGMA** loads 87.5 % of its backbone (missing qk_norm) — verify before trusting numbers.
6. **PLETH+II paired coverage is a subset** in MIMIC-III (not every record has PLETH).
7. ehr/actions per variant carry the **same events**, only `seg_idx` differs — re-align is a cheap `searchsorted`, no waveform re-read.
8. Keep this doc + paths **local** (clinical-cohort detail); do not commit to a public remote.
