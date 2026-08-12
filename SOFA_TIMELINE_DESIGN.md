# SOFA Timeline — Design & Coverage Findings

Status: **design / pre-implementation** (2026-06-30). Nothing built yet.
Scope: the 3 ICU datasets only — **MIMIC-III, Emory, UCSF**. (MOVER/VitalDB/MC-MED
are perioperative/ED — SOFA not meaningful there.)

## 0. Goal

Produce a **regular, training-ready SOFA severity timeline** per ICU entity,
aligned to waveform segment indices, stored in the **same on-disk shape as
`ehr_events.npy`** (`EHR_EVENT_DTYPE`), so a training adapter can merge it in.

This replaces the current event-driven SOFA (MIMIC-III `post_sepsis_cohort.py`),
which is sparse, never emits a total, and has scoring bugs (see §3).

## 1. Locked design decisions

| Decision | Value | Note |
|---|---|---|
| Aggregation window | **6 h trailing worst (max)** per component | deviates from clinical 24 h on purpose — want high-res, waveform-aligned acuity |
| Grid step | **1 h** | one score row per hour per component |
| Missing handling | **carry-forward 24 h + mask** | see §5 for the 3-state mask |
| Storage | **separate `sofa_*.npy`** (4 partitions), `EHR_EVENT_DTYPE` | do NOT touch canonical `ehr_events.npy`; adapter merges at train time |
| Shared code | `physio_data/severity_scores.py`, variable-driven | one formula, 3 datasets; graceful degradation on missing components |
| Pipeline stage | post-stage, after Stage E (needs `ehr_events`+`time_ms`) | mirrors existing `post_sepsis_trajectory.py` |

## 2. Reference standard (MIT-LCP mimic-code `sofa.sql` + Vincent 1996)

Clinical SOFA: hourly, each component = **worst value over trailing 24 h**,
missing component → 0 (normal). Threshold table:

| Component | Input | 4 | 3 | 2 | 1 | 0 |
|---|---|---|---|---|---|---|
| Resp | PaO₂/FiO₂ mmHg | <100 **& vent** | <200 **& vent** | <300 | <400 | ≥400 |
| Coag | Platelets ×10³/µL | <20 | <50 | <100 | <150 | ≥150 |
| Liver | Bilirubin mg/dL | ≥12 | ≥6 | ≥2 | ≥1.2 | <1.2 |
| Cardio | MAP + pressors µg/kg/min | Dopa>15 / Epi/NE>0.1 | Dopa 5–15 / Epi/NE≤0.1 | Dopa>0 / Dobu any | MAP<70 | MAP≥70 no药 |
| CNS | GCS | <6 | 6–9 | 10–12 | 13–14 | 15 |
| Renal | Creatinine mg/dL or UO mL/24h | ≥5 / UO<200 | 3.5–4.9 / UO 200–499 | 2.0–3.4 | 1.2–1.9 | <1.2 |

Sources: MIT-LCP/mimic-code `mimic-iv/concepts/score/sofa.sql`; original Vincent et al. 1996.

**Our deviation**: 6 h worst instead of 24 h. Document this in any paper — our SOFA
is a high-resolution variant, not bit-identical to clinical SOFA.

## 3. Review of existing implementation (`mimic3/post_sepsis_cohort.py::compute_sofa_from_ehr_events`)

Event-driven (one component emitted per driver measurement). Findings:

- 🔴 **Liver: score=1 band collapsed** — `elif bili>=1.2: score=2` maps bilirubin
  1.2–1.99 to 2 (should be 1). Overscores mild hyperbilirubinemia.
- 🔴 **`SOFA_total` (300) never emitted** — only the 6 components are written; defined
  in the dict but never appended.
- 🔴 **No 24 h/6 h worst aggregation, no carry-forward** — just point values at
  irregular times; components never coincide so a coherent total can't form.
- 🟠 **Resp ignores ventilation** — `mechvent` (205) exists but unused → non-ventilated
  patients with low P/F wrongly scored 3/4.
- 🟠 **Renal ignores urine output** — `urine_output` (206) exists but unused → oliguric
  renal failure underscored.
- 🟠 **Cardio missing tier-2** (low-dose dopamine / dobutamine) and lumps all pressors
  into one NE-eq value (var 200 is designed that way, but tier-2 is unreachable).
- 🟡 Only MIMIC-III has any SOFA code. Emory Stage G builds cohort labels only; UCSF none.

## 4. Coverage check — SOFA-driver variables in canonical output

Sampled 150 entities/dataset, % carrying each driver var (2026-06-30, server `bedanalysis`):

| Component | Driver (var_id) | MIMIC-III | Emory | UCSF |
|---|---|---|---|---|
| Coag | Platelets (7) | 99% ✅ | 73% ✅ | 99% ✅ |
| Liver | Bilirubin (6) | 61% ✅ | 68% ✅ | 74% ✅ |
| Renal | Creatinine (5) | 99% ✅ | 70% ✅ | 99% ✅ |
| Renal | urine_output (206) | 82% ✅ | **0% ❌** | **0% ❌** |
| Cardio | MAP (106/112) | 87% ✅ | 88/66% ✅ | 95/56% ✅ |
| Cardio | vasopressor (200) | 15% ✅* | **0% ❌** | **0% ❌** |
| Resp | paO₂ (14) | 60% ✅ | 43% ✅ | 87% ✅ |
| Resp | FiO₂ (203) | 40% ✅ | **0% ❌** | **0% ❌** |
| Resp | mechvent (205) | 41% ✅ | **0% ❌** | **0% ❌** |
| CNS | GCS (108) | **11% ⚠️** | **0% ❌** | **0% ❌** |

\* 15% is normal — only treated patients have a pressor rate.

### Per-dataset feasibility

- **MIMIC-III** — all 6 components computable. Caveat: **GCS only 11%** because the
  registry maps only CareVue itemid 198; MetaVision GCS (220739 + 223900/223901)
  is unmapped. Cheap enrichment available.
- **Emory** — only **Coag, Liver** fully computable. **Renal** = creatinine-only
  (no UO), **Cardio** = MAP-only (**caps at score 1**, no pressors). **Resp** = paO₂
  present but **no FiO₂ → P/F uncomputable**. **CNS** absent. Raw EHR HAS the missing
  pieces: `JGSEPSIS_FIO2.csv`, `JGSEPSIS_MEDS.csv` (pressors, 44 GB),
  `JGSEPSIS_OUTPUT.txt`/`INOUTS_ALL.txt` (UO), `JGSEPSIS_VENT` (vent); GCS table TBD.
  → extraction feasible.
- **UCSF** — same shape as Emory (Coag/Liver full; Renal/Cardio degraded; Resp/CNS
  absent). Missing drivers are in **flowsheets + medication orders**, both **deferred**
  during onboarding (`FLOWSHEETVALUEFACT` skipped, meds = orders not admin, dose dirty).
  → extraction harder.

## 5. Proposed mechanics (once path chosen)

1. Build hourly grid `t_0…t_k` over the entity's admission window (capped baseline/future).
2. For each component at each `t_k`:
   - worst (max-severity) driver value in **[t_k−6h, t_k]** → score;
   - if 6 h window empty, carry forward last value within **[t_k−24h, t_k)** (stale);
   - if 24 h empty, component = **0 (imputed)**.
3. `SOFA_total` (300) = sum of available components.
4. Emit `EHR_EVENT_DTYPE` rows for var_ids 300–306 (score `value`).
5. **Mask** (3-state, stays in dtype): parallel rows in a reserved mask range
   (proposal: 310 = total, 311–316 = components), `value ∈ {1.0 fresh ≤6h, 0.5 stale
   6–24h, 0.0 imputed/missing}`.
6. Run all rows through `physio_data.ehr_trajectory.split_events()` → real `seg_idx`
   in-window, sentinels elsewhere → write `sofa_{baseline,recent,events,future}.npy`.
7. Training adapter forward-fills hourly scores across the 30 s segments.

`severity_scores.py` takes a per-dataset config mapping each component → input
var_id(s); components with no mapped/available driver are skipped and recorded in
`meta.json` (`sofa_components_available`).

## 6. OPEN DECISION (blocking implementation)

How to handle Emory/UCSF missing drivers. Options on the table:
- **A (recommended)** — MIMIC-III full now (+GCS itemid enrichment); Emory add an
  extract mini-stage for FiO₂/pressor/UO/vent/GCS → near-full SOFA (flagship sepsis
  dataset, worth it); UCSF partial SOFA + mask now, revisit extraction later.
- **B** — all three partial SOFA + mask now (fastest; Emory/UCSF total clinically weak).
- **C** — extract for both Emory AND UCSF (most complete; UCSF requires reopening
  flowsheets + meds — large effort, dirty dose data).
- **D** — MIMIC-III only for now.

Sub-decisions pending: 3-state mask encoding (§5.5), GCS itemid enrichment for MIMIC-III,
whether to also store a clinical 24 h-worst variant alongside the 6 h one.
