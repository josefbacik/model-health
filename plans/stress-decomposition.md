# Plan: decomposing measured signals into training vs. external components

> **Status:** design only, not yet implemented. Sidetracked from race-retro work — pick this up when ready.

## The problem

Garmin's `avg_stress` (and to a lesser extent `resting_hr` and `hrv_last_night`) is a single measurement that conflates two very different sources:

1. **Training stress** — the physiological cost of recent workouts. Hard runs raise your daily stress, raise your RHR for a day or two, and depress HRV.
2. **External / life stress** — work, family, sleep deprivation, illness, travel, weather. Same physiological signature.

These have opposite implications for race readiness. *High training + low external* means you're working hard and recovering well. *Low training + high external* means life is interfering with training. Garmin's number can't tell them apart, but the surrounding data can.

This came out of looking at the race retro and noticing that the marathon bucket showed "higher stress correlated with faster races" — almost certainly because Vancouver had a heavy training block (high training stress) while the recent 2026 marathons were relatively undertrained but in a calmer life period. The raw stress number can't see that distinction.

## The hypothesis we want to test

Once decomposed, **good races correlate with high training-stress + low external-stress, and bad races correlate with similar/lower training-stress but high external-stress**. The race-retro tool can test this directly once the decomposed columns exist.

## The conceptual model

```
measured_signal_t  =  baseline  +  training_component_t  +  external_component_t  +  noise
```

We don't observe the components, only the sum. But we *do* observe the inputs that drive `training_component`: yesterday's mileage, training load, rolling volume, etc. So:

1. Fit a model that predicts the measured signal from **training inputs only**.
2. The model's prediction = estimate of `baseline + training_component`.
3. The model's residual (actual − predicted) = estimate of `external_component`.

For a high-stress day after a hard workout, the residual should be small (training explains it). For a high-stress day after a rest day, the residual should be large — that's life talking.

## Why this is feasible with the data we have

| signal | days available | usable years |
|---|---|---|
| `avg_stress` | ~2,250 (33% null overall, but 95-100% from 2020 on) | 2020-01-27 onward |
| `resting_hr` | ~3,036 | 2017 onward |
| `hrv_last_night` | ~470 | 2024-12-26 onward |

Stress and RHR have plenty of data for a real model. HRV is thinner but still ~470 days, which is enough for a small linear fit.

Garmin also publishes `training_load` per workout (in the activities parquet) and `training_load` daily (in performance metrics) — so we have an existing input feature without needing to invent one.

## Gotchas (these don't kill the approach but shape interpretation)

1. **Reverse causation in training itself.** Life stress causes you to skip workouts, so `training_load` is partially *caused by* external stress. A model that "controls for training" will under-attribute to external stress and over-attribute to training. Bias is in a known direction → residuals are a *lower bound* on external stress, not an exact estimate. Honest framing in the report.

2. **Lag matters a lot.** Stress cost of yesterday's long run shows up today, peaks 24-48h later for some workouts. Features must include lagged training load (`lag1`, `lag2`, `lag3`), not just same-day. HRV is similarly delayed; RHR is more acute (same day or +1).

3. **Personal baseline drift.** RHR baseline shifts over years as fitness changes. Without modeling this, residuals will trend with fitness rather than just isolating life stress. Fix: include a slow-moving baseline (90-day rolling mean of the signal itself, lagged out so it doesn't leak the target) as one of the inputs, OR fit on rolling windows.

4. **Non-training causes that look training-like.** Hot weather, sickness, travel — all raise stress/RHR independently of effort. The decomposition won't distinguish "stressed about work" from "fighting a cold"; both go in the residual. Acceptable for the user's purposes.

5. **Linear vs. non-linear.** Stress probably doesn't respond linearly to training (a 20km run isn't 2× the stress of a 10km run). Linear is interpretable (slope = "exchange rate"), random forest is more accurate but opaque. **Recommendation: start linear.** The whole point is to *understand* the decomposition, not maximize predictive accuracy.

## Concrete first build

### New module: `src/decompose.rs`

### Inputs
- `daily_health` (target signals: `avg_stress`, `resting_hr`, `hrv_last_night`)
- `activities` (per-workout `training_load`, `duration_sec`, `distance_m`)
- `performance_metrics` (daily `training_load` if available — Garmin's own rollup)

### Per-day training feature set
| feature | description |
|---|---|
| `train_load_today` | sum of today's activities' `training_load` (0 on rest days) |
| `train_load_lag1` | same, 1 day back |
| `train_load_lag2` | same, 2 days back |
| `train_load_lag3` | same, 3 days back |
| `train_distance_lag1` | running distance, 1 day back |
| `train_distance_lag2` | running distance, 2 days back |
| `train_load_7d` | 7-day rolling sum |
| `train_load_28d` | 28-day rolling sum (captures fitness state) |
| `signal_baseline_90d` | 90-day rolling mean of the *target* signal, lagged 1 day to avoid leakage (absorbs long-term drift) |

### Per-target model
- One linear regression each for `avg_stress`, `resting_hr`, `hrv_last_night`.
- Fit on all days where the target is non-null.
- Use time-series cross-validation (similar shape to `model::time_series_cv`) to estimate honest error and avoid leakage from future days.
- Ordinary least squares is ~30 lines of Rust; not worth pulling in a new dependency for it.

### Outputs

**A new parquet file** `decomposed_health.parquet` (sibling of the others), columns:
- `date`
- `avg_stress`, `stress_predicted_from_training`, `stress_external`
- `resting_hr`, `rhr_predicted_from_training`, `rhr_external`
- `hrv_last_night`, `hrv_predicted_from_training`, `hrv_external`

(`*_external` = measured − predicted.)

**A console report** that prints, per target:
- Coefficients (which training features matter most, with units)
- R² (how much variance training inputs actually explain — this is the key honesty metric: if training only explains 20% of stress variance, the other 80% is "everything else" and that's load-bearing)
- Distribution of residuals (mean, std, max)
- A few examples: "biggest external stress days" — high actual stress + low predicted stress, sorted

### Integration with race retro

Add new features in `compute_race_features`:
- `stress_external_28d`, `stress_training_28d`, `stress_external_7d`, `stress_training_7d`
- `rhr_external_28d`, `rhr_training_28d`
- `hrv_external_28d`, `hrv_training_28d`

These join from `decomposed_health.parquet` on date. The contrast and correlation reports pick them up automatically — no code changes needed there.

This is the moment the user's hypothesis becomes directly testable: the contrast table will show good races vs bad races with the training/external split visible.

## CLI shape

```sh
# fit the decomposition models and write decomposed_health.parquet
model-health decompose

# fit + verbose report (print coefficients, residual distribution, top external-stress days)
model-health decompose --report
```

Run as a separate step (not on every train/predict) because the decomposition is a one-shot fit that's only re-needed after new data is fetched.

## Open design questions

1. **Linear vs. RF**: leaning linear for interpretability, but worth measuring how much variance linear leaves on the table. If R² < 0.15 on stress, may need more flexibility. Easy to add later.
2. **How to reuse `model.rs` infrastructure**: probably write OLS from scratch (it's tiny) and reuse the time-series CV / metric helpers. Don't add `LinearModel` as a parallel target type — keep `model.rs` focused on the next-day forecasting use case.
3. **Where to put decomposed columns**: separate parquet (`decomposed_health.parquet`) vs. extending `daily_health.parquet`. **Recommendation: separate parquet.** Keeps raw fetched data clean and avoids re-deriving on every fetch.
4. **Should `decompose` be invoked automatically as part of `train`?** Probably no — keep them separate so the user can iterate on the decomposition independently of next-day prediction work.
5. **Validation**: how do we know the decomposition is "right"? We can't directly, since we never observe the true components. Sanity checks: residuals should be larger on known stressful weeks (work travel, illness) and smaller on calm weeks. Could ask the user to spot-check: print the top-10 highest-residual stress days and ask "do you remember anything happening here?"

## Estimated scope

- New module ~300-400 lines
- New CLI subcommand ~10 lines
- Race-retro feature additions ~50 lines (in `compute_race_features`)
- Plus tests / fixtures: TBD

Big enough to warrant a focused session. Worth doing because:
- It's the foundation for the broader "what drives my stress / sleep / RHR" question from the original brainstorm.
- It cleanly resolves the Vancouver-marathon paradox (high training stress + low life stress → fastest race) once we can see the decomposition.
- It generalizes: the same residual approach could later be used to ask "which days are sleep-debt days vs. training-fatigue days?"

## Smaller alternative if the full plan feels too big

A console-only POC that:
- Fits the linear models in-memory
- Doesn't write any new parquet
- Just prints the coefficients, R², and the top-10 highest-residual stress days

That answers "is the decomposition meaningful enough to invest plumbing in" before committing to the full file/integration work. ~150 lines, no persistence layer, no race-retro integration.

Decide between full and POC at the time we pick this up.
