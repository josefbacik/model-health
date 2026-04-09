# Plan: decomposing measured signals into training vs. external components

> **Status:** implemented + plumbed into race retro. See `src/decompose.rs`.
>
> **v5 resolution of marathon-PR outlier:** The original `signal_baseline`
> feature (90-day rolling mean of the target signal) mixed training and life
> stress indistinguishably — its 0.85 coefficient dominated the model and
> caused it to attribute Vancouver's known life stress to "training" (hiding
> it in a negative residual: external_28d = -3.2). Replacing
> `signal_baseline` with `distance_90d_km` (90-day rolling sum of running
> distance) ties the baseline proxy purely to training volume. Combined with
> `distance_7d_km_sq` for non-linear acute-load response, the decomposition
> now correctly identifies high-life-stress periods across all five marathons:
>
> | Race | External_28d | 4wk km | Notes |
> |------|-------------|--------|-------|
> | Vancouver | +5.7 | 110 | Known high life stress (confirmed) |
> | Richmond | +5.3 | 184 | Elevated life stress |
> | Cary (PR) | +2.1 | 248 | Low life stress, high training |
> | Tobacco Road | -5.9 | 98 | Calm period |
> | Disney | -8.7 | 85 | Very calm (2 weeks off work) |
>
> Tradeoff: R² dropped from 29% → 13% for stress because signal_baseline
> was a powerful autoregressive predictor. The lower R² means more variance
> lands in the residual, but the residual now measures what we actually want:
> non-training stress. The reverse-causation caveat still applies (residuals
> are a lower bound on external stress).

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

3. **Personal baseline drift.** RHR baseline shifts over years as fitness changes. Without modeling this, residuals will trend with fitness rather than just isolating life stress. The original fix (v1-v4) used `signal_baseline` — a 90-day rolling mean of the target signal itself — but this mixed training and life stress indistinguishably (see v5 status block above). v5 handles baseline drift via `distance_90d_km`, which tracks slow training-volume drift without absorbing life stress into the baseline.

4. **Non-training causes that look training-like.** Hot weather, sickness, travel — all raise stress/RHR independently of effort. The decomposition won't distinguish "stressed about work" from "fighting a cold"; both go in the residual. Acceptable for the user's purposes.

5. **Linear vs. non-linear.** Stress probably doesn't respond linearly to training (a 20km run isn't 2× the stress of a 10km run). Linear is interpretable (slope = "exchange rate"), random forest is more accurate but opaque. **Recommendation: start linear.** The whole point is to *understand* the decomposition, not maximize predictive accuracy.

## Concrete first build

### New module: `src/decompose.rs`

### Inputs
- `daily_health` (target signals: `avg_stress`, `resting_hr`)
- `activities` (per-workout `distance_m`, `activity_type`)

### Per-day training feature set (v5 — current)
| feature | description |
|---|---|
| `distance_today_km` | sum of today's running distances (0 on rest days) |
| `distance_7d_km` | 7-day rolling sum of running distance (shift-1) |
| `distance_28d_km` | 28-day rolling sum of running distance (shift-1) |
| `distance_7d_km_sq` | squared 7d sum — captures non-linear acute-load stress response |
| `distance_90d_km` | 90-day rolling sum of running distance (shift-1) — replaces `signal_baseline` to avoid mixing training and life stress in the baseline proxy |
| `cross_train_today_km` | sum of today's cycling/swimming/hiking distances |
| `cross_train_7d_km` | 7-day rolling sum of cross-training (shift-1) |
| `sleep_hours_last_night` | sleep from the night that just ended (Garmin convention) |
| `sleep_hours_prior_7d` | mean of 7 nights before last night (shift-1-then-rolling) |

### Per-target model
- One OLS linear regression each for `avg_stress` and `resting_hr`.
- HRV was tried and dropped (~470 days insufficient for stable fit).
- Fit on all days where the target and all features are non-null.
- Uses smartcore's `LinearRegression::fit`.

### Outputs

**A new parquet file** `decomposed_health.parquet` (under `data_dir`), columns:
- `date`
- `stress`, `stress_training`, `stress_external`
- `rhr`, `rhr_training`, `rhr_external`

(`*_training` = OLS prediction, `*_external` = measured − predicted.)

**A console report** that prints, per target:
- Coefficients (which training features matter most, with units)
- R² (how much variance training inputs explain)
- Distribution of residuals (MAE, range)
- Top-10 highest-residual days (biggest "external" stress events)

### Integration with race retro

`races.rs` reads `decomposed_health.parquet` and computes per-race features:
- `stress_external_7d`, `stress_external_28d`, `stress_training_7d`, `stress_training_28d`
- `rhr_external_7d`, `rhr_external_28d`

These appear automatically in the contrast and correlation tables.

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
