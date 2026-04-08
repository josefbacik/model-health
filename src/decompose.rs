//! Decompose measured signals (stress, RHR, HRV) into a training-explained
//! component and an external (life) residual component.
//!
//! This is the **POC version** of the decomposition described in
//! `plans/stress-decomposition.md`. It is intentionally console-only:
//!   - fits an OLS linear regression for each target
//!   - prints coefficients, R², residual statistics, and the top-10
//!     highest-residual days
//!   - does NOT write a new parquet (the full version would persist
//!     `decomposed_health.parquet` for downstream consumers like races.rs)
//!
//! The point of the POC is to answer the question "is the decomposition
//! meaningful enough to invest plumbing in?" before building the full
//! pipeline. If R² is very low, training inputs barely explain the signal
//! and the residual interpretation is suspect. If R² is reasonable
//! (say >0.15), the residual is a real lower bound on "non-training-driven
//! variation" and worth feeding into race retros.

use polars::prelude::*;
use smartcore::linalg::basic::arrays::Array;
use smartcore::linear::linear_regression::LinearRegression;

use crate::config::Config;
use crate::data;
use crate::error::{AppError, Result};
use crate::model;
use crate::validation;

/// Targets to decompose. Tuple is (column name in daily_health, short label,
/// signal-baseline rolling window in days).
///
/// HRV was tried during POC iteration and dropped. With ~470 days of data
/// and a 90-day baseline window the autoregression coefficient sign-flipped
/// (v2: -0.65, fitting noise). Shrinking the baseline to 30d fixed the sign
/// but dropped R² from 0.13 → 0.06 because rolling-mean with min_periods=30
/// over a 30-day window aggressively produces nulls when the input has any
/// gaps. Conclusion: HRV doesn't have enough data for a clean decomposition,
/// and sleep — the only feature that actually moved HRV — can be inspected
/// directly via the race retro's raw HRV features.
const TARGETS: &[(&str, &str, usize)] = &[("avg_stress", "stress", 90), ("resting_hr", "rhr", 90)];

/// Per-day features. Order drives the printed coefficient table.
///
/// History: v1 included `distance_lag2_km` / `distance_lag3_km` (collinear
/// with distance_7d_km, sign-flipped). v2 added cross-training, sleep,
/// Garmin's `training_load`, and dropped lag2/lag3. v3 (this version) drops
/// `distance_lag1_km` (still slightly collinear with distance_7d_km) and
/// `training_load_today/7d` (coefficient ≈ 0.000 in v2 — too sparse to
/// contribute against the existing distance signal).
const FEATURES: &[&str] = &[
    // Running distance: today + medium/long rolling sums.
    "distance_today_km",
    "distance_7d_km",
    "distance_28d_km",
    // Cross-training (cycling, swimming, etc.).
    "cross_train_today_km",
    "cross_train_7d_km",
    // Sleep. `last_night` is row D's sleep_hours (Garmin records sleep
    // against the morning of the day it ends, so row D's value IS last
    // night). `prior_7d` is the mean of the 7 nights *before* last night
    // (shift-1-then-rolling), kept orthogonal to last_night the same way
    // distance_7d_km is kept orthogonal to distance_today_km.
    "sleep_hours_last_night",
    "sleep_hours_prior_7d",
    // Slow-moving baseline of the target itself, lagged 1 day.
    "signal_baseline",
];

/// Running activity types — used for the per-day distance aggregation. The
/// list matches `races.rs::RUNNING_TYPES` and is the user's dominant
/// training mode.
const RUNNING_TYPES: &[&str] = &["running", "treadmill_running", "virtual_run"];

/// Non-running endurance activity types whose distance contributes to the
/// "cross-training" signal. We deliberately exclude strength_training (no
/// distance) and racquetball / kayaking variants (one-off, low volume).
const CROSS_TRAIN_TYPES: &[&str] = &[
    "cycling",
    "indoor_cycling",
    "virtual_ride",
    "road_biking",
    "mountain_biking",
    "lap_swimming",
    "open_water_swimming",
    "hiking",
];

/// POC entry point. Loads data, fits OLS for each target, prints reports.
pub fn run(config: &Config) -> Result<()> {
    println!("=== Stress / RHR decomposition (POC) ===\n");
    println!(
        "Fitting an OLS linear regression for each target signal. Features:\n\
         running distance (today + 7d/28d rolling sums), cross-training distance\n\
         (today + 7d), sleep hours (last night + 7d mean), and a 90-day rolling\n\
         mean of the signal itself as a slow-moving baseline.\n\n\
         The residual = actual - predicted is the part of the signal that training\n\
         and sleep inputs cannot explain — interpret as a lower bound on 'external'\n\
         variation (life, work, illness, weather, etc.) that's left over even after\n\
         accounting for what bad sleep would predict.\n"
    );

    // --- Load + clean daily health -------------------------------------
    let daily_lf = data::load_daily_health(config)?;
    let daily_lf = validation::clean_daily_health(daily_lf)?;
    let daily = daily_lf.sort(["date"], Default::default()).collect()?;

    // --- Aggregate activities to per-day load metrics ------------------
    let acts_lf = data::load_activities(config)?;
    let acts_per_day = aggregate_activities_per_day(acts_lf)?;

    // --- Join activities into daily spine ------------------------------
    // Left join so days with no activity survive; sparse load columns become 0.
    // Also derive sleep_hours from sleep_seconds while we're here so the
    // per-target feature builder doesn't have to repeat the conversion.
    let joined = daily
        .lazy()
        .join(
            acts_per_day.lazy(),
            [col("date")],
            [col("date")],
            JoinArgs::new(JoinType::Left),
        )
        .with_columns([
            col("distance_today_km").fill_null(lit(0.0)),
            col("cross_train_today_km").fill_null(lit(0.0)),
            (col("sleep_seconds").cast(DataType::Float64) / lit(3600.0)).alias("sleep_hours"),
        ])
        .sort(["date"], Default::default())
        .collect()?;

    println!(
        "Joined daily-health × running spine: {} days\n",
        joined.height()
    );

    // --- Fit + report per target ---------------------------------------
    for (target_col, label, baseline_days) in TARGETS {
        match decompose_target(&joined, target_col, label, *baseline_days) {
            Ok(()) => {}
            Err(e) => {
                println!("--- {label} ({target_col}) ---");
                println!("  Failed: {e}\n");
            }
        }
    }

    Ok(())
}

/// Group activities by local date and produce per-day load columns:
///   - `distance_today_km`         sum of running distances
///   - `cross_train_today_km`      sum of cycling / swimming / other distances
///
/// Days with no running activity get null in `distance_today_km`; days with
/// no cross-training activity get null in `cross_train_today_km`. The caller
/// fills these with 0 after the join with the daily-health spine.
fn aggregate_activities_per_day(lf: LazyFrame) -> Result<DataFrame> {
    // Build a per-row classification: is this run, cross-train, or other?
    // Then group by date and sum each category separately. Doing it as a
    // single group_by is faster than two separate filtered groups.
    let is_run = type_in_list(RUNNING_TYPES);
    let is_cross = type_in_list(CROSS_TRAIN_TYPES);
    let dist_km = col("distance_m").cast(DataType::Float64) / lit(1000.0);

    let df = lf
        .with_column(col("start_time_local").dt().date().alias("date"))
        .with_columns([
            // Per-row contributions (zero if the row isn't in the relevant
            // category) so we can sum them by date in one pass.
            when(is_run.clone())
                .then(dist_km.clone())
                .otherwise(lit(0.0))
                .alias("_run_km"),
            when(is_cross.clone())
                .then(dist_km.clone())
                .otherwise(lit(0.0))
                .alias("_cross_km"),
        ])
        .group_by([col("date")])
        .agg([
            col("_run_km").sum().alias("distance_today_km"),
            col("_cross_km").sum().alias("cross_train_today_km"),
        ])
        .sort(["date"], Default::default())
        .collect()?;

    Ok(df)
}

/// Build an `activity_type IN (...)` filter over a const list of type strings.
/// Polars `is_in` requires a feature flag we don't currently enable, so we
/// build an OR-chain instead — matches the pattern used in `races.rs`.
fn type_in_list(types: &[&str]) -> Expr {
    types
        .iter()
        .map(|t| col("activity_type").eq(lit(*t)))
        .reduce(|a, b| a.or(b))
        .expect("types must be non-empty")
}

/// Build features for one target, fit OLS, and print the report. The
/// `baseline_days` argument controls the rolling-mean window for the
/// `signal_baseline` autoregressive feature; it's per-target because HRV
/// data is too short for a 90-day window.
fn decompose_target(
    joined: &DataFrame,
    target_col: &str,
    label: &str,
    baseline_days: usize,
) -> Result<()> {
    // Distance rolling sums use the full window (input is 0-filled on rest
    // days, so there are no input nulls — strict min_periods is fine).
    let rolling_7d_strict = RollingOptionsFixedWindow {
        window_size: 7,
        min_periods: 7,
        ..Default::default()
    };
    let rolling_28d_strict = RollingOptionsFixedWindow {
        window_size: 28,
        min_periods: 28,
        ..Default::default()
    };
    // Sleep rolling means need a relaxed min_periods because raw sleep data
    // has occasional gaps (no-wear days, validation null-outs). With strict
    // 7-of-7, a single missing night kills 7 consecutive rows. The
    // `features.rs` precedent uses `min_periods: 5` for the same reason.
    let rolling_7d_sleep = RollingOptionsFixedWindow {
        window_size: 7,
        min_periods: 5,
        ..Default::default()
    };
    let rolling_baseline = RollingOptionsFixedWindow {
        window_size: baseline_days,
        min_periods: baseline_days,
        ..Default::default()
    };

    let with_features = joined
        .clone()
        .lazy()
        .with_columns([
            // --- Running distance ------------------------------------------
            col("distance_today_km")
                .shift(lit(1))
                .rolling_sum(rolling_7d_strict.clone())
                .alias("distance_7d_km"),
            col("distance_today_km")
                .shift(lit(1))
                .rolling_sum(rolling_28d_strict)
                .alias("distance_28d_km"),
            // --- Cross-training (cycling, swimming, etc.) ------------------
            col("cross_train_today_km")
                .shift(lit(1))
                .rolling_sum(rolling_7d_strict)
                .alias("cross_train_7d_km"),
            // --- Sleep -----------------------------------------------------
            // Garmin records sleep against the morning the sleep ends, so
            // sleep_hours at row D == "the night that just ended on day D"
            // == "last night" for predicting day-D's stress. No shift needed.
            col("sleep_hours").alias("sleep_hours_last_night"),
            // The 7-day mean is shifted by 1 so it covers the 7 nights
            // *before* last night — kept orthogonal to last_night, mirroring
            // how distance_7d_km is orthogonal to distance_today_km.
            col("sleep_hours")
                .shift(lit(1))
                .rolling_mean(rolling_7d_sleep)
                .alias("sleep_hours_prior_7d"),
            // --- Slow-moving baseline of the target ------------------------
            // Rolling mean of the signal itself, lagged 1 day to avoid
            // leaking the target value into its own predictor. Absorbs slow
            // baseline drift (fitness, weight loss, etc.).
            col(target_col)
                .cast(DataType::Float64)
                .shift(lit(1))
                .rolling_mean(rolling_baseline)
                .alias("signal_baseline"),
        ])
        .collect()?;

    let pre_drop_rows = with_features.height();

    // Drop rows missing the target or any feature.
    let mut required: Vec<Expr> = FEATURES.iter().map(|f| col(*f)).collect();
    required.push(col(target_col));
    let clean = with_features.lazy().drop_nulls(Some(required)).collect()?;

    let n = clean.height();
    let dropped = pre_drop_rows.saturating_sub(n);
    println!("--- {label} ({target_col}) ---");
    println!("  Rows: {n} fit  ({dropped} dropped from {pre_drop_rows} for null target/features)");
    // Hard-coded floor for stable OLS — 100 rows × 8 features is ~12
    // observations per feature, the conventional rule of thumb. The
    // existing config.min_training_days (60) is for the next-day model
    // and isn't strict enough for this fit, so the threshold lives here.
    if n < 100 {
        println!("  Not enough rows for stable decomposition (need ≥100). Skipping.\n");
        return Ok(());
    }

    // Build (X, y) via the shared matrix builder in model.rs. Allocates a
    // small Vec<String> from the &'static FEATURES list once per fit.
    let feature_names: Vec<String> = FEATURES.iter().map(|s| s.to_string()).collect();
    let x = model::dataframe_to_matrix(&clean, &feature_names)?;
    let y: Vec<f64> = clean
        .column(target_col)?
        .cast(&DataType::Float64)?
        .f64()?
        .into_no_null_iter()
        .collect();
    // Defensive: target is in `required` above so drop_nulls guarantees y
    // has no nulls. This check is belt-and-suspenders against future
    // refactors that might change which columns are dropped.
    if y.len() != n {
        return Err(AppError::Model(format!(
            "{target_col}: target column had nulls after drop_nulls (n={n}, y={})",
            y.len()
        )));
    }

    let model = LinearRegression::fit(&x, &y, Default::default())
        .map_err(|e| AppError::Model(format!("OLS fit failed: {e}")))?;

    let preds = model
        .predict(&x)
        .map_err(|e| AppError::Model(format!("OLS predict failed: {e}")))?;
    let residuals: Vec<f64> = y.iter().zip(preds.iter()).map(|(a, p)| a - p).collect();

    // R² and residual stats.
    let mean_y = y.iter().sum::<f64>() / n as f64;
    let ss_res: f64 = residuals.iter().map(|r| r * r).sum();
    let ss_tot: f64 = y.iter().map(|v| (v - mean_y).powi(2)).sum();
    let r_squared = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };
    let mae: f64 = residuals.iter().map(|r| r.abs()).sum::<f64>() / n as f64;
    let max_pos = residuals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let max_neg = residuals.iter().copied().fold(f64::INFINITY, f64::min);

    println!(
        "  R²:              {:.3}  ({:.0}% of variance explained by training inputs)",
        r_squared,
        r_squared * 100.0
    );
    println!("  Target mean:     {mean_y:.2}");
    println!("  Residual MAE:    {mae:.2}");
    println!("  Residual range:  {max_neg:+.2} to {max_pos:+.2}");
    println!("  Intercept:       {:>+10.4}", *model.intercept());
    println!("  Coefficients (target units per unit feature):");
    let coefs = model.coefficients();
    for (i, feat) in FEATURES.iter().enumerate() {
        let v = *coefs.get((i, 0));
        println!("    {feat:<22} {v:>+10.4}");
    }

    // Top 10 highest residual days = largest unexplained-by-training values.
    // For stress/RHR these are days where the signal was *higher* than
    // training would predict — interpretation: external factor (life,
    // illness, weather) pushed it up.
    print_top_residuals(&clean, &y, &residuals, 10);
    println!();
    Ok(())
}

/// Print the top-`n` highest residual days. These are the days where the
/// signal was furthest *above* what training inputs predicted — the
/// canonical "unexplained" days, which the user may recognize as known
/// stressful events (work travel, illness, family stress) and can use to
/// validate that the residual is capturing what we think it's capturing.
fn print_top_residuals(df: &DataFrame, y: &[f64], residuals: &[f64], n: usize) {
    println!("  Top {n} highest-residual days (signal > training prediction):");
    println!(
        "    {:<12} {:>9} {:>11} {:>11}",
        "date", "actual", "predicted", "residual"
    );

    let mut indexed: Vec<(usize, f64)> = residuals.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (idx, res) in indexed.iter().take(n) {
        let actual = y[*idx];
        let predicted = actual - res;
        let date = model::date_at(df, *idx)
            .map(|d| d.to_string())
            .unwrap_or_else(|_| "?".into());
        println!("    {date:<12} {actual:>9.1} {predicted:>11.1} {res:>+11.2}");
    }
}
