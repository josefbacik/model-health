//! Decompose measured signals (stress, RHR) into a training-explained
//! component and an external (life) residual component.
//!
//! For each target signal, fits an OLS linear regression on per-day
//! training and sleep features, prints a console report (coefficients, R²,
//! residual statistics, top-10 highest-residual days), and writes per-day
//! predictions and residuals to `data_dir/decomposed_health.parquet`.
//!
//! The downstream consumer is `races.rs`, which reads the parquet and
//! computes per-race trailing-window features like `stress_external_28d`
//! (mean of the stress residual over the 28 days before a race). This
//! lets the race retro separate "stress from training" from "stress from
//! life" when comparing good vs bad races.
//!
//! HRV was tried as a third target during POC iteration and dropped — see
//! the comment on `TARGETS` for why.

use std::path::PathBuf;

use polars::prelude::*;
use smartcore::linalg::basic::arrays::Array;
use smartcore::linear::linear_regression::LinearRegression;
use tracing::info;

use crate::config::Config;
use crate::data;
use crate::error::{AppError, Result};
use crate::model;
use crate::validation;

/// One target's complete fit: the cleaned input frame, the actual values,
/// the OLS predictions, and the residuals. Carried out of `fit_target` so
/// that `print_target_report` can render the report and `save_decomposed`
/// can persist per-day predictions/residuals to parquet.
struct TargetFit {
    /// Garmin column name being decomposed (e.g. "avg_stress").
    target_col: String,
    /// Short label used in reports and parquet column prefixes ("stress").
    label: String,
    /// The cleaned, drop-nullsed feature DataFrame the OLS was fit on.
    /// Carries the `date` column so per-row predictions can be joined
    /// back to a date axis when persisting.
    clean: DataFrame,
    /// Target values (length == clean.height()).
    y: Vec<f64>,
    /// OLS predictions on `clean` (same length as `y`).
    preds: Vec<f64>,
    /// Residuals = y - preds.
    residuals: Vec<f64>,
    /// Coefficients in the same order as FEATURES.
    coefficients: Vec<f64>,
    /// OLS intercept.
    intercept: f64,
    /// Pre-drop / post-drop row counts, for the printed report.
    pre_drop_rows: usize,
    /// Coefficient of determination on the training set.
    r_squared: f64,
    /// MAE of residuals.
    mae: f64,
    /// Most-positive and most-negative residuals.
    max_pos_residual: f64,
    max_neg_residual: f64,
}

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
const TARGETS: &[(&str, &str)] = &[("avg_stress", "stress"), ("resting_hr", "rhr")];

/// Per-day features. Order drives the printed coefficient table.
///
/// History: v1 included `distance_lag2_km` / `distance_lag3_km` (collinear
/// with distance_7d_km, sign-flipped). v2 added cross-training, sleep,
/// Garmin's `training_load`, and dropped lag2/lag3. v3 dropped
/// `distance_lag1_km` and `training_load_today/7d`. v4 tried ACWR (both
/// 7d/28d and 28d/90d) but it was collinear with the component distance
/// features and sign-flipped. v5 (this version) replaces `signal_baseline`
/// (90-day rolling mean of the target signal) with `distance_90d_km`
/// (90-day rolling sum of running distance) and adds `distance_7d_km_sq`
/// for non-linear acute-load response. signal_baseline mixed training and
/// life stress indistinguishably — its 0.85 coefficient dominated the
/// model and caused it to attribute Vancouver's life stress to "training"
/// (hiding it in a negative residual). The distance-based baseline ties
/// the slow-drift proxy purely to training volume, so the residual now
/// correctly identifies high-life-stress periods (Vancouver: +5.7,
/// Richmond: +5.3) vs low-life-stress periods (Cary PR: +2.1).
const FEATURES: &[&str] = &[
    // Running distance: today + short/medium/long rolling sums.
    //
    // `distance_7d_km_sq` captures the non-linear acute-load effect: a big
    // week (80 km) produces ~8.6 more stress points than a normal week
    // (40 km) beyond what the linear term predicts. `distance_28d_km_sq`
    // was tried and dropped — its coefficient was ~0, negligible at any
    // realistic 4-week volume.
    //
    // `distance_90d_km` replaces the old `signal_baseline` (90-day rolling
    // mean of the target signal itself). signal_baseline absorbed slow drift
    // from BOTH training and life stress, so the model couldn't distinguish
    // them — e.g. Vancouver's elevated life stress inflated the baseline,
    // causing the model to over-predict training stress and hide the life
    // stress in a negative residual. Using a 90-day distance sum instead
    // ties the baseline proxy to training volume only, so the residual
    // reflects genuine non-training variation.
    "distance_today_km",
    "distance_7d_km",
    "distance_28d_km",
    "distance_7d_km_sq",
    "distance_90d_km",
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
    println!("=== Stress / RHR decomposition ===\n");
    println!(
        "Fitting an OLS regression for each target signal. Features:\n\
         running distance (today + 7d/28d/90d rolling sums + squared terms for\n\
         non-linear volume response), cross-training distance (today + 7d),\n\
         and sleep hours (last night + 7d mean).\n\n\
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
    let mut fits: Vec<TargetFit> = Vec::new();
    for (target_col, label) in TARGETS {
        match fit_target(&joined, target_col, label) {
            Ok(Some(fit)) => {
                print_target_report(&fit);
                fits.push(fit);
            }
            Ok(None) => {
                // Too few rows; the fit_target body printed why.
            }
            Err(e) => {
                println!("--- {label} ({target_col}) ---");
                println!("  Failed: {e}\n");
            }
        }
    }

    // --- Persist per-day predictions + residuals to parquet -----------
    if fits.is_empty() {
        println!("No targets fit successfully — nothing to save.");
    } else {
        let path = decomposed_health_path(config);
        save_decomposed(&fits, &path)?;
        println!("Saved decomposed_health.parquet to {}", path.display());
    }

    Ok(())
}

/// Path to the parquet that holds per-day predictions and residuals. Lives
/// under `data_dir` rather than `garmin_storage_path` because it's *derived*
/// from raw data, not fetched.
fn decomposed_health_path(config: &Config) -> PathBuf {
    config.data_dir.join("decomposed_health.parquet")
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

/// Build features for one target and fit OLS. Returns `Some(TargetFit)` on
/// success or `None` if there are too few rows for a stable fit (in which
/// case the function has already printed the "skipping" message).
///
/// Doing the fit here without printing lets `run` collect every successful
/// fit so they can be persisted to a single parquet, while keeping the
/// printed report driven by `print_target_report` from the same struct.
fn fit_target(joined: &DataFrame, target_col: &str, label: &str) -> Result<Option<TargetFit>> {
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
    let rolling_90d_strict = RollingOptionsFixedWindow {
        window_size: 90,
        min_periods: 90,
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
            // --- 90-day rolling sum: training-volume baseline ---------------
            // Replaces the old `signal_baseline` (90-day mean of the target
            // signal), which mixed training and life stress indistinguishably.
            // Using distance instead ties the baseline proxy purely to
            // training volume, so the residual captures life stress cleanly.
            col("distance_today_km")
                .shift(lit(1))
                .rolling_sum(rolling_90d_strict)
                .alias("distance_90d_km"),
            // --- Cross-training (cycling, swimming, etc.) ------------------
            col("cross_train_today_km")
                .shift(lit(1))
                .rolling_sum(rolling_7d_strict)
                .alias("cross_train_7d_km"),
        ])
        .with_columns([
            // --- Squared acute-load term ------------------------------------
            // Captures non-linear stress response to big training weeks.
            (col("distance_7d_km") * col("distance_7d_km")).alias("distance_7d_km_sq"),
        ])
        .with_columns([
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
        ])
        .collect()?;

    let pre_drop_rows = with_features.height();

    // Drop rows missing the target or any feature.
    let mut required: Vec<Expr> = FEATURES.iter().map(|f| col(*f)).collect();
    required.push(col(target_col));
    let clean = with_features.lazy().drop_nulls(Some(required)).collect()?;

    let n = clean.height();
    // Hard-coded floor for stable OLS — 100 rows × 8 features is ~12
    // observations per feature, the conventional rule of thumb. The
    // existing config.min_training_days (60) is for the next-day model
    // and isn't strict enough for this fit, so the threshold lives here.
    if n < 100 {
        println!("--- {label} ({target_col}) ---");
        let dropped = pre_drop_rows.saturating_sub(n);
        println!(
            "  Rows: {n} fit  ({dropped} dropped from {pre_drop_rows} for null target/features)"
        );
        println!("  Not enough rows for stable decomposition (need ≥100). Skipping.\n");
        return Ok(None);
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

    // Pull coefficients out of the smartcore model into a Vec so the
    // TargetFit doesn't borrow from a model owned by this stack frame.
    let coefs_matrix = model.coefficients();
    let coefficients: Vec<f64> = (0..FEATURES.len())
        .map(|i| *coefs_matrix.get((i, 0)))
        .collect();
    let intercept = *model.intercept();

    Ok(Some(TargetFit {
        target_col: target_col.to_string(),
        label: label.to_string(),
        clean,
        y,
        preds,
        residuals,
        coefficients,
        intercept,
        pre_drop_rows,
        r_squared,
        mae,
        max_pos_residual: max_pos,
        max_neg_residual: max_neg,
    }))
}

/// Render the per-target report from a completed fit. Pulled out of
/// `fit_target` so the same struct can drive both the printed report and
/// the parquet save.
fn print_target_report(fit: &TargetFit) {
    let n = fit.y.len();
    let dropped = fit.pre_drop_rows.saturating_sub(n);
    let mean_y = fit.y.iter().sum::<f64>() / n as f64;

    println!("--- {} ({}) ---", fit.label, fit.target_col);
    println!(
        "  Rows: {n} fit  ({dropped} dropped from {} for null target/features)",
        fit.pre_drop_rows
    );
    println!(
        "  R²:              {:.3}  ({:.0}% of variance explained by training inputs)",
        fit.r_squared,
        fit.r_squared * 100.0
    );
    println!("  Target mean:     {mean_y:.2}");
    println!("  Residual MAE:    {:.2}", fit.mae);
    println!(
        "  Residual range:  {:+.2} to {:+.2}",
        fit.max_neg_residual, fit.max_pos_residual
    );
    println!("  Intercept:       {:>+10.4}", fit.intercept);
    println!("  Coefficients (target units per unit feature):");
    for (feat, v) in FEATURES.iter().zip(fit.coefficients.iter()) {
        println!("    {feat:<22} {v:>+10.4}");
    }
    print_top_residuals(&fit.clean, &fit.y, &fit.residuals, 10);
    println!();
}

/// Persist per-day actual / predicted / residual values for every successful
/// target fit to a single parquet file. Per-target columns are named with the
/// short label as a prefix:
///
///   date | stress | stress_training | stress_external | rhr | rhr_training | rhr_external
///
/// Different targets may have different fit row sets (drop_nulls drops on
/// per-target features) so the per-fit frames are full-outer-joined on `date`.
/// Days that one target couldn't fit get null in that target's columns.
fn save_decomposed(fits: &[TargetFit], path: &std::path::Path) -> Result<()> {
    if fits.is_empty() {
        return Err(AppError::Model(
            "save_decomposed called with no fits".into(),
        ));
    }

    // Build one (date, actual, predicted, external) DataFrame per target.
    let per_target: Vec<DataFrame> = fits
        .iter()
        .map(build_per_fit_frame)
        .collect::<Result<Vec<_>>>()?;

    // Full-outer-join them all on `date`. Start with the first frame; outer-
    // join each subsequent one. Polars outer joins produce a `date_right`
    // column that we coalesce into `date` after each step.
    let mut joined = per_target[0].clone().lazy();
    for next in per_target.iter().skip(1) {
        joined = joined.join(
            next.clone().lazy(),
            [col("date")],
            [col("date")],
            JoinArgs::new(JoinType::Full).with_coalesce(JoinCoalesce::CoalesceColumns),
        );
    }
    let mut combined = joined.sort(["date"], Default::default()).collect()?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Atomic write via .tmp + rename — same pattern as the fetcher.
    let tmp = path.with_extension("parquet.tmp");
    {
        let mut file = std::fs::File::create(&tmp)?;
        ParquetWriter::new(&mut file)
            .with_compression(ParquetCompression::Zstd(None))
            .finish(&mut combined)?;
    }
    std::fs::rename(&tmp, path)?;

    info!(
        rows = combined.height(),
        targets = fits.len(),
        path = %path.display(),
        "Wrote decomposed_health.parquet"
    );
    Ok(())
}

/// Build the per-target slice of the decomposed parquet: a DataFrame with
/// columns `date`, `{label}`, `{label}_predicted`, `{label}_external`.
fn build_per_fit_frame(fit: &TargetFit) -> Result<DataFrame> {
    // Extract the date column as a Series and clone it; everything else is
    // built from the parallel y / preds / residuals Vecs.
    let date_series = fit.clean.column("date")?.clone();

    let actual_name = fit.label.clone();
    // Column naming: `{label}_training` for the OLS-predicted (training-
    // explained) component, `{label}_external` for the residual. Using
    // `_training` rather than `_predicted` so the column names align with
    // the downstream race features (`stress_training_28d`, etc.) without
    // a naming translation layer.
    let training_name = format!("{}_training", fit.label);
    let external_name = format!("{}_external", fit.label);

    let actual = Series::new(actual_name.as_str().into(), &fit.y);
    let training = Series::new(training_name.as_str().into(), &fit.preds);
    let external = Series::new(external_name.as_str().into(), &fit.residuals);

    DataFrame::new(vec![
        date_series,
        actual.into(),
        training.into(),
        external.into(),
    ])
    .map_err(|e| AppError::Model(format!("build_per_fit_frame for {}: {e}", fit.label)))
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
