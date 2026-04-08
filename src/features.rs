use polars::prelude::*;
use tracing::{info, warn};

use crate::config::Config;
use crate::data;
use crate::error::{AppError, Result};
use crate::validation;

/// All targets the training pipeline supports, mapped to (source_column, shift).
/// Shift is applied with `col(source).shift(shift)` so a shift of `-1` means
/// "row N's target is row N+1's source value" — i.e. predicting tomorrow.
///
/// Targets must follow the `next_day_<source>` naming convention: the
/// persistence baseline in `model::time_series_cv` strips the `next_day_`
/// prefix to find the source column. If you add a target that breaks this
/// convention, the persistence baseline will silently be skipped.
pub fn supported_targets() -> &'static [(&'static str, &'static str, i64)] {
    &[
        ("next_day_resting_hr", "resting_hr", -1),
        ("next_day_sleep_hours", "sleep_hours", -1),
        ("next_day_steps", "steps", -1),
        ("next_day_hrv", "hrv_last_night", -1),
    ]
}

fn lookup_target(target: &str) -> Result<(&'static str, i64)> {
    for (name, src, shift) in supported_targets() {
        if *name == target {
            return Ok((*src, *shift));
        }
    }
    let names: Vec<&str> = supported_targets().iter().map(|(n, _, _)| *n).collect();
    Err(AppError::Data(format!(
        "Unknown target '{target}'. Supported targets: {}",
        names.join(", ")
    )))
}

/// Try to load a LazyFrame from a loader. Returns None if the data isn't
/// available — callers should treat that as "skip this source" rather than
/// failing the whole pipeline. Different datasets are fetched independently.
fn try_load<F>(name: &str, loader: F) -> Option<LazyFrame>
where
    F: FnOnce() -> Result<LazyFrame>,
{
    match loader() {
        Ok(lf) => Some(lf),
        Err(e) => {
            warn!(source = name, error = %e, "Skipping unavailable data source");
            None
        }
    }
}

/// Build the features-only LazyFrame (no target column) plus the list of
/// feature column names produced. This is the shared core of the public
/// `build_feature_matrix` and `build_prediction_features` functions.
///
/// The feature list is dynamic: optional sources (performance, activities,
/// weight, BP) only contribute columns when their parquet directories exist.
fn build_features_lazy(config: &Config) -> Result<(LazyFrame, Vec<String>)> {
    info!("Building feature matrix");

    // --- Daily health spine -------------------------------------------------
    let health_lf = data::load_daily_health(config)?;
    let before = health_lf.clone().collect()?;
    let cleaned_lf = validation::clean_daily_health(health_lf)?;
    let after = cleaned_lf.clone().collect()?;
    validation::log_cleaning_diff(&before, &after);

    let mut lf = cleaned_lf.sort(["date"], Default::default()).with_columns([
        (col("sleep_seconds").cast(DataType::Float64) / lit(3600.0)).alias("sleep_hours"),
        (col("body_battery_end") - col("body_battery_start"))
            .cast(DataType::Float64)
            .alias("body_battery_delta"),
        col("resting_hr")
            .cast(DataType::Float64)
            .alias("resting_hr"),
        col("steps").cast(DataType::Float64).alias("steps"),
        col("active_calories")
            .cast(DataType::Float64)
            .alias("active_calories"),
        col("avg_stress")
            .cast(DataType::Float64)
            .alias("avg_stress"),
        col("hrv_last_night")
            .cast(DataType::Float64)
            .alias("hrv_last_night"),
        col("avg_spo2").cast(DataType::Float64).alias("avg_spo2"),
        col("sleep_score")
            .cast(DataType::Float64)
            .alias("sleep_score"),
    ]);

    // Track which feature columns we'll keep. Start with the daily-health
    // features that we know exist after cleaning.
    let mut feat_cols: Vec<String> = vec![
        "resting_hr".into(),
        "sleep_hours".into(),
        "steps".into(),
        "active_calories".into(),
        "avg_stress".into(),
        "body_battery_delta".into(),
        "hrv_last_night".into(),
        "avg_spo2".into(),
        "sleep_score".into(),
    ];

    // --- Day-of-week cyclical encoding -------------------------------------
    lf = lf.with_columns([
        (col("date").dt().weekday().cast(DataType::Float64)
            * lit(2.0 * std::f64::consts::PI / 7.0))
        .sin()
        .alias("day_sin"),
        (col("date").dt().weekday().cast(DataType::Float64)
            * lit(2.0 * std::f64::consts::PI / 7.0))
        .cos()
        .alias("day_cos"),
    ]);
    feat_cols.push("day_sin".into());
    feat_cols.push("day_cos".into());

    // --- Performance metrics (LEFT JOIN on date) ---------------------------
    if let Some(perf_lf) = try_load("performance_metrics", || {
        data::load_performance_metrics(config)
    }) {
        let perf_features = ["vo2max", "training_readiness", "endurance_score"];
        let perf_select: Vec<Expr> = std::iter::once(col("date"))
            .chain(
                perf_features
                    .iter()
                    .map(|c| col(*c).cast(DataType::Float64).alias(*c)),
            )
            .collect();
        let perf_lf = perf_lf
            .sort(["date"], Default::default())
            .group_by([col("date")]) // collapse possible duplicate-date rows
            .agg(
                perf_features
                    .iter()
                    .map(|c| col(*c).last().alias(*c))
                    .collect::<Vec<_>>(),
            )
            .select(perf_select)
            .sort(["date"], Default::default());

        // Forward-fill so a single weekly VO2max measurement applies to days
        // around it. Cap fill so a long gap doesn't carry stale numbers forward.
        let perf_lf = perf_lf.with_columns(
            perf_features
                .iter()
                .map(|c| col(*c).forward_fill(Some(14)))
                .collect::<Vec<_>>(),
        );

        lf = lf.join(
            perf_lf,
            [col("date")],
            [col("date")],
            JoinArgs::new(JoinType::Left),
        );
        for c in perf_features {
            feat_cols.push(c.to_string());
        }
    }

    // --- Activities (aggregate to daily, then LEFT JOIN) -------------------
    if let Some(act_lf) = try_load("activities", || data::load_activities(config)) {
        // start_time_local is a timestamp; truncate to date for the join key.
        let daily = act_lf
            .with_column(col("start_time_local").dt().date().alias("date"))
            .group_by([col("date")])
            .agg([
                col("activity_id").count().alias("activity_count"),
                col("duration_sec")
                    .cast(DataType::Float64)
                    .sum()
                    .alias("activity_duration_sec"),
                col("distance_m")
                    .cast(DataType::Float64)
                    .sum()
                    .alias("activity_distance_m"),
                col("calories")
                    .cast(DataType::Float64)
                    .sum()
                    .alias("activity_calories"),
                col("training_load")
                    .cast(DataType::Float64)
                    .max()
                    .alias("activity_training_load"),
            ])
            .sort(["date"], Default::default());

        lf = lf.join(
            daily,
            [col("date")],
            [col("date")],
            JoinArgs::new(JoinType::Left),
        );

        // Activities sparsity is meaningful: a missing date == 0 activities,
        // not "unknown". Fill activity-derived columns with 0 rather than
        // forward-filling.
        let act_features = [
            "activity_count",
            "activity_duration_sec",
            "activity_distance_m",
            "activity_calories",
            "activity_training_load",
        ];
        lf = lf.with_columns(
            act_features
                .iter()
                .map(|c| {
                    col(*c)
                        .cast(DataType::Float64)
                        .fill_null(lit(0.0))
                        .alias(*c)
                })
                .collect::<Vec<_>>(),
        );
        for c in act_features {
            feat_cols.push(c.to_string());
        }
    }

    // --- Weight (sparse, forward-fill with cap) ----------------------------
    if let Some(weight_lf) = try_load("weight", || data::load_weight(config)) {
        let weight_features = ["weight_kg", "body_fat"];
        // Multiple weigh-ins on the same date: take the last one (per date sort).
        let weight_lf = weight_lf
            .sort(["date"], Default::default())
            .group_by([col("date")])
            .agg(
                weight_features
                    .iter()
                    .map(|c| col(*c).last().alias(*c))
                    .collect::<Vec<_>>(),
            )
            .sort(["date"], Default::default())
            .with_columns(
                weight_features
                    .iter()
                    .map(|c| col(*c).forward_fill(Some(30)))
                    .collect::<Vec<_>>(),
            );

        lf = lf.join(
            weight_lf,
            [col("date")],
            [col("date")],
            JoinArgs::new(JoinType::Left),
        );
        for c in weight_features {
            feat_cols.push(c.to_string());
        }
    }

    // --- Blood pressure (very sparse, forward-fill with cap) ---------------
    if let Some(bp_lf) = try_load("blood_pressure", || data::load_blood_pressure(config)) {
        let bp_features = ["systolic", "diastolic"];
        let bp_lf = bp_lf
            .sort(["date"], Default::default())
            .group_by([col("date")])
            .agg(
                bp_features
                    .iter()
                    .map(|c| col(*c).cast(DataType::Float64).last().alias(*c))
                    .collect::<Vec<_>>(),
            )
            .sort(["date"], Default::default())
            .with_columns(
                bp_features
                    .iter()
                    .map(|c| col(*c).forward_fill(Some(30)))
                    .collect::<Vec<_>>(),
            );

        lf = lf.join(
            bp_lf,
            [col("date")],
            [col("date")],
            JoinArgs::new(JoinType::Left),
        );
        for c in bp_features {
            feat_cols.push(c.to_string());
        }
    }

    // --- 7-day rolling features --------------------------------------------
    let window_size = 7;
    let min_periods = 5;
    let rolling_opts = RollingOptionsFixedWindow {
        window_size,
        min_periods,
        ..Default::default()
    };

    lf = lf.with_columns([
        col("resting_hr")
            .rolling_mean(rolling_opts.clone())
            .alias("resting_hr_7d_mean"),
        col("resting_hr")
            .rolling_std(rolling_opts.clone())
            .alias("resting_hr_7d_std"),
        col("sleep_hours")
            .rolling_mean(rolling_opts.clone())
            .alias("sleep_hours_7d_mean"),
        col("steps")
            .rolling_mean(rolling_opts.clone())
            .alias("steps_7d_mean"),
        col("active_calories")
            .rolling_mean(rolling_opts.clone())
            .alias("active_calories_7d_mean"),
        col("hrv_last_night")
            .rolling_mean(rolling_opts.clone())
            .alias("hrv_last_night_7d_mean"),
    ]);
    for c in [
        "resting_hr_7d_mean",
        "resting_hr_7d_std",
        "sleep_hours_7d_mean",
        "steps_7d_mean",
        "active_calories_7d_mean",
        "hrv_last_night_7d_mean",
    ] {
        feat_cols.push(c.to_string());
    }

    Ok((lf, feat_cols))
}

/// Maximum fraction of nulls a feature column may have before it gets
/// dropped from the training set entirely. Without this, sparse newer
/// metrics like HRV would force us to discard years of older data.
const MAX_FEATURE_NULL_RATE: f64 = 0.30;

/// Filter out feature columns whose null rate exceeds the threshold. Logs
/// dropped columns and returns the surviving feature list. Operates on a
/// collected DataFrame so the rates reflect what's actually on disk.
fn prune_sparse_features(df: &DataFrame, feat_cols: &[String]) -> Vec<String> {
    let n = df.height() as f64;
    if n == 0.0 {
        return feat_cols.to_vec();
    }
    let mut kept = Vec::with_capacity(feat_cols.len());
    for c in feat_cols {
        let nulls = df
            .column(c.as_str())
            .map(|col| col.null_count() as f64)
            .unwrap_or(n);
        let rate = nulls / n;
        if rate > MAX_FEATURE_NULL_RATE {
            warn!(
                feature = %c,
                null_rate = format!("{:.1}%", rate * 100.0),
                "Dropping sparse feature column"
            );
        } else {
            kept.push(c.clone());
        }
    }
    kept
}

/// Build the full training feature matrix. Returns a DataFrame containing the
/// surviving feature columns, the target column, and `date`, with rows
/// missing any feature OR the target dropped.
pub fn build_feature_matrix(config: &Config, target: &str) -> Result<(DataFrame, Vec<String>)> {
    let (target_source, shift) = lookup_target(target)?;
    let (mut lf, feat_cols) = build_features_lazy(config)?;

    // Build the target column from its source.
    lf = lf.with_column(col(target_source).shift(lit(shift)).alias(target));

    // Materialize once so we can measure per-column null rates and prune
    // features that are too sparse to train on.
    let raw_df = lf.collect()?;
    let kept = prune_sparse_features(&raw_df, &feat_cols);
    if kept.is_empty() {
        return Err(AppError::Data(
            "All feature columns exceeded the null-rate threshold".into(),
        ));
    }

    // Re-select kept features + target + date, then drop rows still missing
    // any required value.
    let mut select_cols: Vec<Expr> = kept.iter().map(|c| col(c.as_str())).collect();
    select_cols.push(col(target));
    select_cols.push(col("date"));
    let mut required: Vec<Expr> = kept.iter().map(|c| col(c.as_str())).collect();
    required.push(col(target));

    let df = raw_df
        .lazy()
        .select(select_cols)
        .drop_nulls(Some(required))
        .collect()?;

    info!(
        rows = df.height(),
        features = kept.len(),
        target,
        "Feature matrix built"
    );
    Ok((df, kept))
}

/// Build features for prediction: the most recent row whose required feature
/// columns are all non-null. No target column is computed (since we're
/// predicting, the answer for the latest row is exactly what we don't have
/// yet).
///
/// `required_features` is the list of feature column names the trained model
/// expects (read from `ModelMetadata::feature_names`). Only those columns are
/// required to be non-null on the returned row, which keeps prediction
/// aligned with whatever was actually used at training time.
///
/// **Contract:** callers MUST pass the trained model's `feature_names`, not
/// the feature list returned by `build_feature_matrix` on a fresh run. The
/// training-time list reflects which columns survived the sparse-feature
/// pruning at the moment the model was fit, and the model only knows how to
/// consume that exact set. The `Vec<String>` returned by this function is
/// just an echo of what was passed in, kept in the signature for symmetry
/// with `build_feature_matrix` — do not derive feature names from it.
pub fn build_prediction_features(
    config: &Config,
    required_features: &[String],
) -> Result<(DataFrame, Vec<String>)> {
    let (lf, _feat_cols) = build_features_lazy(config)?;

    let mut select_cols: Vec<Expr> = required_features.iter().map(|c| col(c.as_str())).collect();
    select_cols.push(col("date"));
    let lf = lf.select(select_cols);

    let required: Vec<Expr> = required_features.iter().map(|c| col(c.as_str())).collect();
    let df = lf.drop_nulls(Some(required)).collect()?;

    if df.height() == 0 {
        return Err(AppError::Data(
            "No rows have all required feature columns populated; cannot build prediction features"
                .to_string(),
        ));
    }

    let last = df.tail(Some(1));
    info!(
        features = required_features.len(),
        "Prediction feature row built"
    );
    Ok((last, required_features.to_vec()))
}
