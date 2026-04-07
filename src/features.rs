use polars::prelude::*;
use tracing::info;

use crate::config::Config;
use crate::data;
use crate::error::{AppError, Result};
use crate::validation;

/// Names of all feature columns produced by the pipeline.
pub fn feature_columns() -> Vec<&'static str> {
    vec![
        "resting_hr",
        "sleep_hours",
        "steps",
        "active_calories",
        "avg_stress",
        "body_battery_delta",
        "resting_hr_7d_mean",
        "resting_hr_7d_std",
        "sleep_hours_7d_mean",
        "steps_7d_mean",
        "active_calories_7d_mean",
        "day_sin",
        "day_cos",
    ]
}

/// Build the full feature matrix from raw Garmin data.
/// Returns a DataFrame with feature columns and the target column.
pub fn build_feature_matrix(config: &Config, target: &str) -> Result<(DataFrame, Vec<String>)> {
    info!("Building feature matrix");

    // Load raw data, then clean. The cleaning happens before any feature
    // computation so sentinel values and implausible readings don't pollute
    // rolling-window means.
    let health_lf = data::load_daily_health(config)?;
    let before = health_lf.clone().collect()?;
    let cleaned_lf = validation::clean_daily_health(health_lf)?;
    let after = cleaned_lf.clone().collect()?;
    validation::log_cleaning_diff(&before, &after);

    // Note: cleaning is order-independent (no stateful operations), so it's
    // safe to sort here. If you ever add a stateful rule like forward-fill
    // to validation, sort first.
    let mut lf = cleaned_lf
        .sort(["date"], Default::default())
        // Core daily features
        .with_columns([
            // Sleep in hours
            (col("sleep_seconds").cast(DataType::Float64) / lit(3600.0)).alias("sleep_hours"),
            // Body battery delta
            (col("body_battery_end") - col("body_battery_start"))
                .cast(DataType::Float64)
                .alias("body_battery_delta"),
            // Cast core columns to f64 for consistency
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
        ]);

    // Day-of-week cyclical encoding
    lf = lf.with_columns([
        // NaiveDate weekday as 0-6, encode as sine/cosine
        (col("date").dt().weekday().cast(DataType::Float64)
            * lit(2.0 * std::f64::consts::PI / 7.0))
        .sin()
        .alias("day_sin"),
        (col("date").dt().weekday().cast(DataType::Float64)
            * lit(2.0 * std::f64::consts::PI / 7.0))
        .cos()
        .alias("day_cos"),
    ]);

    // 7-day rolling window features
    let window_size = 7;
    let min_periods = 5; // Require at least 5 of 7 days
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
    ]);

    // Target: next-day value (shift the target column back by 1 so row N predicts row N+1)
    let target_source = match target {
        "next_day_resting_hr" => "resting_hr",
        other => {
            return Err(AppError::Data(format!(
                "Unknown target '{other}'. Available targets: next_day_resting_hr"
            )));
        }
    };

    lf = lf.with_column(col(target_source).shift(lit(-1)).alias(target));

    // Collect the feature columns we want
    let feat_cols = feature_columns();
    let mut select_cols: Vec<Expr> = feat_cols.iter().map(|&c| col(c)).collect();
    select_cols.push(col(target));
    select_cols.push(col("date"));

    lf = lf.select(select_cols);

    // Drop rows with any nulls (handles rolling window warm-up and missing data)
    lf = lf.drop_nulls(None);

    let df = lf.collect()?;
    let actual_features: Vec<String> = feat_cols.iter().map(|s| s.to_string()).collect();

    info!(
        rows = df.height(),
        features = actual_features.len(),
        "Feature matrix built"
    );

    Ok((df, actual_features))
}
