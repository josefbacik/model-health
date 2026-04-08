use std::path::Path;

use chrono::NaiveDate;
use polars::prelude::*;
use tracing::info;

use crate::config::Config;
use crate::error::{AppError, Result};

/// Returns true if `dir` exists and contains at least one `.parquet` file.
///
/// Used as a pre-check before `LazyFrame::scan_parquet`, which will happily
/// return Ok for a glob that matches zero files and only fail later at
/// `.collect()` time with a confusing "expected at least 1 source" error.
pub fn dir_has_parquet(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .any(|e| e.path().extension().and_then(|s| s.to_str()) == Some("parquet"))
        })
        .unwrap_or(false)
}

/// Load Parquet files for a given entity type from garmin-cli's storage.
fn scan_entity(base_path: &Path, entity_dir: &str) -> Result<LazyFrame> {
    let dir = base_path.join(entity_dir);
    if !dir_has_parquet(&dir) {
        return Err(AppError::Data(format!(
            "No parquet files found for {entity_dir} in {}",
            dir.display()
        )));
    }

    let pattern = dir.join("*.parquet");
    let pattern_str = pattern.to_string_lossy().to_string();

    let args = ScanArgsParquet {
        allow_missing_columns: true,
        ..Default::default()
    };
    LazyFrame::scan_parquet(&pattern_str, args)
        .map_err(|e| AppError::Data(format!("Failed to scan {entity_dir} parquet files: {e}")))
}

/// Load daily health data as a LazyFrame.
pub fn load_daily_health(config: &Config) -> Result<LazyFrame> {
    scan_entity(&config.garmin_storage_path, "daily_health")
}

/// Load performance metrics as a LazyFrame.
pub fn load_performance_metrics(config: &Config) -> Result<LazyFrame> {
    scan_entity(&config.garmin_storage_path, "performance_metrics")
}

/// Load activities as a LazyFrame.
pub fn load_activities(config: &Config) -> Result<LazyFrame> {
    scan_entity(&config.garmin_storage_path, "activities")
}

/// Load weight entries as a LazyFrame.
pub fn load_weight(config: &Config) -> Result<LazyFrame> {
    scan_entity(&config.garmin_storage_path, "weight")
}

/// Load blood pressure measurements as a LazyFrame.
pub fn load_blood_pressure(config: &Config) -> Result<LazyFrame> {
    scan_entity(&config.garmin_storage_path, "blood_pressure")
}

/// Filter a LazyFrame by date range.
#[allow(dead_code)]
pub fn filter_date_range(
    lf: LazyFrame,
    date_col: &str,
    from: Option<NaiveDate>,
    to: Option<NaiveDate>,
) -> LazyFrame {
    let mut lf = lf;
    if let Some(from) = from {
        lf = lf.filter(col(date_col).gt_eq(lit(from)));
    }
    if let Some(to) = to {
        lf = lf.filter(col(date_col).lt_eq(lit(to)));
    }
    lf
}

/// Profile data quality for a given dataset.
/// Prints column names, types, null percentages, date range, and basic stats.
pub fn profile_data(config: &Config) -> Result<()> {
    println!("=== Data Profile ===\n");
    println!("Storage path: {}\n", config.garmin_storage_path.display());

    // Profile each entity type
    for (name, loader) in [
        (
            "Daily Health",
            load_daily_health as fn(&Config) -> Result<LazyFrame>,
        ),
        ("Performance Metrics", load_performance_metrics),
        ("Activities", load_activities),
        ("Weight", load_weight),
        ("Blood Pressure", load_blood_pressure),
    ] {
        match loader(config) {
            Ok(lf) => {
                println!("--- {name} ---");
                match profile_lazyframe(lf, name) {
                    Ok(()) => {}
                    Err(e) => println!("  Error profiling: {e}"),
                }
                println!();
            }
            Err(e) => {
                println!("--- {name} ---");
                println!("  Not available: {e}\n");
            }
        }
    }

    Ok(())
}

fn profile_lazyframe(lf: LazyFrame, _entity_name: &str) -> Result<()> {
    let df = lf.collect()?;
    let row_count = df.height();
    println!("  Rows: {row_count}");

    if row_count == 0 {
        println!("  (empty)");
        return Ok(());
    }

    // Print schema
    println!("  Columns:");
    let schema = df.schema();
    for (name, dtype) in schema.iter() {
        let null_count = df.column(name.as_str())?.null_count();
        let null_pct = if row_count > 0 {
            (null_count as f64 / row_count as f64) * 100.0
        } else {
            0.0
        };
        println!("    {name:<35} {dtype:<20} nulls: {null_count:>6} ({null_pct:>5.1}%)");
    }

    // Date range if there's a 'date' column
    if schema.contains("date") {
        let dates = df.column("date")?;
        if let (Ok(min_date), Ok(max_date)) = (dates.min_reduce(), dates.max_reduce()) {
            println!(
                "  Date range: {:?} to {:?}",
                min_date.value(),
                max_date.value()
            );
        }

        // Count distinct dates for gap detection
        if let Ok(n_dates) = dates.n_unique() {
            println!("  Unique dates: {n_dates}");
        }
    }

    // Basic stats for numeric columns
    println!("  Key stats:");
    for col_name in [
        "resting_hr",
        "steps",
        "sleep_seconds",
        "sleep_score",
        "avg_stress",
        "avg_respiration",
        "avg_spo2",
        "lowest_spo2",
        "hrv_last_night",
        "hrv_weekly_avg",
        "body_battery_start",
        "body_battery_end",
        "vo2max",
        "fitness_age",
        "training_readiness",
        "endurance_score",
        "weight_kg",
        "bmi",
        "body_fat",
        "systolic",
        "diastolic",
        "pulse",
    ] {
        if schema.contains(col_name)
            && let Ok(col) = df.column(col_name)
        {
            let series = col.as_materialized_series();
            let mean = series.mean();
            let min = series.min_reduce().ok().map(|s| format!("{}", s.value()));
            let max = series.max_reduce().ok().map(|s| format!("{}", s.value()));
            let non_null = row_count - col.null_count();
            if let Some(mean) = mean {
                println!(
                    "    {col_name:<22} mean: {mean:>8.1}  min: {:>8}  max: {:>8}  non-null: {}",
                    min.as_deref().unwrap_or("?"),
                    max.as_deref().unwrap_or("?"),
                    non_null
                );
            }
        }
    }

    Ok(())
}

/// Summary info for a dataset: row count and date range.
pub struct DataSummary {
    pub row_count: usize,
    pub min_date: Option<String>,
    pub max_date: Option<String>,
}

/// Get a quick summary (row count, date range) for a dataset.
pub fn summarize(lf: LazyFrame, date_col: &str) -> Result<DataSummary> {
    // Compute min/max via lazy aggregation so polars handles type formatting
    let stats = lf
        .clone()
        .select([
            col(date_col).min().alias("min_date"),
            col(date_col).max().alias("max_date"),
            col(date_col).count().alias("count"),
        ])
        .collect()?;

    let row_count = stats
        .column("count")?
        .get(0)
        .ok()
        .and_then(|v| v.try_extract::<u32>().ok())
        .unwrap_or(0) as usize;

    if row_count == 0 {
        return Ok(DataSummary {
            row_count,
            min_date: None,
            max_date: None,
        });
    }

    let min_date = stats
        .column("min_date")?
        .get(0)
        .ok()
        .map(|v| format!("{}", v));
    let max_date = stats
        .column("max_date")?
        .get(0)
        .ok()
        .map(|v| format!("{}", v));

    Ok(DataSummary {
        row_count,
        min_date,
        max_date,
    })
}

/// Check if we have enough data to train a model.
pub fn validate_training_data(config: &Config) -> Result<()> {
    // Surface the friendly "run sync" message when nothing has been fetched
    // yet, instead of letting `scan_entity` produce its raw error.
    if !dir_has_parquet(&config.garmin_storage_path.join("daily_health")) {
        return Err(AppError::Data(format!(
            "Need at least {} days of data for training, but no daily health data is present. Run `model-health sync` to download data.",
            config.min_training_days
        )));
    }

    let lf = load_daily_health(config)?;
    let df = lf.collect()?;
    let row_count = df.height();

    if row_count < config.min_training_days {
        return Err(AppError::Data(format!(
            "Need at least {} days of data for training, but only have {}. Run `model-health sync` to download more data.",
            config.min_training_days, row_count
        )));
    }

    // Check core columns have acceptable null rates
    let max_null_pct = 0.30;
    for col_name in ["resting_hr", "sleep_seconds", "steps"] {
        if let Ok(col) = df.column(col_name) {
            let null_pct = col.null_count() as f64 / row_count as f64;
            if null_pct > max_null_pct {
                return Err(AppError::Data(format!(
                    "Column '{col_name}' has {:.0}% null values (max allowed: {:.0}%). Data quality is insufficient for training.",
                    null_pct * 100.0,
                    max_null_pct * 100.0
                )));
            }
        } else {
            return Err(AppError::Data(format!(
                "Required column '{col_name}' not found in daily health data"
            )));
        }
    }

    info!(rows = row_count, "Training data validation passed");
    Ok(())
}
