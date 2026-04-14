use std::path::Path;
use std::sync::Arc;

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

/// Promote two numeric DataTypes to their common supertype.
/// Returns None for incompatible types (e.g. String vs Int32).
fn numeric_supertype(a: &DataType, b: &DataType) -> Option<DataType> {
    use DataType::*;
    if a == b {
        return Some(a.clone());
    }
    match (a, b) {
        // Integer widening
        (Int8, Int16) | (Int16, Int8) => Some(Int16),
        (Int8 | Int16, Int32) | (Int32, Int8 | Int16) => Some(Int32),
        (Int8 | Int16 | Int32, Int64) | (Int64, Int8 | Int16 | Int32) => Some(Int64),
        // Float widening
        (Float32, Float64) | (Float64, Float32) => Some(Float64),
        // Int -> Float promotion
        (Int8 | Int16 | Int32, Float32) | (Float32, Int8 | Int16 | Int32) => Some(Float64),
        (Int8 | Int16 | Int32 | Int64, Float64) | (Float64, Int8 | Int16 | Int32 | Int64) => {
            Some(Float64)
        }
        (Int64, Float32) | (Float32, Int64) => Some(Float64),
        _ => None,
    }
}

/// Build a union schema from all parquet files in a directory.
///
/// When parquet files are written over time, newer files may contain columns
/// that older files lack (e.g. `consumed_calories` added to daily_health).
/// Polars' `allow_missing_columns` handles files *missing* columns from the
/// resolved schema, but errors on files with *extra* columns.  By reading
/// each file's metadata and merging into a superset schema we ensure every
/// column is known up-front.
fn union_schema(dir: &Path) -> Result<SchemaRef> {
    let mut schema = Schema::default();
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| AppError::Data(format!("Cannot read {}: {e}", dir.display())))?
        .flatten()
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("parquet"))
        .collect();
    entries.sort_by_key(|e| e.path());

    for entry in entries {
        let path = entry.path();
        let path_str = path.to_string_lossy().to_string();
        let file_schema = LazyFrame::scan_parquet(&path_str, Default::default())
            .and_then(|mut lf| lf.collect_schema())
            .map_err(|e| {
                AppError::Data(format!("Cannot read schema of {}: {e}", path.display()))
            })?;
        for (name, dtype) in file_schema.iter() {
            match schema.get(name.as_str()) {
                Some(existing) if existing != dtype => {
                    let super_type = numeric_supertype(existing, dtype).ok_or_else(|| {
                        AppError::Data(format!(
                            "Column '{}' has conflicting types: {} vs {}",
                            name, existing, dtype
                        ))
                    })?;
                    schema.with_column(name.clone(), super_type);
                }
                None => {
                    schema.with_column(name.clone(), dtype.clone());
                }
                _ => {} // same type, nothing to do
            }
        }
    }

    Ok(Arc::new(schema))
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

    let schema = union_schema(&dir)?;

    let pattern = dir.join("*.parquet");
    let pattern_str = pattern.to_string_lossy().to_string();

    let args = ScanArgsParquet {
        allow_missing_columns: true,
        schema: Some(schema),
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

/// Load daily nutrition / food logging summaries as a LazyFrame.
pub fn load_nutrition(config: &Config) -> Result<LazyFrame> {
    scan_entity(&config.garmin_storage_path, "nutrition")
}

/// Load per-activity time-series detail data as a LazyFrame.
#[allow(dead_code)]
pub fn load_activity_details(config: &Config) -> Result<LazyFrame> {
    scan_entity(&config.garmin_storage_path, "activity_details")
}

/// Load per-activity lap/split data as a LazyFrame.
#[allow(dead_code)]
pub fn load_activity_splits(config: &Config) -> Result<LazyFrame> {
    scan_entity(&config.garmin_storage_path, "activity_splits")
}

/// Load the decomposed-health parquet (output of `model-health decompose`).
/// Lives under `data_dir` rather than `garmin_storage_path` because it's
/// derived from raw data, not fetched. Returns an error if the file doesn't
/// exist — callers that want optional behavior should handle the error.
pub fn load_decomposed_health(config: &Config) -> Result<LazyFrame> {
    let path = config.data_dir.join("decomposed_health.parquet");
    if !path.exists() {
        return Err(AppError::Data(format!(
            "decomposed_health.parquet not found at {}. Run `model-health decompose` first.",
            path.display()
        )));
    }
    let path_str = path.to_string_lossy().to_string();
    LazyFrame::scan_parquet(&path_str, Default::default())
        .map_err(|e| AppError::Data(format!("Failed to scan decomposed_health.parquet: {e}")))
}

/// Activity types classified as "running" for volume / load calculations.
/// Shared across modules so that injury-risk, decompose, and race analysis
/// agree on what counts as a run.
pub const RUNNING_TYPES: &[&str] = &["running", "treadmill_running", "virtual_run"];

/// Build an `activity_type IN (...)` filter expression from a slice of type
/// strings. Uses an OR-chain because Polars `is_in` behaviour varies across
/// versions.
pub fn type_in_list(types: &[&str]) -> Expr {
    types
        .iter()
        .map(|t| col("activity_type").eq(lit(*t)))
        .reduce(|a, b| a.or(b))
        .expect("types must be non-empty")
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
        ("Nutrition", load_nutrition),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a DataFrame to a parquet file.
    fn write_parquet(df: &mut DataFrame, path: &Path) {
        let file = std::fs::File::create(path).unwrap();
        ParquetWriter::new(file).finish(df).unwrap();
    }

    #[test]
    fn scan_entity_with_extra_columns_in_newer_file() {
        // Simulate the bug: older files have {date, steps}, newer file adds {consumed_calories}.
        // Without the union_schema fix, this would fail with
        // "parquet file contained extra columns and no selection was given".
        let tmp = tempfile::tempdir().unwrap();
        let entity_dir = tmp.path().join("daily_health");
        std::fs::create_dir_all(&entity_dir).unwrap();

        // Older file: date + steps only
        let mut old = df![
            "date" => &["2025-01-01", "2025-01-02"],
            "steps" => &[5000i32, 6000],
        ]
        .unwrap();
        write_parquet(&mut old, &entity_dir.join("2025-01.parquet"));

        // Newer file: date + steps + consumed_calories (extra column)
        let mut new = df![
            "date" => &["2026-04-01"],
            "steps" => &[7000i32],
            "consumed_calories" => &[2100i32],
        ]
        .unwrap();
        write_parquet(&mut new, &entity_dir.join("2026-04.parquet"));

        // This must succeed — the old behavior would error here.
        let lf = scan_entity(tmp.path(), "daily_health").unwrap();
        let result = lf.collect().unwrap();

        assert_eq!(result.height(), 3);
        assert!(result.schema().contains("consumed_calories"));
        assert!(result.schema().contains("steps"));

        // Old rows should have null for consumed_calories
        let cc = result.column("consumed_calories").unwrap();
        assert_eq!(cc.null_count(), 2);
    }

    #[test]
    fn scan_entity_uniform_schema() {
        // When all files share the same schema, everything should work as before.
        let tmp = tempfile::tempdir().unwrap();
        let entity_dir = tmp.path().join("test_entity");
        std::fs::create_dir_all(&entity_dir).unwrap();

        let mut a = df!["id" => &[1i32, 2], "val" => &[10i32, 20]].unwrap();
        let mut b = df!["id" => &[3i32], "val" => &[30i32]].unwrap();
        write_parquet(&mut a, &entity_dir.join("a.parquet"));
        write_parquet(&mut b, &entity_dir.join("b.parquet"));

        let lf = scan_entity(tmp.path(), "test_entity").unwrap();
        let result = lf.collect().unwrap();
        assert_eq!(result.height(), 3);
    }

    #[test]
    fn scan_entity_missing_dir_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let result = scan_entity(tmp.path(), "nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn union_schema_merges_all_columns() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();

        let mut a = df!["x" => &[1i32], "y" => &[2i32]].unwrap();
        let mut b = df!["y" => &[3i32], "z" => &[4i64]].unwrap();
        write_parquet(&mut a, &dir.join("a.parquet"));
        write_parquet(&mut b, &dir.join("b.parquet"));

        let schema = union_schema(dir).unwrap();
        assert!(schema.contains("x"));
        assert!(schema.contains("y"));
        assert!(schema.contains("z"));
        assert_eq!(schema.len(), 3);
        // y is i32 in both files, so it stays i32
        assert_eq!(schema.get("y").unwrap(), &DataType::Int32);
    }

    #[test]
    fn union_schema_promotes_conflicting_types() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();

        // a has val as i32, b has val as i64 — should promote to i64
        let mut a = df!["val" => &[1i32]].unwrap();
        let mut b = df!["val" => &[2i64]].unwrap();
        write_parquet(&mut a, &dir.join("a.parquet"));
        write_parquet(&mut b, &dir.join("b.parquet"));

        let schema = union_schema(dir).unwrap();
        assert_eq!(schema.get("val").unwrap(), &DataType::Int64);
    }
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
