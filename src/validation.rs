//! Data validation and cleaning for raw Garmin data.
//!
//! Garmin's API uses sentinel values (e.g. `-1` for "no data") and sometimes
//! returns physically implausible values (RHR of 250, sleep of 0 seconds on
//! no-wear days, etc.). This module replaces those with `null` so that
//! `drop_nulls` in feature engineering can filter them out cleanly.
//!
//! Cleaning rules for daily health:
//!
//! 1. **Sentinel values → null**
//!    - `avg_stress < 0` (Garmin uses `-1` for no data)
//!    - `max_stress < 0`
//!
//!    Note: as of the weight/BP refactor, `fetch.rs` now also filters these
//!    sentinels at the API boundary before writing to parquet. The rules
//!    below are kept as a defensive layer so older parquet files written by
//!    earlier versions (or by garmin-cli) get cleaned at training time.
//!
//! 2. **Implausible values → null**
//!    - `resting_hr` outside `[25, 200]` — floor accommodates elite endurance athletes
//!    - `sleep_seconds <= 0` or `> 16` hours
//!    - `steps < 0` or `> 200_000`
//!    - `total_calories < 500` or `> 10_000` (BMR is included; double-counted activities can spike)
//!    - `active_calories < 0` or `> 8_000`
//!    - `body_battery_start` / `body_battery_end` outside `[0, 100]`
//!
//! 3. **No-wear day detection**: rows where `resting_hr` and `sleep_seconds`
//!    are both null AND `steps` is null/zero are dropped entirely.
//!
//! 4. **Deduplication**: rows are deduped by date (latest wins) so duplicate
//!    upstream writes don't pollute rolling-window features.

use polars::prelude::*;
use tracing::info;

use crate::error::Result;

/// Apply all cleaning rules to a daily health LazyFrame.
///
/// `cleaning_stats` is logged at info level so callers can see how much
/// data was filtered. Cleaning rules are skipped silently if their column
/// is not present in the input schema (since `data::load_daily_health`
/// uses `allow_missing_columns: true`).
pub fn clean_daily_health(mut lf: LazyFrame) -> Result<LazyFrame> {
    let schema = lf.collect_schema()?;
    let has = |name: &str| schema.contains(name);

    // --- Sentinel values: stress < 0 means "no data" ---
    if has("avg_stress") {
        lf = lf.with_column(
            null_when(col("avg_stress"), col("avg_stress").lt(lit(0_i32))).alias("avg_stress"),
        );
    }
    if has("max_stress") {
        lf = lf.with_column(
            null_when(col("max_stress"), col("max_stress").lt(lit(0_i32))).alias("max_stress"),
        );
    }

    // --- Implausible RHR (range covers elite athletes through any plausible resting value) ---
    if has("resting_hr") {
        lf = lf.with_column(
            null_when(
                col("resting_hr"),
                col("resting_hr")
                    .lt(lit(25_i32))
                    .or(col("resting_hr").gt(lit(200_i32))),
            )
            .alias("resting_hr"),
        );
    }

    // --- Sleep: 0 = no recording, > 16 hours = impossible ---
    if has("sleep_seconds") {
        lf = lf.with_column(
            null_when(
                col("sleep_seconds"),
                col("sleep_seconds")
                    .lt_eq(lit(0_i32))
                    .or(col("sleep_seconds").gt(lit(16_i32 * 3600))),
            )
            .alias("sleep_seconds"),
        );
    }

    // --- Steps: bounded sanity check ---
    if has("steps") {
        lf = lf.with_column(
            null_when(
                col("steps"),
                col("steps")
                    .lt(lit(0_i32))
                    .or(col("steps").gt(lit(200_000_i32))),
            )
            .alias("steps"),
        );
    }

    // --- total_calories: BMR is included, so floor at 500 (deeply implausible);
    //     ceiling at 10000 catches double-counted activities ---
    if has("total_calories") {
        lf = lf.with_column(
            null_when(
                col("total_calories"),
                col("total_calories")
                    .lt(lit(500_i32))
                    .or(col("total_calories").gt(lit(10_000_i32))),
            )
            .alias("total_calories"),
        );
    }

    // --- active_calories: bounded sanity check ---
    if has("active_calories") {
        lf = lf.with_column(
            null_when(
                col("active_calories"),
                col("active_calories")
                    .lt(lit(0_i32))
                    .or(col("active_calories").gt(lit(8_000_i32))),
            )
            .alias("active_calories"),
        );
    }

    // --- Body battery: must be in [0, 100], else null both endpoints
    //     so the delta computation in features.rs doesn't see spurious jumps ---
    if has("body_battery_start") && has("body_battery_end") {
        let bad = col("body_battery_start")
            .lt(lit(0_i32))
            .or(col("body_battery_start").gt(lit(100_i32)))
            .or(col("body_battery_end").lt(lit(0_i32)))
            .or(col("body_battery_end").gt(lit(100_i32)));
        lf = lf.with_columns([
            null_when(col("body_battery_start"), bad.clone()).alias("body_battery_start"),
            null_when(col("body_battery_end"), bad).alias("body_battery_end"),
        ]);
    }

    // --- No-wear day filter: drop rows with no useful signal ---
    // We need at least one of: a valid resting_hr, a valid sleep_seconds, or
    // some non-zero steps reading.
    if has("resting_hr") && has("sleep_seconds") && has("steps") {
        lf = lf.filter(
            col("resting_hr")
                .is_not_null()
                .or(col("sleep_seconds").is_not_null())
                .or(col("steps").gt(lit(0_i32))),
        );
    }

    // --- Deduplicate by date (latest write wins) ---
    // We sort after dedup so callers can rely on deterministic row order.
    if has("date") {
        lf = lf
            .unique(Some(vec!["date".to_string()]), UniqueKeepStrategy::Last)
            .sort(["date"], Default::default());
    }

    Ok(lf)
}

/// Helper: replace `expr` with null when `condition` is true.
fn null_when(expr: Expr, condition: Expr) -> Expr {
    when(condition).then(lit(NULL)).otherwise(expr)
}

/// Log a summary of cleaning effects so users can see data quality at a glance.
///
/// Reports rows dropped (no-wear days, dedupe) and the absolute null count
/// per column in the cleaned frame. Note: rows that contained sentinel values
/// AND were also no-wear days (common — they correlate) are dropped entirely
/// rather than nulled, so the per-column null count may be lower than you'd
/// naively expect.
pub fn log_cleaning_diff(before: &DataFrame, after: &DataFrame) {
    let before_rows = before.height();
    let after_rows = after.height();
    let dropped = before_rows.saturating_sub(after_rows);

    let cols = [
        "resting_hr",
        "sleep_seconds",
        "steps",
        "total_calories",
        "active_calories",
        "avg_stress",
        "max_stress",
        "body_battery_start",
        "body_battery_end",
    ];

    let mut nulls_after: Vec<(String, usize)> = Vec::new();
    for c in cols.iter() {
        if let Ok(col) = after.column(c) {
            let n = col.null_count();
            if n > 0 {
                nulls_after.push((c.to_string(), n));
            }
        }
    }

    info!(
        before_rows,
        after_rows,
        dropped_rows = dropped,
        nulls_after_per_col = ?nulls_after,
        "Daily health cleaning summary"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn d(s: &str) -> NaiveDate {
        NaiveDate::parse_from_str(s, "%Y-%m-%d").unwrap()
    }

    /// Build a minimal daily health DataFrame for testing.
    /// Uses a builder-ish struct so tests don't need to pass empty slices for
    /// columns they don't care about.
    #[derive(Default)]
    struct HealthBuilder {
        dates: Vec<NaiveDate>,
        rhr: Vec<Option<i32>>,
        sleep: Vec<Option<i32>>,
        steps: Vec<Option<i32>>,
        stress: Vec<Option<i32>>,
        max_stress: Vec<Option<i32>>,
        cal: Vec<Option<i32>>,
        active_cal: Vec<Option<i32>>,
        bb_start: Vec<Option<i32>>,
        bb_end: Vec<Option<i32>>,
    }

    impl HealthBuilder {
        fn row(
            mut self,
            date: &str,
            rhr: Option<i32>,
            sleep: Option<i32>,
            steps: Option<i32>,
            stress: Option<i32>,
            cal: Option<i32>,
        ) -> Self {
            self.dates.push(d(date));
            self.rhr.push(rhr);
            self.sleep.push(sleep);
            self.steps.push(steps);
            self.stress.push(stress);
            self.max_stress.push(stress.map(|s| s + 20));
            self.cal.push(cal);
            self.active_cal.push(cal.map(|c| c / 4));
            self.bb_start.push(Some(50));
            self.bb_end.push(Some(50));
            self
        }

        fn with_body_battery(mut self, start: Option<i32>, end: Option<i32>) -> Self {
            // Override the most recent row's body battery values.
            let n = self.bb_start.len();
            if n > 0 {
                self.bb_start[n - 1] = start;
                self.bb_end[n - 1] = end;
            }
            self
        }

        fn with_active_calories(mut self, ac: Option<i32>) -> Self {
            let n = self.active_cal.len();
            if n > 0 {
                self.active_cal[n - 1] = ac;
            }
            self
        }

        fn build(self) -> DataFrame {
            let date_vals: Vec<Option<NaiveDate>> = self.dates.iter().map(|d| Some(*d)).collect();
            df!(
                "date" => &date_vals,
                "resting_hr" => &self.rhr,
                "sleep_seconds" => &self.sleep,
                "steps" => &self.steps,
                "avg_stress" => &self.stress,
                "max_stress" => &self.max_stress,
                "total_calories" => &self.cal,
                "active_calories" => &self.active_cal,
                "body_battery_start" => &self.bb_start,
                "body_battery_end" => &self.bb_end,
            )
            .unwrap()
        }
    }

    fn get_i32(df: &DataFrame, col: &str, row: usize) -> Option<i32> {
        match df.column(col).unwrap().get(row).unwrap() {
            AnyValue::Int32(v) => Some(v),
            AnyValue::Null => None,
            other => panic!("unexpected type: {:?}", other),
        }
    }

    fn clean(df: DataFrame) -> DataFrame {
        clean_daily_health(df.lazy()).unwrap().collect().unwrap()
    }

    #[test]
    fn test_negative_stress_becomes_null() {
        let df = HealthBuilder::default()
            .row(
                "2024-01-01",
                Some(60),
                Some(28800),
                Some(8000),
                Some(-1),
                Some(2000),
            )
            .row(
                "2024-01-02",
                Some(60),
                Some(28800),
                Some(8000),
                Some(35),
                Some(2000),
            )
            .build();
        let cleaned = clean(df);
        assert_eq!(get_i32(&cleaned, "avg_stress", 0), None);
        assert_eq!(get_i32(&cleaned, "avg_stress", 1), Some(35));
    }

    #[test]
    fn test_implausible_rhr_becomes_null() {
        let df = HealthBuilder::default()
            .row(
                "2024-01-01",
                Some(20),
                Some(28800),
                Some(8000),
                Some(30),
                Some(2000),
            )
            .row(
                "2024-01-02",
                Some(60),
                Some(28800),
                Some(8000),
                Some(30),
                Some(2000),
            )
            .row(
                "2024-01-03",
                Some(250),
                Some(28800),
                Some(8000),
                Some(30),
                Some(2000),
            )
            .build();
        let cleaned = clean(df);
        assert_eq!(get_i32(&cleaned, "resting_hr", 0), None);
        assert_eq!(get_i32(&cleaned, "resting_hr", 1), Some(60));
        assert_eq!(get_i32(&cleaned, "resting_hr", 2), None);
    }

    #[test]
    fn test_athlete_rhr_is_kept() {
        // RHR of 28 should be valid for an elite endurance athlete (was rejected at floor=30).
        let df = HealthBuilder::default()
            .row(
                "2024-01-01",
                Some(28),
                Some(28800),
                Some(8000),
                Some(30),
                Some(2000),
            )
            .build();
        let cleaned = clean(df);
        assert_eq!(get_i32(&cleaned, "resting_hr", 0), Some(28));
    }

    #[test]
    fn test_rhr_boundaries() {
        let df = HealthBuilder::default()
            .row(
                "2024-01-01",
                Some(25),
                Some(28800),
                Some(8000),
                Some(30),
                Some(2000),
            ) // floor: kept
            .row(
                "2024-01-02",
                Some(24),
                Some(28800),
                Some(8000),
                Some(30),
                Some(2000),
            ) // below: nulled
            .row(
                "2024-01-03",
                Some(200),
                Some(28800),
                Some(8000),
                Some(30),
                Some(2000),
            ) // ceiling: kept
            .row(
                "2024-01-04",
                Some(201),
                Some(28800),
                Some(8000),
                Some(30),
                Some(2000),
            ) // above: nulled
            .build();
        let cleaned = clean(df);
        assert_eq!(get_i32(&cleaned, "resting_hr", 0), Some(25));
        assert_eq!(get_i32(&cleaned, "resting_hr", 1), None);
        assert_eq!(get_i32(&cleaned, "resting_hr", 2), Some(200));
        assert_eq!(get_i32(&cleaned, "resting_hr", 3), None);
    }

    #[test]
    fn test_zero_sleep_becomes_null() {
        let df = HealthBuilder::default()
            .row(
                "2024-01-01",
                Some(60),
                Some(0),
                Some(8000),
                Some(30),
                Some(2000),
            )
            .row(
                "2024-01-02",
                Some(60),
                Some(28800),
                Some(8000),
                Some(30),
                Some(2000),
            )
            .build();
        let cleaned = clean(df);
        assert_eq!(get_i32(&cleaned, "sleep_seconds", 0), None);
        assert_eq!(get_i32(&cleaned, "sleep_seconds", 1), Some(28800));
    }

    #[test]
    fn test_excessive_sleep_becomes_null() {
        let df = HealthBuilder::default()
            .row(
                "2024-01-01",
                Some(60),
                Some(20 * 3600),
                Some(8000),
                Some(30),
                Some(2000),
            )
            .build();
        let cleaned = clean(df);
        assert_eq!(get_i32(&cleaned, "sleep_seconds", 0), None);
    }

    #[test]
    fn test_sleep_boundary() {
        let df = HealthBuilder::default()
            .row(
                "2024-01-01",
                Some(60),
                Some(16 * 3600),
                Some(8000),
                Some(30),
                Some(2000),
            ) // 16h: kept
            .row(
                "2024-01-02",
                Some(60),
                Some(16 * 3600 + 1),
                Some(8000),
                Some(30),
                Some(2000),
            ) // > 16h: nulled
            .build();
        let cleaned = clean(df);
        assert_eq!(get_i32(&cleaned, "sleep_seconds", 0), Some(16 * 3600));
        assert_eq!(get_i32(&cleaned, "sleep_seconds", 1), None);
    }

    #[test]
    fn test_low_calories_becomes_null() {
        let df = HealthBuilder::default()
            .row(
                "2024-01-01",
                Some(60),
                Some(28800),
                Some(8000),
                Some(30),
                Some(100),
            )
            .row(
                "2024-01-02",
                Some(60),
                Some(28800),
                Some(8000),
                Some(30),
                Some(2000),
            )
            .build();
        let cleaned = clean(df);
        assert_eq!(get_i32(&cleaned, "total_calories", 0), None);
        assert_eq!(get_i32(&cleaned, "total_calories", 1), Some(2000));
    }

    #[test]
    fn test_excessive_calories_becomes_null() {
        let df = HealthBuilder::default()
            .row(
                "2024-01-01",
                Some(60),
                Some(28800),
                Some(8000),
                Some(30),
                Some(50000),
            )
            .build();
        let cleaned = clean(df);
        assert_eq!(get_i32(&cleaned, "total_calories", 0), None);
    }

    #[test]
    fn test_active_calories_bounds() {
        let df = HealthBuilder::default()
            .row(
                "2024-01-01",
                Some(60),
                Some(28800),
                Some(8000),
                Some(30),
                Some(2000),
            )
            .with_active_calories(Some(-50))
            .row(
                "2024-01-02",
                Some(60),
                Some(28800),
                Some(8000),
                Some(30),
                Some(2000),
            )
            .with_active_calories(Some(15000))
            .row(
                "2024-01-03",
                Some(60),
                Some(28800),
                Some(8000),
                Some(30),
                Some(2000),
            )
            .with_active_calories(Some(500))
            .build();
        let cleaned = clean(df);
        assert_eq!(get_i32(&cleaned, "active_calories", 0), None);
        assert_eq!(get_i32(&cleaned, "active_calories", 1), None);
        assert_eq!(get_i32(&cleaned, "active_calories", 2), Some(500));
    }

    #[test]
    fn test_body_battery_out_of_range_nulls_both() {
        // If either endpoint is invalid, both must be nulled so the delta isn't bogus.
        let df = HealthBuilder::default()
            .row(
                "2024-01-01",
                Some(60),
                Some(28800),
                Some(8000),
                Some(30),
                Some(2000),
            )
            .with_body_battery(Some(-1), Some(50))
            .row(
                "2024-01-02",
                Some(60),
                Some(28800),
                Some(8000),
                Some(30),
                Some(2000),
            )
            .with_body_battery(Some(80), Some(150))
            .row(
                "2024-01-03",
                Some(60),
                Some(28800),
                Some(8000),
                Some(30),
                Some(2000),
            )
            .with_body_battery(Some(80), Some(20))
            .build();
        let cleaned = clean(df);
        assert_eq!(get_i32(&cleaned, "body_battery_start", 0), None);
        assert_eq!(get_i32(&cleaned, "body_battery_end", 0), None);
        assert_eq!(get_i32(&cleaned, "body_battery_start", 1), None);
        assert_eq!(get_i32(&cleaned, "body_battery_end", 1), None);
        assert_eq!(get_i32(&cleaned, "body_battery_start", 2), Some(80));
        assert_eq!(get_i32(&cleaned, "body_battery_end", 2), Some(20));
    }

    #[test]
    fn test_steps_bounds() {
        let df = HealthBuilder::default()
            .row(
                "2024-01-01",
                Some(60),
                Some(28800),
                Some(-100),
                Some(30),
                Some(2000),
            )
            .row(
                "2024-01-02",
                Some(60),
                Some(28800),
                Some(500_000),
                Some(30),
                Some(2000),
            )
            .row(
                "2024-01-03",
                Some(60),
                Some(28800),
                Some(8000),
                Some(30),
                Some(2000),
            )
            .build();
        let cleaned = clean(df);
        assert_eq!(get_i32(&cleaned, "steps", 0), None);
        assert_eq!(get_i32(&cleaned, "steps", 1), None);
        assert_eq!(get_i32(&cleaned, "steps", 2), Some(8000));
    }

    #[test]
    fn test_no_wear_day_dropped() {
        let df = HealthBuilder::default()
            .row("2024-01-01", None, None, Some(0), Some(-1), None)
            .row(
                "2024-01-02",
                Some(60),
                Some(28800),
                Some(8000),
                Some(30),
                Some(2000),
            )
            .build();
        let cleaned = clean(df);
        assert_eq!(cleaned.height(), 1);
    }

    #[test]
    fn test_partial_wear_day_kept() {
        // Watch worn during day (steps recorded) but not at night (no sleep/HR).
        let df = HealthBuilder::default()
            .row("2024-01-01", None, None, Some(5000), Some(25), Some(2200))
            .build();
        let cleaned = clean(df);
        assert_eq!(cleaned.height(), 1);
        assert_eq!(get_i32(&cleaned, "steps", 0), Some(5000));
    }

    #[test]
    fn test_all_null_row_dropped() {
        let df = HealthBuilder::default()
            .row("2024-01-01", None, None, None, None, None)
            .row(
                "2024-01-02",
                Some(60),
                Some(28800),
                Some(8000),
                Some(30),
                Some(2000),
            )
            .build();
        let cleaned = clean(df);
        assert_eq!(cleaned.height(), 1);
    }

    #[test]
    fn test_dedupe_by_date_keeps_latest() {
        let df = HealthBuilder::default()
            .row(
                "2024-01-01",
                Some(60),
                Some(28800),
                Some(5000),
                Some(30),
                Some(2000),
            )
            .row(
                "2024-01-01",
                Some(65),
                Some(30000),
                Some(7000),
                Some(35),
                Some(2200),
            )
            .build();
        let cleaned = clean(df);
        assert_eq!(cleaned.height(), 1);
        assert_eq!(get_i32(&cleaned, "resting_hr", 0), Some(65));
        assert_eq!(get_i32(&cleaned, "steps", 0), Some(7000));
    }

    #[test]
    fn test_missing_columns_skipped() {
        // If a column doesn't exist in the input, the cleaning rule for it is skipped.
        let df = df!(
            "date" => &[Some(d("2024-01-01"))],
            "resting_hr" => &[Some(60_i32)],
        )
        .unwrap();
        // Should not panic.
        let cleaned = clean_daily_health(df.lazy()).unwrap().collect().unwrap();
        assert_eq!(cleaned.height(), 1);
    }
}
