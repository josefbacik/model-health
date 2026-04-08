//! Race retrospective and contrast analysis.
//!
//! This module is designed around the reality that we have very few races
//! (~10-20) but lots of surrounding daily-health and activity data. Instead
//! of trying to *predict* race performance, we use races as a small set of
//! ground-truth labeled outcomes and ask: when you ran your better races
//! (close to PR), what did the surrounding training and recovery look like,
//! versus when you ran your worse races?
//!
//! Pipeline:
//!   1. Load activities, parse `eventType.typeKey == "race"` from raw_json,
//!      filter to running types, classify into a standard distance bucket.
//!   2. Compute personal-record pace per bucket from the labeled set.
//!   3. For each race, compute a feature vector summarising the trailing
//!      training-and-recovery window (mileage, sleep, stress, HRV, RHR, etc.).
//!   4. Within each bucket, classify races as "good" (within `GOOD_PCT_OFF_PR`
//!      of PR pace) vs "bad" (more than `BAD_PCT_OFF_PR` off) and contrast
//!      the feature vectors.
//!
//! The good/bad thresholds are intentionally placeholders. Once enough races
//! are labeled in Garmin we'll look at the actual pace distribution and pick
//! sensible cuts; the structure of the contrast doesn't change.

use std::collections::BTreeMap;

use chrono::{Duration, NaiveDate};
use polars::prelude::*;
use serde_json::Value as JsonValue;
use tracing::{info, warn};

use crate::config::Config;
use crate::data;
use crate::error::{AppError, Result};

// ---------------------------------------------------------------------------
// Distance buckets
// ---------------------------------------------------------------------------

/// Standard race distances we know how to bucket. Anything outside these
/// ranges (8K, 15K, ultras, etc.) is currently ignored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DistanceBucket {
    FiveK,
    TenK,
    Half,
    Marathon,
}

impl DistanceBucket {
    pub fn label(&self) -> &'static str {
        match self {
            DistanceBucket::FiveK => "5K",
            DistanceBucket::TenK => "10K",
            DistanceBucket::Half => "Half Marathon",
            DistanceBucket::Marathon => "Marathon",
        }
    }

    /// Inclusive distance range (km) used to assign a running activity to
    /// this bucket. Bounds are deliberately generous on the upper side because
    /// GPS-measured race courses commonly come out a bit long.
    pub fn range_km(&self) -> (f64, f64) {
        match self {
            DistanceBucket::FiveK => (4.80, 5.40),
            DistanceBucket::TenK => (9.60, 10.80),
            DistanceBucket::Half => (20.50, 22.20),
            DistanceBucket::Marathon => (41.50, 43.50),
        }
    }

    /// Garmin's daily race-time predictor field name for this bucket.
    pub fn garmin_predictor_col(&self) -> &'static str {
        match self {
            DistanceBucket::FiveK => "race_5k_sec",
            DistanceBucket::TenK => "race_10k_sec",
            DistanceBucket::Half => "race_half_sec",
            DistanceBucket::Marathon => "race_marathon_sec",
        }
    }

    pub fn all() -> [DistanceBucket; 4] {
        [
            DistanceBucket::FiveK,
            DistanceBucket::TenK,
            DistanceBucket::Half,
            DistanceBucket::Marathon,
        ]
    }

    /// Classify a distance (km) into a bucket, or None if it doesn't match
    /// any standard race distance.
    pub fn classify(dist_km: f64) -> Option<Self> {
        for b in Self::all() {
            let (lo, hi) = b.range_km();
            if dist_km >= lo && dist_km <= hi {
                return Some(b);
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Race model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Race {
    #[allow(dead_code)] // kept for future cross-referencing back to the activities parquet
    pub activity_id: i64,
    pub date: NaiveDate,
    pub name: String,
    pub distance_km: f64,
    pub duration_sec: f64,
    pub bucket: DistanceBucket,
}

impl Race {
    pub fn pace_sec_per_km(&self) -> f64 {
        self.duration_sec / self.distance_km
    }
}

/// Format pace (seconds per km) as `M:SS`. Handles negative values for
/// pace deltas (e.g. "-0:15" means "15 seconds per km faster").
fn fmt_pace(sec_per_km: f64) -> String {
    let neg = sec_per_km < 0.0;
    let total = sec_per_km.abs().round() as i64;
    let m = total / 60;
    let s = total % 60;
    // Suppress the sign when the rounded value is zero so we don't print "-0:00"
    // for tiny negative deltas (e.g. -0.3 sec/km).
    let sign = if neg && total != 0 { "-" } else { "" };
    format!("{sign}{m}:{s:02}")
}

/// Format duration (seconds) as `H:MM:SS` or `MM:SS`.
fn fmt_time(sec: f64) -> String {
    let total = sec.round() as i64;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}

// ---------------------------------------------------------------------------
// Race loader
// ---------------------------------------------------------------------------

/// Activity types we treat as runs eligible to be races.
const RUNNING_TYPES: &[&str] = &["running", "treadmill_running", "virtual_run"];

/// Load all races from the activities parquet. A race is any running activity
/// whose raw_json `eventType.typeKey == "race"` and whose distance falls into
/// a recognized bucket.
pub fn load_races(config: &Config) -> Result<Vec<Race>> {
    let lf = data::load_activities(config)?;
    let df = lf.collect()?;

    let n = df.height();
    let ids = df.column("activity_id")?.i64()?.clone();
    let names = df.column("activity_name")?.str()?.clone();
    let types = df.column("activity_type")?.str()?.clone();
    let starts = df.column("start_time_local")?.datetime()?.clone();
    let durs = df.column("duration_sec")?.f64()?.clone();
    let dists = df.column("distance_m")?.f64()?.clone();
    let raws = df.column("raw_json")?.str()?.clone();

    let mut races = Vec::new();
    let mut skipped_distance = 0usize;
    let mut skipped_parse = 0usize;
    let mut skipped_tiny = 0usize;
    for i in 0..n {
        let act_type = types.get(i).unwrap_or("");
        if !RUNNING_TYPES.contains(&act_type) {
            continue;
        }
        let raw = raws.get(i).unwrap_or("");
        let parsed: JsonValue = match serde_json::from_str(raw) {
            Ok(v) => v,
            Err(_) => {
                skipped_parse += 1;
                continue;
            }
        };
        let is_race = parsed
            .get("eventType")
            .and_then(|v| v.get("typeKey"))
            .and_then(|v| v.as_str())
            .map(|s| s == "race")
            .unwrap_or(false);
        if !is_race {
            continue;
        }

        let dist_m = dists.get(i).unwrap_or(0.0);
        let dur = durs.get(i).unwrap_or(0.0);
        if dist_m < 100.0 || dur < 60.0 {
            // Implausibly small for a real race; almost certainly a logging
            // glitch. Warn so the user notices if a real race got dropped.
            let act_id = ids.get(i).unwrap_or(0);
            let name = names.get(i).unwrap_or("");
            warn!(
                activity_id = act_id,
                name = name,
                distance_m = dist_m,
                duration_sec = dur,
                "Skipping race-tagged activity with implausibly small distance/duration"
            );
            skipped_tiny += 1;
            continue;
        }
        let dist_km = dist_m / 1000.0;
        let Some(bucket) = DistanceBucket::classify(dist_km) else {
            skipped_distance += 1;
            continue;
        };

        // start_time_local is stored as a naive local timestamp in microseconds.
        // Treating it as UTC for the date extraction gives the correct local date.
        let micros = starts.get(i).unwrap_or(0);
        let date = chrono::DateTime::from_timestamp_micros(micros)
            .map(|dt| dt.naive_utc().date())
            .unwrap_or_else(|| NaiveDate::from_ymd_opt(1970, 1, 1).unwrap());

        races.push(Race {
            activity_id: ids.get(i).unwrap_or(0),
            date,
            name: names.get(i).unwrap_or("").to_string(),
            distance_km: dist_km,
            duration_sec: dur,
            bucket,
        });
    }

    races.sort_by_key(|r| r.date);

    info!(
        races = races.len(),
        skipped_unrecognized_distance = skipped_distance,
        skipped_unparseable_json = skipped_parse,
        skipped_implausibly_tiny = skipped_tiny,
        "Loaded races"
    );

    Ok(races)
}

// ---------------------------------------------------------------------------
// Personal records
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Pr {
    pub bucket: DistanceBucket,
    pub race: Race,
}

impl Pr {
    pub fn pace_sec_per_km(&self) -> f64 {
        self.race.pace_sec_per_km()
    }
}

/// Compute the PR (best pace) for each bucket from the labeled race set.
pub fn compute_prs(races: &[Race]) -> Vec<Pr> {
    let mut prs = Vec::new();
    for bucket in DistanceBucket::all() {
        let best = races.iter().filter(|r| r.bucket == bucket).min_by(|a, b| {
            a.pace_sec_per_km()
                .partial_cmp(&b.pace_sec_per_km())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some(best) = best {
            prs.push(Pr {
                bucket,
                race: best.clone(),
            });
        }
    }
    prs
}

// ---------------------------------------------------------------------------
// Race feature engineering
// ---------------------------------------------------------------------------

/// Names of the trailing-window features we compute per race. Order here
/// drives row order in the contrast table — stress + training volume sit
/// at the top because those are the levers the user actually controls.
const FEATURE_NAMES: &[&str] = &[
    // --- stress (the user's primary "is my life supporting training" signal) ---
    "stress_7d",
    "stress_28d",
    "stress_12w",
    "high_stress_days_28d_pct",
    // --- training volume ---
    "mileage_4w_km",
    "mileage_8w_km",
    "mileage_12w_km",
    "longest_run_8w_km",
    "n_runs_4w",
    "taper_pct",
    // --- sleep ---
    "sleep_hours_7d",
    "sleep_hours_28d",
    // --- recovery / autonomic ---
    "rhr_7d",
    "rhr_28d",
    "rhr_delta_7d_vs_28d",
    "hrv_7d",
    "hrv_28d",
    "hrv_12w",
    "hrv_delta_7d_vs_28d",
    "hrv_unbalanced_days_28d_pct",
    // --- body / fitness ---
    "weight_kg",
    "vo2max_at_race",
    "garmin_predicted_pace",
];

/// Garmin classifies daily average stress as: 0-25 resting, 26-50 low,
/// 51-75 medium, 76-100 high. We treat any day above this as "elevated".
const ELEVATED_STRESS_THRESHOLD: f64 = 50.0;

/// A feature vector keyed by feature name. Missing values stay as `None`
/// rather than getting silently zero-filled, because for the contrast we
/// want to *exclude* missing values from the comparison rather than treat
/// "no data" as "zero".
pub type FeatureMap = BTreeMap<String, Option<f64>>;

/// Bundled, pre-collected DataFrames used by per-race feature computation.
/// Loading these once and reusing across races avoids re-scanning parquet.
struct DataBundle {
    runs: DataFrame,           // running activities only, with date + distance_km
    daily: Option<DataFrame>,  // daily_health, cleaned
    perf: Option<DataFrame>,   // performance_metrics
    weight: Option<DataFrame>, // weight entries
}

impl DataBundle {
    fn load(config: &Config) -> Result<Self> {
        // --- Activities → running-only with a `date` column -----------------
        let acts = data::load_activities(config)?.collect()?;
        let runs = filter_runs_with_date(&acts)?;

        // --- Daily health (optional, but nearly always present) ------------
        let daily = match data::load_daily_health(config) {
            Ok(lf) => match crate::validation::clean_daily_health(lf) {
                Ok(cleaned) => match cleaned.sort(["date"], Default::default()).collect() {
                    Ok(df) => Some(df),
                    Err(e) => {
                        warn!(error = %e, "Could not collect daily_health for race features");
                        None
                    }
                },
                Err(e) => {
                    warn!(error = %e, "Could not clean daily_health for race features");
                    None
                }
            },
            Err(_) => None,
        };

        // --- Performance metrics -------------------------------------------
        let perf = match data::load_performance_metrics(config) {
            Ok(lf) => lf.sort(["date"], Default::default()).collect().ok(),
            Err(_) => None,
        };

        // --- Weight --------------------------------------------------------
        let weight = match data::load_weight(config) {
            Ok(lf) => lf.sort(["date"], Default::default()).collect().ok(),
            Err(_) => None,
        };

        Ok(Self {
            runs,
            daily,
            perf,
            weight,
        })
    }
}

/// Filter activities to running-only and add a `date` column derived from
/// `start_time_local`. Returned frame has columns: `date`, `distance_km`,
/// `duration_sec`.
fn filter_runs_with_date(activities: &DataFrame) -> Result<DataFrame> {
    let lf = activities.clone().lazy();
    // Build the filter from the same RUNNING_TYPES constant the race loader
    // uses, so the two stay in sync if we ever add e.g. "track_running".
    // Polars `is_in` is awkward across versions; OR-chain over the constant
    // is simple and version-stable.
    let type_filter = RUNNING_TYPES
        .iter()
        .map(|t| col("activity_type").eq(lit(*t)))
        .reduce(|a, b| a.or(b))
        .expect("RUNNING_TYPES must be non-empty");

    let df = lf
        .filter(type_filter)
        .with_columns([
            col("start_time_local").dt().date().alias("date"),
            (col("distance_m").cast(DataType::Float64) / lit(1000.0)).alias("distance_km"),
        ])
        .select([col("date"), col("distance_km"), col("duration_sec")])
        .collect()?;

    Ok(df)
}

/// Sum a numeric column for rows whose `date` is in `[start, end]` (inclusive).
fn window_sum(df: &DataFrame, value_col: &str, start: NaiveDate, end: NaiveDate) -> Option<f64> {
    let out = df
        .clone()
        .lazy()
        .filter(
            col("date")
                .gt_eq(lit(start))
                .and(col("date").lt_eq(lit(end))),
        )
        .select([col(value_col).cast(DataType::Float64).sum()])
        .collect()
        .ok()?;
    out.column(value_col).ok()?.f64().ok()?.get(0)
}

/// Mean of a numeric column over a date window. Returns None if no rows
/// or all values are null.
fn window_mean(df: &DataFrame, value_col: &str, start: NaiveDate, end: NaiveDate) -> Option<f64> {
    let out = df
        .clone()
        .lazy()
        .filter(
            col("date")
                .gt_eq(lit(start))
                .and(col("date").lt_eq(lit(end))),
        )
        .select([col(value_col).cast(DataType::Float64).mean()])
        .collect()
        .ok()?;
    out.column(value_col).ok()?.f64().ok()?.get(0)
}

/// Max of a numeric column over a date window.
fn window_max(df: &DataFrame, value_col: &str, start: NaiveDate, end: NaiveDate) -> Option<f64> {
    let out = df
        .clone()
        .lazy()
        .filter(
            col("date")
                .gt_eq(lit(start))
                .and(col("date").lt_eq(lit(end))),
        )
        .select([col(value_col).cast(DataType::Float64).max()])
        .collect()
        .ok()?;
    out.column(value_col).ok()?.f64().ok()?.get(0)
}

/// Count rows in a date window (any column will do — uses `date`).
fn window_count(df: &DataFrame, start: NaiveDate, end: NaiveDate) -> Option<f64> {
    let out = df
        .clone()
        .lazy()
        .filter(
            col("date")
                .gt_eq(lit(start))
                .and(col("date").lt_eq(lit(end))),
        )
        .select([col("date").count()])
        .collect()
        .ok()?;
    out.column("date")
        .ok()?
        .u32()
        .ok()?
        .get(0)
        .map(|v| v as f64)
}

/// Fraction of rows in `[start, end]` whose `value_col` is non-null and
/// strictly greater than `threshold`. Denominator is the count of non-null
/// rows for the column (so missing days don't dilute the rate).
fn window_fraction_above(
    df: &DataFrame,
    value_col: &str,
    threshold: f64,
    start: NaiveDate,
    end: NaiveDate,
) -> Option<f64> {
    let out = df
        .clone()
        .lazy()
        .filter(
            col("date")
                .gt_eq(lit(start))
                .and(col("date").lt_eq(lit(end)))
                .and(col(value_col).is_not_null()),
        )
        .select([
            col(value_col).count().alias("n_total"),
            col(value_col)
                .cast(DataType::Float64)
                .gt(lit(threshold))
                .cast(DataType::Float64)
                .sum()
                .alias("n_above"),
        ])
        .collect()
        .ok()?;
    let n_total = out.column("n_total").ok()?.u32().ok()?.get(0)? as f64;
    let n_above = out.column("n_above").ok()?.f64().ok()?.get(0)?;
    if n_total <= 0.0 {
        None
    } else {
        Some(n_above / n_total)
    }
}

/// Fraction of non-null rows in `[start, end]` whose string `status_col`
/// value matches any of the provided `accept` values. Garmin uses fixed
/// uppercase status strings (e.g. "BALANCED", "UNBALANCED", "LOW", "NONE")
/// so we match exactly. Denominator excludes rows where the status is null
/// so missing days don't dilute the rate.
fn window_fraction_status_in(
    df: &DataFrame,
    status_col: &str,
    accept: &[&str],
    start: NaiveDate,
    end: NaiveDate,
) -> Option<f64> {
    let collected = df
        .clone()
        .lazy()
        .filter(
            col("date")
                .gt_eq(lit(start))
                .and(col("date").lt_eq(lit(end)))
                .and(col(status_col).is_not_null()),
        )
        .select([col(status_col)])
        .collect()
        .ok()?;
    let s = collected.column(status_col).ok()?.str().ok()?.clone();
    // Iterate the column once. Denominator counts only rows where `get` returns
    // a real string — defends against any null sneaking past `is_not_null`.
    let mut n_total = 0usize;
    let mut n_hit = 0usize;
    for i in 0..s.len() {
        if let Some(v) = s.get(i) {
            n_total += 1;
            if accept.contains(&v) {
                n_hit += 1;
            }
        }
    }
    if n_total == 0 {
        return None;
    }
    Some(n_hit as f64 / n_total as f64)
}

/// Most recent non-null value of `value_col` in `[start, end]`.
fn window_last(df: &DataFrame, value_col: &str, start: NaiveDate, end: NaiveDate) -> Option<f64> {
    // `df` is pre-sorted by date in DataBundle::load, so no re-sort here.
    let out = df
        .clone()
        .lazy()
        .filter(
            col("date")
                .gt_eq(lit(start))
                .and(col("date").lt_eq(lit(end)))
                .and(col(value_col).is_not_null()),
        )
        .select([col(value_col).cast(DataType::Float64).last()])
        .collect()
        .ok()?;
    out.column(value_col).ok()?.f64().ok()?.get(0)
}

/// Compute the per-race feature vector. Race date `D` is excluded from all
/// windows (windows end at `D - 1`) so features only see *prior* data.
fn compute_race_features(bundle: &DataBundle, race: &Race) -> FeatureMap {
    let mut feats: FeatureMap = FEATURE_NAMES
        .iter()
        .map(|n| (n.to_string(), None))
        .collect();

    let day_before = race.date - Duration::days(1);
    let d4w = race.date - Duration::days(28);
    let d8w = race.date - Duration::days(56);
    let d12w = race.date - Duration::days(84);
    let d1w = race.date - Duration::days(7);

    // --- Activity-based volume features --------------------------------
    feats.insert(
        "mileage_4w_km".into(),
        window_sum(&bundle.runs, "distance_km", d4w, day_before),
    );
    feats.insert(
        "mileage_8w_km".into(),
        window_sum(&bundle.runs, "distance_km", d8w, day_before),
    );
    feats.insert(
        "mileage_12w_km".into(),
        window_sum(&bundle.runs, "distance_km", d12w, day_before),
    );
    feats.insert(
        "longest_run_8w_km".into(),
        window_max(&bundle.runs, "distance_km", d8w, day_before),
    );
    feats.insert(
        "n_runs_4w".into(),
        window_count(&bundle.runs, d4w, day_before),
    );

    // taper_pct: how much smaller was the last week than the average
    // of the prior 3 weeks? Positive value = larger taper.
    let last_week_km = window_sum(&bundle.runs, "distance_km", d1w, day_before);
    let prior_3w_km = {
        let prior_start = race.date - Duration::days(28);
        let prior_end = race.date - Duration::days(8);
        window_sum(&bundle.runs, "distance_km", prior_start, prior_end)
    };
    let taper_pct = match (last_week_km, prior_3w_km) {
        (Some(lw), Some(p3)) if p3 > 0.0 => Some(1.0 - (lw / (p3 / 3.0))),
        _ => None,
    };
    feats.insert("taper_pct".into(), taper_pct);

    // --- Daily-health-based recovery features --------------------------
    if let Some(daily) = bundle.daily.as_ref() {
        // sleep_hours derived from sleep_seconds
        let sleep_7d = window_mean(daily, "sleep_seconds", d1w, day_before).map(|s| s / 3600.0);
        let sleep_28d = window_mean(daily, "sleep_seconds", d4w, day_before).map(|s| s / 3600.0);
        feats.insert("sleep_hours_7d".into(), sleep_7d);
        feats.insert("sleep_hours_28d".into(), sleep_28d);

        feats.insert(
            "stress_7d".into(),
            window_mean(daily, "avg_stress", d1w, day_before),
        );
        feats.insert(
            "stress_28d".into(),
            window_mean(daily, "avg_stress", d4w, day_before),
        );
        feats.insert(
            "stress_12w".into(),
            window_mean(daily, "avg_stress", d12w, day_before),
        );
        feats.insert(
            "high_stress_days_28d_pct".into(),
            window_fraction_above(
                daily,
                "avg_stress",
                ELEVATED_STRESS_THRESHOLD,
                d4w,
                day_before,
            ),
        );

        let rhr_7d = window_mean(daily, "resting_hr", d1w, day_before);
        let rhr_28d = window_mean(daily, "resting_hr", d4w, day_before);
        feats.insert("rhr_7d".into(), rhr_7d);
        feats.insert("rhr_28d".into(), rhr_28d);
        feats.insert(
            "rhr_delta_7d_vs_28d".into(),
            match (rhr_7d, rhr_28d) {
                (Some(a), Some(b)) => Some(a - b),
                _ => None,
            },
        );

        let hrv_7d = window_mean(daily, "hrv_last_night", d1w, day_before);
        let hrv_28d = window_mean(daily, "hrv_last_night", d4w, day_before);
        feats.insert("hrv_7d".into(), hrv_7d);
        feats.insert("hrv_28d".into(), hrv_28d);
        feats.insert(
            "hrv_12w".into(),
            window_mean(daily, "hrv_last_night", d12w, day_before),
        );
        feats.insert(
            "hrv_delta_7d_vs_28d".into(),
            match (hrv_7d, hrv_28d) {
                (Some(a), Some(b)) => Some(a - b),
                _ => None,
            },
        );
        // Fraction of days in last 28 where Garmin tagged HRV as outside the
        // user's personal balanced range. Excludes "NONE" (no measurement).
        feats.insert(
            "hrv_unbalanced_days_28d_pct".into(),
            window_fraction_status_in(daily, "hrv_status", &["UNBALANCED", "LOW"], d4w, day_before),
        );
    }

    // --- Weight (most recent within 30d) -------------------------------
    if let Some(weight) = bundle.weight.as_ref() {
        let w_start = race.date - Duration::days(30);
        feats.insert(
            "weight_kg".into(),
            window_last(weight, "weight_kg", w_start, day_before),
        );
    }

    // --- VO2max + Garmin's own race predictor -------------------------
    if let Some(perf) = bundle.perf.as_ref() {
        let p_start = race.date - Duration::days(30);
        feats.insert(
            "vo2max_at_race".into(),
            window_last(perf, "vo2max", p_start, day_before),
        );
        // Garmin's predictor for this race's bucket, expressed as pace per km
        // so it's directly comparable to the actual race pace.
        let pred_col = race.bucket.garmin_predictor_col();
        let pred_sec = window_last(perf, pred_col, p_start, day_before);
        let nominal_km = match race.bucket {
            DistanceBucket::FiveK => 5.0,
            DistanceBucket::TenK => 10.0,
            DistanceBucket::Half => 21.0975,
            DistanceBucket::Marathon => 42.195,
        };
        feats.insert(
            "garmin_predicted_pace".into(),
            pred_sec.map(|s| s / nominal_km),
        );
    }

    // Catch typo'd inserts in dev builds: an insert with a key that isn't in
    // FEATURE_NAMES will silently miss the output table (the printer iterates
    // FEATURE_NAMES, not the map). Fails loudly here at the call site.
    debug_assert!(
        feats.keys().all(|k| FEATURE_NAMES.iter().any(|n| n == k)),
        "compute_race_features inserted a key not present in FEATURE_NAMES"
    );

    feats
}

// ---------------------------------------------------------------------------
// Race classification (good / bad / neutral)
// ---------------------------------------------------------------------------

/// Within `GOOD_PCT_OFF_PR` of PR pace counts as a "good" race.
const GOOD_PCT_OFF_PR: f64 = 0.05;
/// More than `BAD_PCT_OFF_PR` off PR pace counts as a "bad" race.
const BAD_PCT_OFF_PR: f64 = 0.10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quality {
    Good,
    Neutral,
    Bad,
}

impl Quality {
    pub fn label(&self) -> &'static str {
        match self {
            Quality::Good => "good",
            Quality::Neutral => "—",
            Quality::Bad => "bad",
        }
    }
}

/// Classify a race relative to its bucket's PR.
pub fn classify(race: &Race, pr_pace: f64) -> Quality {
    let pct_off = (race.pace_sec_per_km() - pr_pace) / pr_pace;
    if pct_off <= GOOD_PCT_OFF_PR {
        Quality::Good
    } else if pct_off >= BAD_PCT_OFF_PR {
        Quality::Bad
    } else {
        Quality::Neutral
    }
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

/// Top-level entry point for the `model-health races` subcommand.
pub fn run(config: &Config) -> Result<()> {
    let races = load_races(config)?;
    if races.is_empty() {
        return Err(AppError::Data(
            "No races found in activities. Tag races in Garmin Connect (set Event Type to \"Race\") then re-fetch with `model-health fetch --force`.".into(),
        ));
    }

    let prs = compute_prs(&races);

    println!(
        "Loaded {} races across {} buckets.\n",
        races.len(),
        prs.len()
    );
    print_pr_table(&prs);
    println!();

    println!("Loading per-race features (mileage, sleep, stress, HRV, RHR, weight, VO2max)...");
    let bundle = DataBundle::load(config)?;

    for bucket in DistanceBucket::all() {
        let bucket_races: Vec<&Race> = races.iter().filter(|r| r.bucket == bucket).collect();
        if bucket_races.is_empty() {
            continue;
        }
        let pr = prs.iter().find(|p| p.bucket == bucket);
        print_bucket_section(&bundle, bucket, &bucket_races, pr);
    }

    Ok(())
}

fn print_pr_table(prs: &[Pr]) {
    println!("Personal records (from labeled races only):");
    for pr in prs {
        println!(
            "  {:<14} {}  ({}, {})",
            pr.bucket.label(),
            fmt_pace(pr.pace_sec_per_km()),
            fmt_time(pr.race.duration_sec),
            pr.race.date,
        );
    }
}

fn print_bucket_section(
    bundle: &DataBundle,
    bucket: DistanceBucket,
    races: &[&Race],
    pr: Option<&Pr>,
) {
    println!("\n=== {} ({} races) ===", bucket.label(), races.len());
    let Some(pr) = pr else {
        println!("  No PR available.");
        return;
    };
    let pr_pace = pr.pace_sec_per_km();

    // Per-race summary line.
    println!(
        "  {:<10}  {:>6}  {:>7}  {:>7}  {:<7}  name",
        "date", "pace", "time", "%off", "class"
    );
    let mut sorted = races.to_vec();
    sorted.sort_by(|a, b| {
        a.pace_sec_per_km()
            .partial_cmp(&b.pace_sec_per_km())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for r in &sorted {
        let pct = (r.pace_sec_per_km() - pr_pace) / pr_pace * 100.0;
        let q = classify(r, pr_pace);
        println!(
            "  {:<10}  {:>6}  {:>7}  {:>+6.1}%  {:<7}  {}",
            r.date.to_string(),
            fmt_pace(r.pace_sec_per_km()),
            fmt_time(r.duration_sec),
            pct,
            q.label(),
            truncate(&r.name, 50),
        );
    }

    if races.len() < 2 {
        println!("  (need ≥2 races in this bucket for a contrast)");
        return;
    }

    // Compute features for every race in this bucket.
    let with_feats: Vec<(&Race, FeatureMap, Quality)> = sorted
        .iter()
        .map(|r| {
            let f = compute_race_features(bundle, r);
            let q = classify(r, pr_pace);
            (*r, f, q)
        })
        .collect();

    print_feature_table(&with_feats);

    let n_good = with_feats
        .iter()
        .filter(|(_, _, q)| *q == Quality::Good)
        .count();
    let n_bad = with_feats
        .iter()
        .filter(|(_, _, q)| *q == Quality::Bad)
        .count();
    if n_good > 0 && n_bad > 0 {
        print_contrast(&with_feats);
    } else {
        println!(
            "\n  (no good-vs-bad contrast: currently {n_good} good, {n_bad} bad. Threshold: good ≤{:.0}% off PR, bad ≥{:.0}% off PR.)",
            GOOD_PCT_OFF_PR * 100.0,
            BAD_PCT_OFF_PR * 100.0,
        );
    }

    // Pace-correlation report runs regardless of good/bad split, since it's
    // useful even when every race classifies the same way.
    print_pace_correlations(&with_feats);
}

/// Wide-format table: features as rows, races as columns. Useful for eyeballing
/// "what changed across this bucket's races".
fn print_feature_table(rows: &[(&Race, FeatureMap, Quality)]) {
    println!("\n  Per-race features (rows = features, cols = races, sorted by pace):");
    // Header row: race dates
    print!("    {:<22}", "feature");
    for (r, _, q) in rows {
        let tag = match q {
            Quality::Good => "★",
            Quality::Bad => "✗",
            Quality::Neutral => " ",
        };
        print!(" {:>11}", format!("{tag}{}", r.date));
    }
    println!();

    for fname in FEATURE_NAMES {
        print!("    {:<22}", fname);
        for (_, feats, _) in rows {
            let v = feats.get(*fname).and_then(|o| *o);
            print!(" {:>11}", fmt_feature(fname, v));
        }
        println!();
    }
}

/// Pretty-print a feature value, picking units appropriate to the feature.
fn fmt_feature(name: &str, v: Option<f64>) -> String {
    let Some(v) = v else {
        return "—".into();
    };
    match name {
        "garmin_predicted_pace" => fmt_pace(v),
        "taper_pct" | "high_stress_days_28d_pct" | "hrv_unbalanced_days_28d_pct" => {
            format!("{:>+6.1}%", v * 100.0)
        }
        "n_runs_4w" => format!("{v:.0}"),
        _ => format!("{v:.1}"),
    }
}

/// For each feature, print the mean over good races vs bad races and the
/// delta. This is the "what separated good from bad" view.
fn print_contrast(rows: &[(&Race, FeatureMap, Quality)]) {
    let goods: Vec<&FeatureMap> = rows
        .iter()
        .filter(|(_, _, q)| *q == Quality::Good)
        .map(|(_, f, _)| f)
        .collect();
    let bads: Vec<&FeatureMap> = rows
        .iter()
        .filter(|(_, _, q)| *q == Quality::Bad)
        .map(|(_, f, _)| f)
        .collect();

    println!(
        "\n  Contrast (mean of {} good vs mean of {} bad races):",
        goods.len(),
        bads.len()
    );
    println!(
        "    {:<22} {:>10} {:>10} {:>10}",
        "feature", "good μ", "bad μ", "delta"
    );

    for fname in FEATURE_NAMES {
        let good_mean = mean_of(&goods, fname);
        let bad_mean = mean_of(&bads, fname);
        let delta = match (good_mean, bad_mean) {
            (Some(g), Some(b)) => Some(g - b),
            _ => None,
        };
        println!(
            "    {:<22} {:>10} {:>10} {:>10}",
            fname,
            fmt_feature(fname, good_mean),
            fmt_feature(fname, bad_mean),
            fmt_feature(fname, delta),
        );
    }
}

/// Drop pairs containing any non-finite value (NaN, ±inf). A single NaN
/// upstream poisons every accumulator-based stat (mean, variance, slope) and
/// the `denom == 0.0` guards in pearson/linreg won't catch it because
/// `NaN != 0.0`. Filtering once here keeps both helpers safe.
fn finite_pairs(xs: &[f64], ys: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = xs.len().min(ys.len());
    let mut fx = Vec::with_capacity(n);
    let mut fy = Vec::with_capacity(n);
    for i in 0..n {
        if xs[i].is_finite() && ys[i].is_finite() {
            fx.push(xs[i]);
            fy.push(ys[i]);
        }
    }
    (fx, fy)
}

/// Pearson correlation coefficient. Returns None if either input has
/// insufficient samples or zero variance.
fn pearson(xs: &[f64], ys: &[f64]) -> Option<f64> {
    let (xs, ys) = finite_pairs(xs, ys);
    let n = xs.len();
    if n < 2 {
        return None;
    }
    let nf = n as f64;
    let mx = xs.iter().sum::<f64>() / nf;
    let my = ys.iter().sum::<f64>() / nf;
    let mut num = 0.0;
    let mut dxs = 0.0;
    let mut dys = 0.0;
    for (x, y) in xs.iter().zip(ys.iter()) {
        let dx = x - mx;
        let dy = y - my;
        num += dx * dy;
        dxs += dx * dx;
        dys += dy * dy;
    }
    let denom = (dxs * dys).sqrt();
    if denom == 0.0 {
        None
    } else {
        Some(num / denom)
    }
}

/// Single-variable linear regression: y ~= slope*x + intercept.
fn linreg(xs: &[f64], ys: &[f64]) -> Option<(f64, f64)> {
    let (xs, ys) = finite_pairs(xs, ys);
    let n = xs.len();
    if n < 2 {
        return None;
    }
    let nf = n as f64;
    let mx = xs.iter().sum::<f64>() / nf;
    let my = ys.iter().sum::<f64>() / nf;
    let mut num = 0.0;
    let mut den = 0.0;
    for (x, y) in xs.iter().zip(ys.iter()) {
        let dx = x - mx;
        num += dx * (y - my);
        den += dx * dx;
    }
    if den == 0.0 {
        return None;
    }
    let slope = num / den;
    let intercept = my - slope * mx;
    Some((slope, intercept))
}

/// Per-bucket "which features track race pace" report. For each feature
/// with ≥2 non-null observations across the bucket's races, compute Pearson
/// correlation with pace and the slope (seconds-per-km change per unit
/// change in the feature). Sorted by |r|.
///
/// This is intentionally single-variable rather than multivariate: with
/// 3-10 races per bucket, fitting a multi-feature regression overfits. The
/// honest version is "here are the features that move with race pace, one
/// at a time" — and the strongest one becomes the candidate input for a
/// future `pace-target` tool.
fn print_pace_correlations(rows: &[(&Race, FeatureMap, Quality)]) {
    let n_races = rows.len();
    println!("\n  Single-variable relationships with race pace (n={n_races} races in bucket):");
    println!(
        "    {:<26} {:>3} {:>6} {:>20}",
        "feature", "n", "r", "slope (s/km per unit)"
    );

    let mut results: Vec<(String, usize, f64, f64)> = Vec::new();
    for fname in FEATURE_NAMES {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for (r, feats, _) in rows {
            if let Some(Some(v)) = feats.get(*fname) {
                xs.push(*v);
                ys.push(r.pace_sec_per_km());
            }
        }
        // Require n≥3: with only 2 points, r is mathematically always ±1
        // (any 2 points lie exactly on a line), which is meaningless noise.
        if xs.len() < 3 {
            continue;
        }
        let r_val = match pearson(&xs, &ys) {
            Some(v) if v.is_finite() => v,
            _ => continue,
        };
        let slope = match linreg(&xs, &ys) {
            Some((s, _)) if s.is_finite() => s,
            _ => continue,
        };
        results.push((fname.to_string(), xs.len(), r_val, slope));
    }

    if results.is_empty() {
        println!("    (no features with enough non-null observations)");
        return;
    }

    // Sort by absolute correlation, strongest first.
    results.sort_by(|a, b| {
        b.2.abs()
            .partial_cmp(&a.2.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (name, n, r, slope) in results.iter().take(12) {
        println!("    {name:<26} {n:>3} {r:>+6.2} {slope:>+20.2}");
    }
    println!(
        "    (r is Pearson correlation; positive slope means a *higher* feature value\n     is associated with a *slower* race pace. n is races where the feature is\n     non-null. With small n, treat all of this as directional only.)"
    );
}

fn mean_of(vecs: &[&FeatureMap], fname: &str) -> Option<f64> {
    let vals: Vec<f64> = vecs
        .iter()
        .filter_map(|m| m.get(fname).and_then(|o| *o))
        .collect();
    if vals.is_empty() {
        None
    } else {
        Some(vals.iter().sum::<f64>() / vals.len() as f64)
    }
}

fn truncate(s: &str, max: usize) -> String {
    // Slice by char boundary, not byte index — race names contain multi-byte
    // characters (e.g. "Disney World®") that would panic on `&s[..n]`.
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let cutoff = max.saturating_sub(1);
        let prefix: String = s.chars().take(cutoff).collect();
        format!("{prefix}…")
    }
}
