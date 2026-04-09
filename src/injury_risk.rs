//! Injury-risk assessment based on training load and physiological signals.
//!
//! Computes weekly aggregates of running volume, intensity (training effect,
//! hard-run count), and physiological response (resting HR, Garmin stress,
//! sleep), then applies a tiered warning system derived from historical
//! injury-gap analysis:
//!
//!   **Red**    — Stress Δ > +3 AND (RHR Δ > +1 OR hard runs ≥ 3)
//!   **Yellow** — Stress Δ > +3 alone
//!   **Volume** — 4-week / 8-week volume ACWR > 1.5
//!
//! Run with `model-health injury-risk` to see the last 12 weeks plus a
//! current-week assessment.

use polars::prelude::*;

use crate::config::Config;
use crate::data;
use crate::data::RUNNING_TYPES;
use crate::error::Result;
use crate::validation;

/// Number of recent weeks to display in the report (plus the current partial
/// week if data exists for it).
const DISPLAY_WEEKS: usize = 12;

/// Entry point.
pub fn run(config: &Config) -> Result<()> {
    // --- Load and clean daily health ------------------------------------
    let daily_lf = data::load_daily_health(config)?;
    let daily_lf = validation::clean_daily_health(daily_lf)?;

    // Weekly health aggregates: RHR, stress, sleep, body battery, HRV.
    // Week boundaries are Monday-based ISO weeks (Polars truncate("1w")
    // default), matching Garmin's weekly summaries.
    let health_weekly = daily_lf
        .with_column(col("date").dt().truncate(lit("1w")).alias("week"))
        .with_column(
            (col("sleep_seconds").cast(DataType::Float64) / lit(3600.0)).alias("sleep_hours"),
        )
        .group_by([col("week")])
        .agg([
            col("resting_hr")
                .cast(DataType::Float64)
                .mean()
                .alias("rhr_avg"),
            col("avg_stress")
                .cast(DataType::Float64)
                .mean()
                .alias("stress_avg"),
            col("body_battery_start")
                .cast(DataType::Float64)
                .mean()
                .alias("bb_start_avg"),
            col("sleep_hours").mean().alias("sleep_avg"),
            col("hrv_last_night")
                .cast(DataType::Float64)
                .mean()
                .alias("hrv_avg"),
        ])
        .sort(["week"], Default::default());

    // --- Load activities and compute weekly intensity --------------------
    let acts_lf = data::load_activities(config)?;
    let is_run = data::type_in_list(RUNNING_TYPES);

    let intensity_weekly = acts_lf
        .with_column(col("start_time_local").dt().date().alias("date"))
        .with_column(col("date").dt().truncate(lit("1w")).alias("week"))
        .filter(is_run)
        .with_column((col("distance_m").cast(DataType::Float64) / lit(1000.0)).alias("distance_km"))
        .group_by([col("week")])
        .agg([
            col("distance_km").sum().alias("weekly_km"),
            // Use len() not count() so null distances don't cause run_count
            // to disagree with hard_runs (count() excludes nulls).
            len().alias("run_count"),
            col("avg_hr")
                .cast(DataType::Float64)
                .mean()
                .alias("avg_hr_mean"),
            col("training_effect")
                .cast(DataType::Float64)
                .mean()
                .alias("te_mean"),
            col("training_effect")
                .cast(DataType::Float64)
                .max()
                .alias("te_max"),
            // Count hard runs (TE >= 3.0).
            col("training_effect")
                .cast(DataType::Float64)
                .gt_eq(lit(3.0))
                .sum()
                .cast(DataType::UInt32)
                .alias("hard_runs"),
            col("training_load")
                .cast(DataType::Float64)
                .sum()
                .alias("tl_sum"),
        ])
        .sort(["week"], Default::default());

    // --- Join health + intensity on week --------------------------------
    let combined = health_weekly
        .join(
            intensity_weekly,
            [col("week")],
            [col("week")],
            JoinArgs::new(JoinType::Full).with_coalesce(JoinCoalesce::CoalesceColumns),
        )
        .with_columns([
            col("weekly_km").fill_null(lit(0.0)),
            col("run_count").fill_null(lit(0u32)),
            col("hard_runs").fill_null(lit(0u32)),
            col("tl_sum").fill_null(lit(0.0)),
        ])
        .sort(["week"], Default::default());

    // --- Rolling baselines (8-week) and derived deltas ------------------
    let rolling_8w = RollingOptionsFixedWindow {
        window_size: 8,
        min_periods: 4,
        ..Default::default()
    };
    let rolling_4w = RollingOptionsFixedWindow {
        window_size: 4,
        min_periods: 2,
        ..Default::default()
    };

    let with_baselines = combined
        .with_columns([
            col("rhr_avg")
                .rolling_mean(rolling_8w.clone())
                .alias("rhr_8wk"),
            col("stress_avg")
                .rolling_mean(rolling_8w.clone())
                .alias("stress_8wk"),
            col("sleep_avg")
                .rolling_mean(rolling_8w.clone())
                .alias("sleep_8wk"),
            col("weekly_km")
                .rolling_mean(rolling_4w.clone())
                .alias("km_4wk"),
            col("weekly_km")
                .rolling_mean(rolling_8w.clone())
                .alias("km_8wk"),
        ])
        .with_columns([
            (col("rhr_avg") - col("rhr_8wk")).alias("rhr_delta"),
            (col("stress_avg") - col("stress_8wk")).alias("stress_delta"),
            (col("sleep_avg") - col("sleep_8wk")).alias("sleep_delta"),
            // Guard against division by zero when coming back from a long
            // break (km_8wk == 0). Without this, inf would false-trigger
            // the Volume warning after every injury gap.
            when(col("km_8wk").gt(lit(0.0)))
                .then(col("km_4wk") / col("km_8wk"))
                .otherwise(lit(NULL))
                .alias("vol_acwr"),
        ])
        .collect()?;

    // --- Render report --------------------------------------------------
    let n_rows = with_baselines.height();
    let start = n_rows.saturating_sub(DISPLAY_WEEKS);
    let tail = with_baselines.slice(start as i64, n_rows - start);

    print_header();

    // Track whether any warnings fired.
    let mut any_warning = false;

    for idx in 0..tail.height() {
        let row = Row::extract(&tail, idx);
        let level = row.warning_level();
        if level != WarningLevel::None {
            any_warning = true;
        }
        print_row(&row, &level);
    }

    println!();

    // --- Current-week summary -------------------------------------------
    if let Some(last) = Row::try_last(&tail) {
        let level = last.warning_level();
        print_assessment(&last, &level);

        if !any_warning {
            println!("No warning signals in the last {DISPLAY_WEEKS} weeks. Keep it up!");
        }
    }

    Ok(())
}

// -----------------------------------------------------------------------
// Warning-level logic
// -----------------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
enum WarningLevel {
    Red,
    Yellow,
    Volume,
    None,
}

impl WarningLevel {
    fn label(&self) -> &'static str {
        match self {
            WarningLevel::Red => "RED",
            WarningLevel::Yellow => "YEL",
            WarningLevel::Volume => "VOL",
            WarningLevel::None => "",
        }
    }

    fn marker(&self) -> &'static str {
        match self {
            WarningLevel::Red => " !!",
            WarningLevel::Yellow => "  !",
            WarningLevel::Volume => "  ~",
            WarningLevel::None => "",
        }
    }
}

// -----------------------------------------------------------------------
// Per-row extraction helper
// -----------------------------------------------------------------------

/// A single week's worth of signals, extracted from the combined DataFrame.
struct Row {
    week: String,
    weekly_km: f64,
    run_count: u32,
    hard_runs: u32,
    te_max: Option<f64>,
    stress_delta: Option<f64>,
    rhr_delta: Option<f64>,
    sleep_delta: Option<f64>,
    vol_acwr: Option<f64>,
    // Absolutes (for the assessment block).
    rhr_avg: Option<f64>,
    stress_avg: Option<f64>,
    bb_start_avg: Option<f64>,
    hrv_avg: Option<f64>,
}

impl Row {
    fn extract(df: &DataFrame, idx: usize) -> Self {
        let f64_opt = |col_name: &str| -> Option<f64> {
            df.column(col_name)
                .ok()
                .and_then(|s| s.f64().ok())
                .and_then(|ca| ca.get(idx))
        };
        let u32_val = |col_name: &str| -> u32 {
            df.column(col_name)
                .ok()
                .and_then(|s| s.u32().ok())
                .and_then(|ca| ca.get(idx))
                .unwrap_or(0)
        };
        // Polars Date is days-since-Unix-epoch; convert via chrono.
        let week = df
            .column("week")
            .ok()
            .and_then(|s| s.date().ok())
            .and_then(|ca| {
                ca.get(idx).map(|days| {
                    chrono::NaiveDate::from_num_days_from_ce_opt(
                        // 719_163 = days from CE epoch (0001-01-01) to Unix
                        // epoch (1970-01-01).
                        days + 719_163,
                    )
                    .unwrap_or_default()
                    .to_string()
                })
            })
            .unwrap_or_else(|| "?".into());

        Row {
            week,
            weekly_km: f64_opt("weekly_km").unwrap_or(0.0),
            run_count: u32_val("run_count"),
            hard_runs: u32_val("hard_runs"),
            te_max: f64_opt("te_max"),
            stress_delta: f64_opt("stress_delta"),
            rhr_delta: f64_opt("rhr_delta"),
            sleep_delta: f64_opt("sleep_delta"),
            vol_acwr: f64_opt("vol_acwr"),
            rhr_avg: f64_opt("rhr_avg"),
            stress_avg: f64_opt("stress_avg"),
            bb_start_avg: f64_opt("bb_start_avg"),
            hrv_avg: f64_opt("hrv_avg"),
        }
    }

    fn try_last(df: &DataFrame) -> Option<Self> {
        if df.height() == 0 {
            None
        } else {
            Some(Self::extract(df, df.height() - 1))
        }
    }

    fn warning_level(&self) -> WarningLevel {
        let stress_high = self.stress_delta.is_some_and(|d| d > 3.0);
        let rhr_high = self.rhr_delta.is_some_and(|d| d > 1.0);
        let many_hard = self.hard_runs >= 3;
        let was_running = self.weekly_km > 5.0;

        // Only flag training-related injury risk when there's actual training
        // happening. Stress spikes during rest weeks are life stress, not
        // injury risk.

        // Red: stress elevated AND (RHR elevated OR lots of hard runs)
        // while actively training.
        if was_running && stress_high && (rhr_high || many_hard) {
            return WarningLevel::Red;
        }

        // Yellow: stress elevated while running.
        if was_running && stress_high {
            return WarningLevel::Yellow;
        }

        // Volume guard: 4-week avg is >50% above 8-week avg.
        if self.vol_acwr.is_some_and(|a| a > 1.5) {
            return WarningLevel::Volume;
        }

        WarningLevel::None
    }
}

// -----------------------------------------------------------------------
// Printing
// -----------------------------------------------------------------------

fn print_header() {
    println!("=== Injury Risk Assessment ===\n");
    println!(
        "Thresholds — RED: stress +3 AND (RHR +1 OR hard>=3)  |  \
         YEL: stress +3  |  VOL: ACWR > 1.5\n"
    );
    println!(
        "  {:<12} {:>6} {:>5} {:>5} {:>6} {:>7} {:>7} {:>7} {:>6}  Alert",
        "Week", "km", "Runs", "Hard", "TEmax", "Str +/-", "RHR +/-", "Slp +/-", "ACWR"
    );
    println!("  {}", "-".repeat(80));
}

fn print_row(row: &Row, level: &WarningLevel) {
    let te = row.te_max.map_or("     -".into(), |v| format!("{v:>6.1}"));
    let sd = row
        .stress_delta
        .map_or("      -".into(), |v| format!("{v:>+7.1}"));
    let rd = row
        .rhr_delta
        .map_or("      -".into(), |v| format!("{v:>+7.1}"));
    let sl = row
        .sleep_delta
        .map_or("      -".into(), |v| format!("{v:>+7.1}"));
    let ac = row
        .vol_acwr
        .map_or("     -".into(), |v| format!("{v:>6.2}"));

    println!(
        "  {:<12} {:>6.1} {:>5} {:>5} {} {} {} {} {}  {:<3}{}",
        row.week,
        row.weekly_km,
        row.run_count,
        row.hard_runs,
        te,
        sd,
        rd,
        sl,
        ac,
        level.label(),
        level.marker(),
    );
}

fn print_assessment(row: &Row, level: &WarningLevel) {
    println!("--- Most recent week: {} ---\n", row.week);

    // Absolute values.
    let rhr = row.rhr_avg.map_or("-".into(), |v| format!("{v:.0}"));
    let stress = row.stress_avg.map_or("-".into(), |v| format!("{v:.0}"));
    let bb = row.bb_start_avg.map_or("-".into(), |v| format!("{v:.0}"));
    let hrv = row.hrv_avg.map_or("-".into(), |v| format!("{v:.0}"));

    println!(
        "  Volume:  {:.1} km across {} runs ({} hard)",
        row.weekly_km, row.run_count, row.hard_runs
    );
    println!(
        "  RHR:     {} bpm (delta {})",
        rhr,
        row.rhr_delta.map_or("-".into(), |v| format!("{v:+.1}"))
    );
    println!(
        "  Stress:  {} (delta {})",
        stress,
        row.stress_delta.map_or("-".into(), |v| format!("{v:+.1}"))
    );
    println!(
        "  Sleep:   delta {}",
        row.sleep_delta
            .map_or("-".into(), |v| format!("{v:+.1} hrs"))
    );
    println!("  BB:      {}", bb);
    println!("  HRV:     {}", hrv);
    println!(
        "  ACWR:    {}",
        row.vol_acwr.map_or("-".into(), |v| format!("{v:.2}"))
    );
    println!();

    match level {
        WarningLevel::Red => {
            println!("  !! RED ALERT: Stress is elevated AND you have high intensity/RHR.");
            println!("     Recommendation: Take 2-3 easy days. Skip hard workouts.");
            println!("     This pattern has preceded injuries historically.");
        }
        WarningLevel::Yellow => {
            println!("   ! YELLOW: Stress is elevated above your 8-week baseline.");
            println!("     Recommendation: Run easy this week. Skip threshold/VO2max work.");
            println!("     Avoid racing when stress is elevated — intensity on a stressed");
            println!("     body is a common injury trigger.");
        }
        WarningLevel::Volume => {
            println!("   ~ VOLUME WARNING: Your recent volume is ramping faster than your base.");
            println!("     Recommendation: Hold or slightly reduce mileage this week.");
            println!("     Let your 8-week baseline catch up before adding more.");
        }
        WarningLevel::None => {
            println!("  All clear. No warning signals this week.");
        }
    }
    println!();
}
