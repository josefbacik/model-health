//! Cardiac Efficiency (CE) fitness tracking.
//!
//! Computes a grade-adjusted, temperature-normalized "speed per heartbeat"
//! metric for each running activity, then displays trends, current fitness
//! state, and what training/recovery factors drive changes.
//!
//! The grade adjustment uses the Minetti et al. (2002) metabolic cost of
//! running formula so that hilly runs are directly comparable to flat ones.
//!
//! Run with `model-health fitness` to see the report.

use chrono::Datelike;
use polars::prelude::*;
use std::collections::HashMap;

use crate::config::Config;
use crate::data;
use crate::data::RUNNING_TYPES;
use crate::error::Result;
use crate::validation;

/// Temperature coefficient: efficiency change per °C.
/// Negative means hotter = less efficient.
/// Derived from OLS on ~700 runs (2020-2026): eff = 0.0193 - 0.000111*temp,
/// r=-0.32. Effect is ~0.001 CE per 10°C.
pub const TEMP_COEFF: f64 = -0.000111;

/// Reference temperature for normalization (°C).
pub const TEMP_REF: f64 = 20.0;

/// Minimum seconds of steady-state data to compute CE for a run.
const MIN_STEADY_SECONDS: usize = 30;

/// Warmup seconds to skip at the start of each run.
const WARMUP_SECONDS: f64 = 300.0;

/// Minimum speed (m/s) to count as running (filters out walking/standing).
const MIN_SPEED: f64 = 1.5;

/// Minimum heart rate to include (filters out sensor dropouts).
const MIN_HR: f64 = 100.0;

/// Number of recent runs to show individually in the report.
const RECENT_RUNS: usize = 10;

/// Maximum heart rate coefficient of variation (std/mean * 100) to consider
/// a run "steady" for drift analysis. Interval workouts produce large HR
/// swings (CV > 8.5%) from hard/easy alternation, while steady runs on hilly
/// terrain are typically 5-7.5% (hills affect speed but not HR as much).
pub const MAX_HR_CV_PCT: f64 = 8.5;

/// Per-second row of processed running data (post-warmup, filtered, grade-adjusted).
pub struct SteadyRow {
    pub elapsed: f64,
    pub gap_speed: f64,
    pub heart_rate: f64,
    pub temperature: Option<f64>,
    pub cadence: Option<f64>,
    pub gct: Option<f64>,
    pub stride: Option<f64>,
}

/// Load a detail parquet and extract filtered, grade-adjusted per-second rows.
/// Handles altitude smoothing, warmup exclusion, speed/HR filtering, and
/// Minetti grade adjustment. Returns None if the file lacks required columns
/// or has too little data.
pub fn load_steady_rows(detail_path: &std::path::Path) -> Option<Vec<SteadyRow>> {
    let df = LazyFrame::scan_parquet(detail_path.to_string_lossy().as_ref(), Default::default())
        .ok()?
        .collect()
        .ok()?;

    if df.height() < 60 {
        return None;
    }

    let col_f64 =
        |name: &str| -> Option<Column> { df.column(name).ok()?.cast(&DataType::Float64).ok() };

    // Required columns
    let altitude = col_f64("altitude")?;
    let distance = col_f64("distance")?;
    let speed = col_f64("speed")?;
    let hr = col_f64("heart_rate")?;
    let elapsed = col_f64("elapsed_sec")?;

    // Optional columns
    let temperature = col_f64("temperature");
    let cadence = col_f64("cadence");
    let gct = col_f64("ground_contact_time");
    let stride = col_f64("stride_length");

    let alt_f = altitude.f64().ok()?;
    let dist_f = distance.f64().ok()?;
    let speed_f = speed.f64().ok()?;
    let hr_f = hr.f64().ok()?;
    let elapsed_f = elapsed.f64().ok()?;
    let temp_f = temperature.as_ref().and_then(|c| c.f64().ok());
    let cadence_f = cadence.as_ref().and_then(|c| c.f64().ok());
    let gct_f = gct.as_ref().and_then(|c| c.f64().ok());
    let stride_f = stride.as_ref().and_then(|c| c.f64().ok());

    let n = df.height();

    // Smoothed altitude (15-sec rolling mean)
    let mut alt_smooth = vec![0.0f64; n];
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        let start = i.saturating_sub(7);
        let end = (i + 8).min(n);
        let mut sum = 0.0;
        let mut count = 0;
        for j in start..end {
            if let Some(a) = alt_f.get(j) {
                sum += a;
                count += 1;
            }
        }
        alt_smooth[i] = if count > 0 { sum / count as f64 } else { 0.0 };
    }

    let mut rows = Vec::new();
    for i in 1..n {
        let Some(spd) = speed_f.get(i) else { continue };
        let Some(heart) = hr_f.get(i) else { continue };
        let Some(el) = elapsed_f.get(i) else { continue };
        if el < WARMUP_SECONDS || spd < MIN_SPEED || heart < MIN_HR {
            continue;
        }
        let d_dist = match (dist_f.get(i), dist_f.get(i - 1)) {
            (Some(a), Some(b)) if (a - b) >= 1.0 => a - b,
            _ => continue,
        };
        let d_alt = alt_smooth[i] - alt_smooth[i - 1];
        let grade = (d_alt / d_dist).clamp(-0.45, 0.45);
        let cost = minetti_cost(grade).max(1.0);

        rows.push(SteadyRow {
            elapsed: el,
            gap_speed: spd * cost / FLAT_COST,
            heart_rate: heart,
            temperature: temp_f.and_then(|f| f.get(i)),
            cadence: cadence_f.and_then(|f| f.get(i)),
            gct: gct_f.and_then(|f| f.get(i)),
            stride: stride_f.and_then(|f| f.get(i)),
        });
    }

    if rows.len() < MIN_STEADY_SECONDS {
        None
    } else {
        Some(rows)
    }
}

/// Minetti et al. (2002) metabolic cost of running as a function of grade.
///
/// Returns cost in J/kg/m. Grade is fractional (0.05 = 5% uphill).
/// The polynomial is valid for grades roughly -0.45 to +0.45.
fn minetti_cost(grade: f64) -> f64 {
    let g = grade.clamp(-0.45, 0.45);
    155.4 * g.powi(5) - 30.4 * g.powi(4) - 43.3 * g.powi(3) + 46.3 * g.powi(2) + 19.5 * g + 3.6
}

/// Cost of running on flat ground (grade = 0).
const FLAT_COST: f64 = 3.6; // minetti_cost(0.0)

/// Per-run cardiac efficiency result.
#[allow(dead_code)]
pub struct RunCE {
    pub activity_id: i64,
    pub date: chrono::NaiveDate,
    pub distance_km: f64,
    pub duration_min: f64,
    pub ce: f64,        // temperature-adjusted cardiac efficiency
    pub ce_raw: f64,    // raw CE before temp adjustment
    pub gap_speed: f64, // average grade-adjusted speed (m/s)
    pub avg_hr: f64,    // average HR during steady state
    pub avg_temp: Option<f64>,
    pub avg_cadence: Option<f64>,
    pub avg_gct: Option<f64>,
    pub avg_stride: Option<f64>,
    pub seconds_used: usize, // how many seconds of data contributed
}

/// Compute cardiac efficiency for a single activity from its detail parquet.
pub fn compute_run_ce(
    activity_id: i64,
    date: chrono::NaiveDate,
    distance_km: f64,
    duration_min: f64,
    detail_path: &std::path::Path,
) -> Option<RunCE> {
    let rows = load_steady_rows(detail_path)?;

    let n = rows.len();
    let avg_gap = rows.iter().map(|r| r.gap_speed).sum::<f64>() / n as f64;
    let avg_hr = rows.iter().map(|r| r.heart_rate).sum::<f64>() / n as f64;

    let opt_mean = |f: fn(&SteadyRow) -> Option<f64>| -> Option<f64> {
        let vals: Vec<f64> = rows.iter().filter_map(f).collect();
        if vals.is_empty() {
            None
        } else {
            Some(vals.iter().sum::<f64>() / vals.len() as f64)
        }
    };

    let avg_temp = opt_mean(|r| r.temperature);
    let avg_cadence = opt_mean(|r| r.cadence);
    let avg_gct = opt_mean(|r| r.gct);
    let avg_stride = opt_mean(|r| r.stride);

    let ce_raw = avg_gap / avg_hr;
    let ce = match avg_temp {
        Some(t) => ce_raw - TEMP_COEFF * (t - TEMP_REF),
        None => ce_raw,
    };

    Some(RunCE {
        activity_id,
        date,
        distance_km,
        duration_min,
        ce,
        ce_raw,
        gap_speed: avg_gap,
        avg_hr,
        avg_temp,
        avg_cadence,
        avg_gct,
        avg_stride,
        seconds_used: n,
    })
}

/// Format speed as min:sec per km pace string.
pub fn format_pace(speed_mps: f64) -> String {
    if speed_mps <= 0.0 {
        return "---".to_string();
    }
    let total_secs = (1000.0 / speed_mps).round() as u32;
    format!("{}:{:02}", total_secs / 60, total_secs % 60)
}

/// Entry point.
pub fn run(config: &Config) -> Result<()> {
    // Load running activities
    let activities = data::load_activities(config)?
        .filter(data::type_in_list(RUNNING_TYPES))
        .select([
            col("activity_id"),
            col("start_time_local"),
            col("distance_m"),
            col("duration_sec"),
        ])
        .sort(["start_time_local"], Default::default())
        .collect()?;

    let details_dir = config.garmin_storage_path.join("activity_details");
    if !details_dir.exists() {
        println!(
            "No activity details on disk. Run `model-health fetch --from <date> --only activity-details` first."
        );
        return Ok(());
    }

    // Compute CE for each run
    let mut runs: Vec<RunCE> = Vec::new();

    let ids = activities.column("activity_id")?.i64()?;
    let times = activities.column("start_time_local")?.datetime()?;
    let dists = activities.column("distance_m")?.f64()?;
    let durs = activities.column("duration_sec")?.f64()?;

    for i in 0..activities.height() {
        let Some(aid) = ids.get(i) else { continue };
        let Some(ts) = times.get(i) else { continue };
        let Some(date) =
            chrono::DateTime::from_timestamp_micros(ts).map(|dt| dt.naive_utc().date())
        else {
            continue;
        };
        let dist_km = dists.get(i).unwrap_or(0.0) / 1000.0;
        let dur_min = durs.get(i).unwrap_or(0.0) / 60.0;

        let detail_path = details_dir.join(format!("{}.parquet", aid));
        if !detail_path.exists() {
            continue;
        }

        if let Some(ce) = compute_run_ce(aid, date, dist_km, dur_min, &detail_path) {
            runs.push(ce);
        }
    }

    if runs.is_empty() {
        println!("No runs with enough data to compute cardiac efficiency.");
        return Ok(());
    }

    runs.sort_by_key(|r| r.date);

    println!("=== Cardiac Efficiency Report ===\n");
    println!(
        "Metric: grade-adjusted speed / heart rate (Minetti formula, temp-normalized to {}°C)",
        TEMP_REF as i32
    );
    println!(
        "Based on {} running activities with detail data.\n",
        runs.len()
    );

    // --- Current state ---
    let all_ce: Vec<f64> = runs.iter().map(|r| r.ce).collect();
    let recent_n = RECENT_RUNS.min(runs.len());
    let recent = &runs[runs.len() - recent_n..];
    let current_ce = recent.iter().map(|r| r.ce).sum::<f64>() / recent.len() as f64;

    // Percentile
    let mut sorted_ce = all_ce.clone();
    sorted_ce.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let below = sorted_ce.iter().filter(|&&v| v < current_ce).count();
    let percentile = (below as f64 / sorted_ce.len() as f64 * 100.0).round() as u32;

    // Use last run date for CE windows (avoids "no data" if user is resting),
    // but use actual today for health/recovery data.
    let today = runs.last().unwrap().date;
    let actual_today = chrono::Utc::now().date_naive();
    let d28_ago = today - chrono::Duration::days(28);
    let last_28d: Vec<f64> = runs
        .iter()
        .filter(|r| r.date > d28_ago)
        .map(|r| r.ce)
        .collect();
    let ce_28d = if last_28d.is_empty() {
        current_ce
    } else {
        last_28d.iter().sum::<f64>() / last_28d.len() as f64
    };

    // 8-week trend: compare last 28d avg to 28d avg from 56 days ago
    let d56_ago = today - chrono::Duration::days(56);
    let prev_28d: Vec<f64> = runs
        .iter()
        .filter(|r| r.date > d56_ago && r.date <= d28_ago)
        .map(|r| r.ce)
        .collect();
    let trend = if prev_28d.is_empty() {
        None
    } else {
        let prev_avg = prev_28d.iter().sum::<f64>() / prev_28d.len() as f64;
        Some(ce_28d - prev_avg)
    };

    // All-time peak (rolling 10-run avg)
    let mut best_10avg = 0.0f64;
    let mut best_10_date = runs[0].date;
    if runs.len() >= 10 {
        for i in 9..runs.len() {
            let avg: f64 = runs[i - 9..=i].iter().map(|r| r.ce).sum::<f64>() / 10.0;
            if avg > best_10avg {
                best_10avg = avg;
                best_10_date = runs[i].date;
            }
        }
    }

    println!(
        "Current CE (28-day avg):  {:.4}  ({}th percentile all-time)",
        ce_28d, percentile
    );
    if let Some(t) = trend {
        let arrow = if t > 0.0001 {
            "improving"
        } else if t < -0.0001 {
            "declining"
        } else {
            "stable"
        };
        println!("8-week trend:             {:+.4}  ({})", t, arrow);
    }
    println!(
        "All-time peak (10-run):   {:.4}  (around {})",
        best_10avg, best_10_date
    );

    // --- Recent runs ---
    let mean_ce = all_ce.iter().sum::<f64>() / all_ce.len() as f64;

    println!("\n--- Last {} Runs ---\n", recent.len());
    println!(
        "   {:<12}{:>6} {:>7} {:>7} {:>5} {:>5} {:>6}",
        "Date", "Dist", "CE", "GAP", "HR", "Temp", "Spm"
    );
    for r in recent {
        let temp_s = r
            .avg_temp
            .map(|t| format!("{:.0}°C", t))
            .unwrap_or_else(|| "---".into());
        let cad_s = r
            .avg_cadence
            .map(|c| format!("{:.0}", c))
            .unwrap_or_else(|| "---".into());
        let indicator = if r.ce >= mean_ce {
            "\x1b[32m▲\x1b[0m" // green — above average
        } else if r.ce >= mean_ce - 0.0015 {
            "\x1b[33m-\x1b[0m" // yellow — slightly below
        } else {
            "\x1b[31m▼\x1b[0m" // red — well below average
        };
        println!(
            " {} {:<12}{:>5.1}k {:>7.4} {:>7} {:>5.0} {:>5} {:>6}",
            indicator,
            r.date,
            r.distance_km,
            r.ce,
            format_pace(r.gap_speed),
            r.avg_hr,
            temp_s,
            cad_s,
        );
    }

    // --- Monthly trend (last 12 months) ---
    println!("\n--- Monthly Trend (last 12 months) ---\n");
    let twelve_months_ago = today - chrono::Duration::days(365);
    let mut monthly: HashMap<(i32, u32), Vec<f64>> = HashMap::new();
    for r in &runs {
        if r.date > twelve_months_ago {
            let key = (r.date.year(), r.date.month());
            monthly.entry(key).or_default().push(r.ce);
        }
    }

    let mut months: Vec<(i32, u32)> = monthly.keys().cloned().collect();
    months.sort();

    // Find min/max for bar scaling
    let monthly_avgs: Vec<f64> = months
        .iter()
        .map(|k| {
            let v = &monthly[k];
            v.iter().sum::<f64>() / v.len() as f64
        })
        .collect();
    let min_ce = monthly_avgs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_ce = monthly_avgs
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    for (i, (year, month)) in months.iter().enumerate() {
        let avg = monthly_avgs[i];
        let n = monthly[&(*year, *month)].len();
        let bar_width = if (max_ce - min_ce).abs() < 1e-9 {
            10
        } else {
            ((avg - min_ce) / (max_ce - min_ce) * 20.0).round() as usize
        };
        let bar: String = "█".repeat(bar_width) + &"░".repeat(20 - bar_width);
        println!("  {}-{:02}  {:.4}  {}  ({} runs)", year, month, avg, bar, n);
    }

    // --- Training drivers ---
    println!("\n--- What's Driving Your Fitness ---\n");

    // Load daily health for recovery context
    let health_ok = data::load_daily_health(config)
        .and_then(validation::clean_daily_health)
        .and_then(|lf| lf.collect().map_err(Into::into));

    // Compute trailing volumes from activity data
    let d90_ago = today - chrono::Duration::days(90);
    let d28 = today - chrono::Duration::days(28);
    let runs_28d: Vec<&RunCE> = runs.iter().filter(|r| r.date > d28).collect();
    let runs_90d: Vec<&RunCE> = runs.iter().filter(|r| r.date > d90_ago).collect();

    let vol_28d: f64 = runs_28d.iter().map(|r| r.distance_km).sum();
    let vol_90d: f64 = runs_90d.iter().map(|r| r.distance_km).sum();
    let long_run_28d = runs_28d
        .iter()
        .map(|r| r.distance_km)
        .fold(0.0f64, f64::max);
    let runs_per_week = runs_28d.len() as f64 / 4.0;

    // Historical percentile for volume
    let mut month_buf: HashMap<(i32, u32), f64> = HashMap::new();
    for r in &runs {
        *month_buf
            .entry((r.date.year(), r.date.month()))
            .or_default() += r.distance_km;
    }
    let mut monthly_vols: Vec<f64> = month_buf.values().cloned().collect();
    monthly_vols.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let vol_monthly_equiv = vol_28d * 30.0 / 28.0;
    let vol_pct = monthly_vols
        .iter()
        .filter(|&&v| v < vol_monthly_equiv)
        .count() as f64
        / monthly_vols.len().max(1) as f64
        * 100.0;

    println!("Training (last 28 days):");
    println!(
        "  Volume:       {:.0} km  ({:.0}th percentile monthly)",
        vol_28d, vol_pct
    );
    println!("  Runs/week:    {:.1}", runs_per_week);
    println!("  Long run:     {:.1} km", long_run_28d);
    println!("  90-day volume: {:.0} km", vol_90d);

    if let Ok(health) = health_ok {
        // Get last 7 days of health data
        let d7_ago = actual_today - chrono::Duration::days(7);
        let recent_health = health
            .lazy()
            .filter(col("date").gt_eq(lit(d7_ago)))
            .collect();

        if let Ok(rh) = recent_health {
            println!("\nRecovery (7-day avg):");
            let mean_of = |col_name: &str| -> Option<f64> {
                rh.column(col_name)
                    .ok()?
                    .as_materialized_series()
                    .cast(&DataType::Float64)
                    .ok()?
                    .mean()
            };
            if let Some(m) = mean_of("resting_hr") {
                println!("  Resting HR:   {:.0} bpm", m);
            }
            if let Some(m) = mean_of("hrv_last_night") {
                println!("  HRV:          {:.0} ms", m);
            }
            if let Some(m) = mean_of("sleep_seconds") {
                println!("  Sleep:        {:.1} hrs", m / 3600.0);
            }
            if let Some(m) = mean_of("avg_stress") {
                println!("  Stress:       {:.0}", m);
            }
        }
    }

    // --- Correlations computed from this session's data ---
    println!(
        "\n--- Key Relationships (computed from {} runs) ---\n",
        runs.len()
    );

    let mut correlations: Vec<(&str, f64, &str)> = Vec::new();

    // Running dynamics correlations
    let stride_vals: Vec<(f64, f64)> = runs
        .iter()
        .filter_map(|r| Some((r.ce, r.avg_stride?)))
        .collect();
    if stride_vals.len() > 20 {
        let r = pearson(&stride_vals);
        correlations.push(("Stride length", r, "longer stride = more efficient"));
    }

    let gct_vals: Vec<(f64, f64)> = runs
        .iter()
        .filter_map(|r| Some((r.ce, r.avg_gct?)))
        .collect();
    if gct_vals.len() > 20 {
        let r = pearson(&gct_vals);
        correlations.push(("Ground contact", r, "less time on ground = better"));
    }

    let cad_vals: Vec<(f64, f64)> = runs
        .iter()
        .filter_map(|r| Some((r.ce, r.avg_cadence?)))
        .collect();
    if cad_vals.len() > 20 {
        let r = pearson(&cad_vals);
        correlations.push(("Cadence", r, "higher turnover = better"));
    }

    let temp_vals: Vec<(f64, f64)> = runs
        .iter()
        .filter_map(|r| Some((r.ce, r.avg_temp?)))
        .collect();
    if temp_vals.len() > 20 {
        let r = pearson(&temp_vals);
        correlations.push(("Temperature", r, "heat reduces efficiency"));
    }

    // Distance correlation
    let dist_vals: Vec<(f64, f64)> = runs.iter().map(|r| (r.ce, r.distance_km)).collect();
    if dist_vals.len() > 20 {
        let r = pearson(&dist_vals);
        correlations.push(("Run distance", r, "longer runs"));
    }

    // Sort by absolute correlation strength
    correlations.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

    for (name, r, desc) in &correlations {
        println!("  {:<20} r={:+.2}   ({})", name, r, desc);
    }

    println!();
    println!("To improve: build consistent weekly volume with a long run,");
    println!("prioritize recovery (sleep, low stress), and the running");
    println!("economy factors (cadence, ground contact) follow naturally.");

    Ok(())
}

// ---------------------------------------------------------------------------
// Cardiac Drift
// ---------------------------------------------------------------------------

/// Per-run cardiac drift result.
pub struct RunDrift {
    pub date: chrono::NaiveDate,
    pub distance_km: f64,
    pub decoupling_pct: f64, // (CE_h1 - CE_h2) / CE_h1 * 100; positive = normal drift
    pub hr_drift_pct: f64,   // (HR_h2 - HR_h1) / HR_h1 * 100
    pub hr_h1: f64,
    pub hr_h2: f64,
    pub temp: Option<f64>,
}

/// Why a run was excluded from drift analysis.
pub enum DriftSkip {
    TooShort,
    IntervalWorkout,
    UnevenHalves,
}

/// Compute cardiac drift for a single activity.
/// Splits the run into first and second halves (by elapsed time, post-warmup)
/// and compares grade-adjusted efficiency between halves.
/// Returns Ok(RunDrift) on success, or Err(DriftSkip) explaining why it was skipped.
pub fn compute_run_drift(
    date: chrono::NaiveDate,
    distance_km: f64,
    detail_path: &std::path::Path,
) -> std::result::Result<RunDrift, DriftSkip> {
    let rows = load_steady_rows(detail_path).ok_or(DriftSkip::TooShort)?;

    if rows.len() < 60 {
        return Err(DriftSkip::TooShort);
    }

    // Skip interval workouts via HR coefficient of variation.
    // Hills affect speed but HR stays steady on even-effort runs.
    let hr_mean = rows.iter().map(|r| r.heart_rate).sum::<f64>() / rows.len() as f64;
    let hr_var = rows
        .iter()
        .map(|r| (r.heart_rate - hr_mean).powi(2))
        .sum::<f64>()
        / rows.len() as f64;
    let hr_cv = hr_var.sqrt() / hr_mean * 100.0;
    if hr_cv > MAX_HR_CV_PCT {
        return Err(DriftSkip::IntervalWorkout);
    }

    // Split into halves by elapsed time
    let mid_time = (rows.first().unwrap().elapsed + rows.last().unwrap().elapsed) / 2.0;
    let h1: Vec<&SteadyRow> = rows.iter().filter(|r| r.elapsed < mid_time).collect();
    let h2: Vec<&SteadyRow> = rows.iter().filter(|r| r.elapsed >= mid_time).collect();

    if h1.len() < 20 || h2.len() < 20 {
        return Err(DriftSkip::UnevenHalves);
    }

    let gap_h1 = h1.iter().map(|r| r.gap_speed).sum::<f64>() / h1.len() as f64;
    let gap_h2 = h2.iter().map(|r| r.gap_speed).sum::<f64>() / h2.len() as f64;
    let hr_h1 = h1.iter().map(|r| r.heart_rate).sum::<f64>() / h1.len() as f64;
    let hr_h2 = h2.iter().map(|r| r.heart_rate).sum::<f64>() / h2.len() as f64;

    let ce_h1 = gap_h1 / hr_h1;
    let ce_h2 = gap_h2 / hr_h2;

    let decoupling = (ce_h1 - ce_h2) / ce_h1 * 100.0;
    let hr_drift = (hr_h2 - hr_h1) / hr_h1 * 100.0;

    let temps: Vec<f64> = rows.iter().filter_map(|r| r.temperature).collect();
    let avg_temp = if temps.is_empty() {
        None
    } else {
        Some(temps.iter().sum::<f64>() / temps.len() as f64)
    };

    Ok(RunDrift {
        date,
        distance_km,
        decoupling_pct: decoupling,
        hr_drift_pct: hr_drift,
        hr_h1,
        hr_h2,
        temp: avg_temp,
    })
}

/// Cardiac drift report entry point.
pub fn drift(config: &Config) -> Result<()> {
    let activities = data::load_activities(config)?
        .filter(data::type_in_list(RUNNING_TYPES))
        .filter(col("distance_m").gt(lit(5000))) // skip very short runs
        .select([
            col("activity_id"),
            col("start_time_local"),
            col("distance_m"),
        ])
        .sort(["start_time_local"], Default::default())
        .collect()?;

    let details_dir = config.garmin_storage_path.join("activity_details");
    if !details_dir.exists() {
        println!(
            "No activity details on disk. Run `model-health fetch --from <date> --only activity-details` first."
        );
        return Ok(());
    }

    let mut runs: Vec<RunDrift> = Vec::new();
    let mut skipped_intervals = 0u32;

    let ids = activities.column("activity_id")?.i64()?;
    let times = activities.column("start_time_local")?.datetime()?;
    let dists = activities.column("distance_m")?.f64()?;

    for i in 0..activities.height() {
        let Some(aid) = ids.get(i) else { continue };
        let Some(ts) = times.get(i) else { continue };
        let Some(date) =
            chrono::DateTime::from_timestamp_micros(ts).map(|dt| dt.naive_utc().date())
        else {
            continue;
        };
        let dist_km = dists.get(i).unwrap_or(0.0) / 1000.0;

        let detail_path = details_dir.join(format!("{}.parquet", aid));
        if !detail_path.exists() {
            continue;
        }

        match compute_run_drift(date, dist_km, &detail_path) {
            Ok(d) => runs.push(d),
            Err(DriftSkip::IntervalWorkout) => skipped_intervals += 1,
            Err(_) => {}
        }
    }

    if runs.is_empty() {
        println!("No runs with enough data for drift analysis.");
        return Ok(());
    }

    runs.sort_by_key(|r| r.date);

    println!("=== Cardiac Drift Report ===\n");
    println!("Decoupling = drop in pace:HR efficiency from 1st to 2nd half.");
    println!("Benchmark: <5% = strong aerobic base, 5-10% = normal, >10% = needs work");
    let skipped_msg = if skipped_intervals > 0 {
        format!(" ({} interval workouts excluded)", skipped_intervals)
    } else {
        String::new()
    };
    println!(
        "\nAnalyzed {} steady runs (5+ km).{}\n",
        runs.len(),
        skipped_msg
    );

    // Overall stats
    let all_dec: Vec<f64> = runs.iter().map(|r| r.decoupling_pct).collect();
    let mean_dec = all_dec.iter().sum::<f64>() / all_dec.len() as f64;
    let mut sorted = all_dec.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_dec = sorted[sorted.len() / 2];

    println!("Overall: mean={:.1}%, median={:.1}%", mean_dec, median_dec);

    // By distance
    println!("\n--- By Distance ---\n");
    for (lo, hi, label) in [
        (5.0, 8.0, " 5-8 km"),
        (8.0, 12.0, " 8-12 km"),
        (12.0, 18.0, "12-18 km"),
        (18.0, 30.0, "18-30 km"),
        (30.0, 100.0, "   30+ km"),
    ] {
        let bucket: Vec<&RunDrift> = runs
            .iter()
            .filter(|r| r.distance_km >= lo && r.distance_km < hi)
            .collect();
        if bucket.len() >= 5 {
            let avg = bucket.iter().map(|r| r.decoupling_pct).sum::<f64>() / bucket.len() as f64;
            let hr_d = bucket.iter().map(|r| r.hr_drift_pct).sum::<f64>() / bucket.len() as f64;
            println!(
                "  {}: decoupling={:+.1}%  HR drift={:+.1}%  (n={})",
                label,
                avg,
                hr_d,
                bucket.len()
            );
        }
    }

    // By temperature
    let with_temp: Vec<&RunDrift> = runs.iter().filter(|r| r.temp.is_some()).collect();
    if with_temp.len() > 20 {
        println!("\n--- By Temperature ---\n");
        for (lo, hi, label) in [
            (0.0, 18.0, "Cool (<18°C)"),
            (18.0, 24.0, "Mild (18-24°C)"),
            (24.0, 30.0, "Warm (24-30°C)"),
            (30.0, 50.0, "Hot (30+°C)"),
        ] {
            let bucket: Vec<&&RunDrift> = with_temp
                .iter()
                .filter(|r| {
                    let t = r.temp.unwrap();
                    t >= lo && t < hi
                })
                .collect();
            if bucket.len() >= 5 {
                let avg =
                    bucket.iter().map(|r| r.decoupling_pct).sum::<f64>() / bucket.len() as f64;
                println!(
                    "  {:>18}: decoupling={:+.1}%  (n={})",
                    label,
                    avg,
                    bucket.len()
                );
            }
        }
    }

    // Yearly trend
    println!("\n--- Yearly Trend ---\n");
    let mut by_year: HashMap<i32, Vec<f64>> = HashMap::new();
    for r in &runs {
        by_year
            .entry(r.date.year())
            .or_default()
            .push(r.decoupling_pct);
    }
    let mut years: Vec<i32> = by_year.keys().cloned().collect();
    years.sort();
    for y in &years {
        let vals = &by_year[y];
        let avg = vals.iter().sum::<f64>() / vals.len() as f64;
        let bar_len = (avg.clamp(0.0, 15.0) / 15.0 * 20.0).round() as usize;
        let bar: String = "█".repeat(bar_len) + &"░".repeat(20 - bar_len);
        println!("  {}: {:+5.1}%  {}  (n={})", y, avg, bar, vals.len());
    }

    // Last 10 runs
    let recent_n = 10.min(runs.len());
    let recent = &runs[runs.len() - recent_n..];
    println!("\n--- Last {} Runs ---\n", recent_n);
    println!(
        "   {:>12} {:>6} {:>9} {:>7} {:>7} {:>6}",
        "Date", "Dist", "Decouple", "HR 1st", "HR 2nd", "Drift"
    );
    for r in recent {
        let indicator = if r.decoupling_pct < 5.0 {
            "\x1b[32m▲\x1b[0m" // green up — good
        } else if r.decoupling_pct <= 10.0 {
            "\x1b[33m-\x1b[0m" // yellow dash — maintaining
        } else {
            "\x1b[31m▼\x1b[0m" // red down — needs work
        };
        println!(
            " {} {:>12} {:>5.1}k {:>+8.1}% {:>7.0} {:>7.0} {:>+5.1}%",
            indicator, r.date, r.distance_km, r.decoupling_pct, r.hr_h1, r.hr_h2, r.hr_drift_pct,
        );
    }

    // Interpretation
    let recent_avg = recent.iter().map(|r| r.decoupling_pct).sum::<f64>() / recent.len() as f64;
    println!();
    if recent_avg < 5.0 {
        println!(
            "Your recent decoupling ({:.1}%) is under 5% — your aerobic base is solid.",
            recent_avg
        );
    } else if recent_avg < 10.0 {
        println!(
            "Your recent decoupling ({:.1}%) is in the normal range.",
            recent_avg
        );
        println!("Consistent easy running and long runs will bring this down.");
    } else {
        println!(
            "Your recent decoupling ({:.1}%) is elevated — your aerobic base needs work.",
            recent_avg
        );
        println!("Focus on easy-effort runs and gradually build duration.");
    }

    Ok(())
}

/// Compute Pearson correlation coefficient from paired (x, y) values.
fn pearson(pairs: &[(f64, f64)]) -> f64 {
    let n = pairs.len() as f64;
    if n < 3.0 {
        return 0.0;
    }
    let sum_x: f64 = pairs.iter().map(|(x, _)| x).sum();
    let sum_y: f64 = pairs.iter().map(|(_, y)| y).sum();
    let sum_xy: f64 = pairs.iter().map(|(x, y)| x * y).sum();
    let sum_x2: f64 = pairs.iter().map(|(x, _)| x * x).sum();
    let sum_y2: f64 = pairs.iter().map(|(_, y)| y * y).sum();

    let num = n * sum_xy - sum_x * sum_y;
    let den = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
    if den.abs() < 1e-15 { 0.0 } else { num / den }
}
