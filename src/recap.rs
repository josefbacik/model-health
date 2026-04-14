//! Post-run recap: comprehensive analysis of the most recent activity.
//!
//! Pulls together cardiac efficiency, drift, splits, recovery context,
//! comparison to similar recent runs, and fitness trend into a single
//! report. Run with `model-health recap` after your last run.

use polars::prelude::*;

use crate::config::Config;
use crate::data;
use crate::data::RUNNING_TYPES;
use crate::error::Result;
use crate::fitness;
use crate::validation;

/// How many days back to look for "similar" comparison runs.
const COMPARISON_WINDOW_DAYS: i64 = 90;

/// Distance tolerance for "similar" runs (fraction).
const DISTANCE_TOLERANCE: f64 = 0.20;

/// Entry point.
pub fn run(config: &Config) -> Result<()> {
    // --- Load running activities, sorted by time ---
    let activities = data::load_activities(config)?
        .filter(data::type_in_list(RUNNING_TYPES))
        .select([
            col("activity_id"),
            col("activity_name"),
            col("activity_type"),
            col("start_time_local"),
            col("distance_m"),
            col("duration_sec"),
            col("avg_hr"),
            col("max_hr"),
            col("calories"),
            col("elevation_gain"),
            col("training_effect"),
            col("training_load"),
            col("avg_cadence"),
        ])
        .sort(["start_time_local"], Default::default())
        .collect()?;

    if activities.height() == 0 {
        println!("No running activities found.");
        return Ok(());
    }

    let details_dir = config.garmin_storage_path.join("activity_details");
    let splits_dir = config.garmin_storage_path.join("activity_splits");

    // --- Find the most recent run ---
    let last_idx = activities.height() - 1;
    let aid = activities
        .column("activity_id")?
        .i64()?
        .get(last_idx)
        .unwrap();
    let ts = activities
        .column("start_time_local")?
        .datetime()?
        .get(last_idx)
        .unwrap();
    let date = chrono::DateTime::from_timestamp_micros(ts)
        .map(|dt| dt.naive_utc().date())
        .unwrap();
    let dist_m = activities
        .column("distance_m")?
        .f64()?
        .get(last_idx)
        .unwrap_or(0.0);
    let dur_sec = activities
        .column("duration_sec")?
        .f64()?
        .get(last_idx)
        .unwrap_or(0.0);
    let dist_km = dist_m / 1000.0;
    let dur_min = dur_sec / 60.0;

    let name = activities
        .column("activity_name")?
        .str()?
        .get(last_idx)
        .unwrap_or("Run");
    let summary_hr = activities
        .column("avg_hr")?
        .cast(&DataType::Float64)?
        .f64()?
        .get(last_idx);
    let max_hr = activities
        .column("max_hr")?
        .cast(&DataType::Float64)?
        .f64()?
        .get(last_idx);
    let calories = activities
        .column("calories")?
        .cast(&DataType::Float64)?
        .f64()?
        .get(last_idx);
    let elev_gain = activities.column("elevation_gain")?.f64()?.get(last_idx);
    let training_effect = activities.column("training_effect")?.f64()?.get(last_idx);
    let training_load = activities.column("training_load")?.f64()?.get(last_idx);

    // Overall pace
    let avg_pace_mps = if dur_sec > 0.0 { dist_m / dur_sec } else { 0.0 };

    // --- Header ---
    println!("=== Run Recap: {} ===\n", name);

    let dur_m = (dur_sec / 60.0).floor() as u32;
    let dur_s = (dur_sec % 60.0).round() as u32;
    println!(
        "Date: {}    Distance: {:.1} km    Duration: {}:{:02}    Pace: {}/km",
        date,
        dist_km,
        dur_m,
        dur_s,
        fitness::format_pace(avg_pace_mps),
    );

    // Second line of summary stats
    let mut extras = Vec::new();
    if let Some(hr) = summary_hr {
        extras.push(format!("Avg HR: {:.0} bpm", hr));
    }
    if let Some(hr) = max_hr {
        extras.push(format!("Max HR: {:.0}", hr));
    }
    if let Some(g) = elev_gain
        && g > 0.0
    {
        extras.push(format!("Elev: +{:.0}m", g));
    }
    if let Some(c) = calories {
        extras.push(format!("Calories: {:.0}", c));
    }
    if let Some(te) = training_effect {
        extras.push(format!("TE: {:.1}", te));
    }
    if let Some(tl) = training_load {
        extras.push(format!("Load: {:.0}", tl));
    }
    if !extras.is_empty() {
        println!("{}", extras.join("    "));
    }

    // --- Cardiac Efficiency ---
    let detail_path = details_dir.join(format!("{}.parquet", aid));
    let this_ce = if detail_path.exists() {
        fitness::compute_run_ce(aid, date, dist_km, dur_min, &detail_path)
    } else {
        None
    };

    if let Some(ref ce) = this_ce {
        println!("\n--- Performance ---\n");
        println!(
            "  Cardiac Efficiency:  {:.4}  (GAP: {}/km @ {:.0} bpm{})",
            ce.ce,
            fitness::format_pace(ce.gap_speed),
            ce.avg_hr,
            ce.avg_temp
                .map(|t| format!(", {:.0}°C", t))
                .unwrap_or_default(),
        );

        // Running dynamics
        // Garmin stores cadence as single-leg (steps/min for one foot),
        // stride_length in cm, and ground_contact_time in ms.
        let mut dynamics = Vec::new();
        if let Some(c) = ce.avg_cadence {
            dynamics.push(format!("Cadence: {:.0} spm", c * 2.0));
        }
        if let Some(s) = ce.avg_stride {
            dynamics.push(format!("Stride: {:.0} cm", s));
        }
        if let Some(g) = ce.avg_gct {
            dynamics.push(format!("GCT: {:.0} ms", g));
        }
        if !dynamics.is_empty() {
            println!("  Dynamics:            {}", dynamics.join("  |  "));
        }
    }

    // --- Compute all historical CEs for comparison ---
    let ids = activities.column("activity_id")?.i64()?;
    let times = activities.column("start_time_local")?.datetime()?;
    let dists = activities.column("distance_m")?.f64()?;
    let durs = activities.column("duration_sec")?.f64()?;

    let mut all_runs: Vec<fitness::RunCE> = Vec::new();
    if details_dir.exists() {
        for i in 0..activities.height() {
            let Some(rid) = ids.get(i) else { continue };
            let Some(rts) = times.get(i) else { continue };
            let Some(rdate) =
                chrono::DateTime::from_timestamp_micros(rts).map(|dt| dt.naive_utc().date())
            else {
                continue;
            };
            let rdist_km = dists.get(i).unwrap_or(0.0) / 1000.0;
            let rdur_min = durs.get(i).unwrap_or(0.0) / 60.0;

            let dp = details_dir.join(format!("{}.parquet", rid));
            if !dp.exists() {
                continue;
            }
            if let Some(rce) = fitness::compute_run_ce(rid, rdate, rdist_km, rdur_min, &dp) {
                all_runs.push(rce);
            }
        }
    }

    all_runs.sort_by_key(|r| r.date);

    // Show CE context if we have data
    if let Some(ref ce) = this_ce
        && all_runs.len() > 1
    {
        // 28-day average
        let d28_ago = date - chrono::Duration::days(28);
        let recent_ces: Vec<f64> = all_runs
            .iter()
            .filter(|r| r.date > d28_ago && r.activity_id != aid)
            .map(|r| r.ce)
            .collect();

        if !recent_ces.is_empty() {
            let avg_28d = recent_ces.iter().sum::<f64>() / recent_ces.len() as f64;
            let pct_diff = (ce.ce - avg_28d) / avg_28d * 100.0;
            let indicator = if pct_diff > 0.5 {
                "\x1b[32m^\x1b[0m"
            } else if pct_diff < -0.5 {
                "\x1b[31mv\x1b[0m"
            } else {
                " "
            };
            println!(
                "  vs 28-day avg:       {:.4}  ({}{:+.1}%)",
                avg_28d, indicator, pct_diff
            );
        }

        // All-time percentile
        let mut sorted_ce: Vec<f64> = all_runs.iter().map(|r| r.ce).collect();
        sorted_ce.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let below = sorted_ce.iter().filter(|&&v| v < ce.ce).count();
        let percentile = (below as f64 / sorted_ce.len() as f64 * 100.0).round() as u32;
        println!("  All-time percentile: {}th", percentile);
    }

    // --- Cardiac Drift ---
    if detail_path.exists() && dist_km >= 5.0 {
        match fitness::compute_run_drift(date, dist_km, &detail_path) {
            Ok(drift) => {
                let assessment = if drift.decoupling_pct < 5.0 {
                    "\x1b[32mstrong aerobic base\x1b[0m"
                } else if drift.decoupling_pct <= 10.0 {
                    "\x1b[33mnormal\x1b[0m"
                } else {
                    "\x1b[31mneeds work\x1b[0m"
                };
                println!(
                    "\n  Cardiac Drift:       {:+.1}%  ({})",
                    drift.decoupling_pct, assessment
                );
                println!(
                    "  HR: {:.0} -> {:.0} bpm (1st -> 2nd half, drift {:+.1}%)",
                    drift.hr_h1, drift.hr_h2, drift.hr_drift_pct,
                );
            }
            Err(fitness::DriftSkip::IntervalWorkout) => {
                println!("\n  Cardiac Drift:       n/a (interval workout — HR too variable)");
            }
            Err(_) => {}
        }
    }

    // --- Splits ---
    let splits_path = splits_dir.join(format!("{}.parquet", aid));
    if splits_path.exists()
        && let Ok(splits_lf) =
            LazyFrame::scan_parquet(splits_path.to_string_lossy().as_ref(), Default::default())
    {
        // Try INTERVAL splits (Garmin's mile/km markers), then fall back to all
        let try_interval = splits_lf
            .clone()
            .filter(col("split_type").eq(lit("INTERVAL")))
            .sort(["split_number"], Default::default())
            .collect();
        let try_all = || {
            splits_lf
                .sort(["split_number"], Default::default())
                .collect()
        };
        let splits_result = match try_interval {
            Ok(ref df) if df.height() > 0 => try_interval,
            _ => try_all(),
        };
        if let Ok(splits) = splits_result
            && splits.height() > 0
        {
            println!("\n--- Splits ---\n");
            println!(
                "  {:>5}  {:>6}  {:>7}  {:>5}  {:>5}",
                "Split", "Dist", "Pace", "HR", "Cad"
            );

            let split_nums = splits.column("split_number")?.i32()?;
            let split_dists = splits.column("distance_m")?.f64()?;
            let split_durs = splits.column("duration_sec")?.f64()?;
            let split_hrs = splits
                .column("avg_hr")
                .ok()
                .and_then(|c| c.cast(&DataType::Float64).ok());
            let split_cads = splits
                .column("avg_cadence")
                .ok()
                .and_then(|c| c.cast(&DataType::Float64).ok());

            for i in 0..splits.height() {
                let num = split_nums.get(i).unwrap_or(0);
                let d = split_dists.get(i).unwrap_or(0.0);
                let dur = split_durs.get(i).unwrap_or(0.0);

                // Skip tiny remnant splits (< 100m)
                if d < 100.0 {
                    continue;
                }

                let pace_mps = if dur > 0.0 { d / dur } else { 0.0 };

                let hr_s = split_hrs
                    .as_ref()
                    .and_then(|c| c.f64().ok())
                    .and_then(|ca| ca.get(i))
                    .map(|v| format!("{:.0}", v))
                    .unwrap_or_else(|| "---".into());
                let cad_s = split_cads
                    .as_ref()
                    .and_then(|c| c.f64().ok())
                    .and_then(|ca| ca.get(i))
                    .map(|v| format!("{:.0}", v))
                    .unwrap_or_else(|| "---".into());

                let dist_s = if d >= 1000.0 {
                    format!("{:.1}k", d / 1000.0)
                } else {
                    format!("{:.0}m", d)
                };

                println!(
                    "  {:>5}  {:>6}  {:>7}  {:>5}  {:>5}",
                    num,
                    dist_s,
                    fitness::format_pace(pace_mps),
                    hr_s,
                    cad_s,
                );
            }
        }
    }

    // --- Recovery Context ---
    println!("\n--- Recovery Context ---\n");

    let daily_lf = data::load_daily_health(config)?;
    let daily_lf = validation::clean_daily_health(daily_lf)?;
    let daily = daily_lf.collect()?;

    let d90_ago = date - chrono::Duration::days(90);
    let recent_health = daily
        .clone()
        .lazy()
        .filter(
            col("date")
                .gt_eq(lit(d90_ago))
                .and(col("date").lt(lit(date))),
        )
        .collect()?;

    let run_day_health = daily
        .clone()
        .lazy()
        .filter(col("date").eq(lit(date)))
        .collect()?;

    let baseline_of = |col_name: &str, df: &DataFrame| -> Option<(f64, f64)> {
        let s = df
            .column(col_name)
            .ok()?
            .as_materialized_series()
            .cast(&DataType::Float64)
            .ok()?;
        let mean = s.mean()?;
        let n = s.len() - s.null_count();
        if n < 10 {
            return None;
        }
        let vals: Vec<f64> = s.f64().ok()?.into_no_null_iter().collect();
        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
        Some((mean, var.sqrt()))
    };

    let day_val = |col_name: &str| -> Option<f64> {
        let s = run_day_health
            .column(col_name)
            .ok()?
            .as_materialized_series()
            .cast(&DataType::Float64)
            .ok()?;
        if s.is_empty() {
            return None;
        }
        s.f64().ok()?.get(0)
    };

    struct RecoveryFactor {
        name: &'static str,
        col: &'static str,
        direction: f64, // +1 = higher is better, -1 = lower is better
        format_fn: fn(f64) -> String,
    }

    let fmt_f0 = |v: f64| format!("{:.0}", v);
    let fmt_hrs = |v: f64| format!("{:.1}h", v / 3600.0);

    let factors = [
        RecoveryFactor {
            name: "Resting HR",
            col: "resting_hr",
            direction: -1.0,
            format_fn: fmt_f0,
        },
        RecoveryFactor {
            name: "Sleep",
            col: "sleep_seconds",
            direction: 1.0,
            format_fn: fmt_hrs,
        },
        RecoveryFactor {
            name: "Sleep Score",
            col: "sleep_score",
            direction: 1.0,
            format_fn: fmt_f0,
        },
        RecoveryFactor {
            name: "HRV",
            col: "hrv_last_night",
            direction: 1.0,
            format_fn: fmt_f0,
        },
        RecoveryFactor {
            name: "Stress",
            col: "avg_stress",
            direction: -1.0,
            format_fn: fmt_f0,
        },
        RecoveryFactor {
            name: "Body Battery",
            col: "body_battery_start",
            direction: 1.0,
            format_fn: fmt_f0,
        },
    ];

    let mut z_sum = 0.0;
    let mut z_count = 0;
    let mut printed_header = false;

    for f in &factors {
        let Some(val) = day_val(f.col) else { continue };
        let Some((mean, std)) = baseline_of(f.col, &recent_health) else {
            continue;
        };

        if !printed_header {
            println!(
                "  {:<16} {:>10}  {:>10}  Status",
                "Factor", "Run Day", "Avg (90d)"
            );
            println!("  {:-<16} {:-<10}  {:-<10}  {:-<10}", "", "", "", "");
            printed_header = true;
        }

        let z = if std > 0.001 {
            (val - mean) / std * f.direction
        } else {
            0.0
        };
        z_sum += z;
        z_count += 1;

        let (indicator, status) = if z > 0.5 {
            ("\x1b[32m+\x1b[0m", "above avg")
        } else if z < -0.5 {
            ("\x1b[31m-\x1b[0m", "below avg")
        } else {
            (" ", "average")
        };

        println!(
            "{} {:<16} {:>10}  {:>10}  {}",
            indicator,
            f.name,
            (f.format_fn)(val),
            (f.format_fn)(mean),
            status,
        );
    }

    if z_count > 0 {
        let avg_z = z_sum / z_count as f64;
        let score_10 = ((avg_z + 2.0) / 4.0 * 9.0 + 1.0).clamp(1.0, 10.0);
        let (color, label) = if score_10 >= 7.0 {
            ("\x1b[32m", "good recovery")
        } else if score_10 >= 4.0 {
            ("\x1b[33m", "normal recovery")
        } else {
            ("\x1b[31m", "low recovery")
        };
        println!(
            "\n  Readiness that day: {}{:.0}/10\x1b[0m ({})",
            color, score_10, label
        );
    }

    // --- Comparison to Similar Runs ---
    if let Some(ref ce) = this_ce
        && all_runs.len() > 1
    {
        let comp_cutoff = date - chrono::Duration::days(COMPARISON_WINDOW_DAYS);
        let lo = dist_km * (1.0 - DISTANCE_TOLERANCE);
        let hi = dist_km * (1.0 + DISTANCE_TOLERANCE);

        let similar: Vec<&fitness::RunCE> = all_runs
            .iter()
            .filter(|r| {
                r.date > comp_cutoff
                    && r.activity_id != aid
                    && r.distance_km >= lo
                    && r.distance_km <= hi
            })
            .collect();

        if similar.len() >= 3 {
            println!("\n--- Comparison to Similar Runs ---\n");
            println!(
                "  {} runs of {:.0}-{:.0} km in the last {} days:\n",
                similar.len(),
                lo,
                hi,
                COMPARISON_WINDOW_DAYS,
            );

            let avg_ce = similar.iter().map(|r| r.ce).sum::<f64>() / similar.len() as f64;
            let avg_gap = similar.iter().map(|r| r.gap_speed).sum::<f64>() / similar.len() as f64;
            let avg_hr = similar.iter().map(|r| r.avg_hr).sum::<f64>() / similar.len() as f64;

            let ce_diff_pct = (ce.ce - avg_ce) / avg_ce * 100.0;
            let indicator = if ce_diff_pct > 0.5 {
                "\x1b[32mbetter\x1b[0m"
            } else if ce_diff_pct < -0.5 {
                "\x1b[31mworse\x1b[0m"
            } else {
                "similar"
            };

            println!("  {:>20}  {:>8}  {:>8}  {:>8}", "", "CE", "GAP", "Avg HR");
            println!(
                "  {:>20}  {:>8.4}  {:>8}  {:>8.0}",
                "This run",
                ce.ce,
                fitness::format_pace(ce.gap_speed),
                ce.avg_hr,
            );
            println!(
                "  {:>20}  {:>8.4}  {:>8}  {:>8.0}",
                format!("Avg of {} similar", similar.len()),
                avg_ce,
                fitness::format_pace(avg_gap),
                avg_hr,
            );
            println!(
                "\n  This run was {} than similar recent runs ({:+.1}% CE)",
                indicator, ce_diff_pct,
            );

            // Best and worst of the similar group
            let best = similar
                .iter()
                .max_by(|a, b| a.ce.partial_cmp(&b.ce).unwrap())
                .unwrap();
            let worst = similar
                .iter()
                .min_by(|a, b| a.ce.partial_cmp(&b.ce).unwrap())
                .unwrap();
            println!(
                "  Range: {:.4} ({}) to {:.4} ({})",
                worst.ce, worst.date, best.ce, best.date,
            );
        }
    }

    // --- Overall Assessment ---
    println!("\n--- Assessment ---\n");

    let mut notes: Vec<String> = Vec::new();

    if let Some(ref ce) = this_ce {
        // CE vs recent
        let d28_ago = date - chrono::Duration::days(28);
        let recent_ces: Vec<f64> = all_runs
            .iter()
            .filter(|r| r.date > d28_ago && r.activity_id != aid)
            .map(|r| r.ce)
            .collect();
        if !recent_ces.is_empty() {
            let avg = recent_ces.iter().sum::<f64>() / recent_ces.len() as f64;
            let diff_pct = (ce.ce - avg) / avg * 100.0;
            if diff_pct > 2.0 {
                notes.push(format!(
                    "Efficiency was notably \x1b[32mabove\x1b[0m your recent average ({:+.1}%).",
                    diff_pct,
                ));
            } else if diff_pct < -2.0 {
                notes.push(format!(
                    "Efficiency was \x1b[31mbelow\x1b[0m your recent average ({:+.1}%). Check recovery factors above for likely causes.",
                    diff_pct,
                ));
            } else {
                notes.push("Efficiency was in line with your recent average.".into());
            }
        }
    }

    // Drift assessment
    if detail_path.exists()
        && dist_km >= 5.0
        && let Ok(drift) = fitness::compute_run_drift(date, dist_km, &detail_path)
    {
        if drift.decoupling_pct < 3.0 {
            notes.push("Excellent pacing — minimal cardiac drift.".into());
        } else if drift.decoupling_pct > 10.0 {
            notes.push(format!(
                "High cardiac drift ({:.1}%) — consider starting easier or checking hydration/fueling.",
                drift.decoupling_pct,
            ));
        }
    }

    // TE note
    if let Some(te) = training_effect {
        if te >= 4.0 {
            notes.push(format!(
                "Training effect was {:.1} (highly beneficial). Allow adequate recovery.",
                te,
            ));
        } else if te < 2.0 {
            notes.push(format!(
                "Training effect was {:.1} (recovery/maintaining). Good for an easy day.",
                te,
            ));
        }
    }

    if notes.is_empty() {
        println!("  Solid run — nothing unusual flagged.");
    } else {
        for note in &notes {
            println!("  - {}", note);
        }
    }

    println!();
    Ok(())
}
