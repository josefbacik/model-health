//! Recovery-based readiness assessment.
//!
//! Compares today's recovery metrics (body battery, resting HR, sleep, stress)
//! against your personal baselines and looks up historical CE outcomes for
//! runs under similar conditions. Tells you whether today is a good day to
//! push hard, run easy, or rest.
//!
//! Run with `model-health readiness` to see the assessment.

use polars::prelude::*;
use std::collections::HashMap;

use crate::config::Config;
use crate::data;
use crate::data::RUNNING_TYPES;
use crate::error::Result;
use crate::fitness;
use crate::validation;

/// A recovery factor with its current value, personal baseline, and effect direction.
struct Factor {
    name: &'static str,
    value: f64,
    mean: f64,
    std: f64,
    /// Positive means higher is better (sleep, BB); negative means lower is better (RHR, stress)
    direction: f64,
}

impl Factor {
    fn z_score(&self) -> f64 {
        if self.std < 0.001 {
            return 0.0;
        }
        (self.value - self.mean) / self.std * self.direction
    }

    fn status(&self) -> (&'static str, &'static str) {
        let z = self.z_score();
        if z > 0.5 {
            ("\x1b[32m+\x1b[0m", "above avg")
        } else if z < -0.5 {
            ("\x1b[31m-\x1b[0m", "below avg")
        } else {
            (" ", "average")
        }
    }
}

/// Entry point.
pub fn run(config: &Config) -> Result<()> {
    let today = chrono::Utc::now().date_naive();

    // Load health data for baselines and today's values
    let daily_lf = data::load_daily_health(config)?;
    let daily_lf = validation::clean_daily_health(daily_lf)?;
    let daily = daily_lf.collect()?;

    // Compute personal baselines (mean and std) for each metric
    // Use last 90 days for baselines so they reflect current fitness
    let d90_ago = today - chrono::Duration::days(90);
    let recent_health = daily
        .clone()
        .lazy()
        .filter(col("date").gt_eq(lit(d90_ago)))
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
        // Manual std since Series::std() doesn't exist on Column
        let vals: Vec<f64> = s.f64().ok()?.into_no_null_iter().collect();
        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
        Some((mean, var.sqrt()))
    };

    // Get today's values — fall back to yesterday if today's data isn't complete yet
    let mut assessment_date = today;
    let mut today_row = daily
        .clone()
        .lazy()
        .filter(col("date").eq(lit(today)))
        .collect()?;

    // Check if we got usable data (resting_hr is a good canary)
    let has_data = today_row.height() > 0
        && today_row
            .column("resting_hr")
            .ok()
            .and_then(|c| c.get(0).ok())
            .map(|v| v != AnyValue::Null)
            .unwrap_or(false);

    if !has_data {
        let yesterday = today - chrono::Duration::days(1);
        today_row = daily
            .clone()
            .lazy()
            .filter(col("date").eq(lit(yesterday)))
            .collect()?;
        assessment_date = yesterday;
    }

    let today_val = |col_name: &str| -> Option<f64> {
        let s = today_row
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

    println!("=== Readiness Assessment ===\n");
    if assessment_date != today {
        println!(
            "Date: {} (today's data not yet available)\n",
            assessment_date
        );
    } else {
        println!("Date: {}\n", assessment_date);
    }

    // Build factors
    let mut factors: Vec<Factor> = Vec::new();
    let mut readiness_score = 0.0;
    let mut factor_count = 0;

    struct FactorDef {
        name: &'static str,
        col: &'static str,
        direction: f64,
        unit: &'static str,
        format_fn: fn(f64) -> String,
    }

    let fmt_f0 = |v: f64| format!("{:.0}", v);
    let fmt_hrs = |v: f64| format!("{:.1} hrs", v / 3600.0);

    let defs = [
        FactorDef {
            name: "Resting HR",
            col: "resting_hr",
            direction: -1.0,
            unit: "bpm",
            format_fn: fmt_f0,
        },
        FactorDef {
            name: "Sleep",
            col: "sleep_seconds",
            direction: 1.0,
            unit: "",
            format_fn: fmt_hrs,
        },
        FactorDef {
            name: "Sleep Score",
            col: "sleep_score",
            direction: 1.0,
            unit: "",
            format_fn: fmt_f0,
        },
        FactorDef {
            name: "HRV",
            col: "hrv_last_night",
            direction: 1.0,
            unit: "ms",
            format_fn: fmt_f0,
        },
        FactorDef {
            name: "Stress",
            col: "avg_stress",
            direction: -1.0,
            unit: "",
            format_fn: fmt_f0,
        },
        FactorDef {
            name: "Body Battery",
            col: "body_battery_start",
            direction: 1.0,
            unit: "",
            format_fn: fmt_f0,
        },
    ];

    println!(
        "  {:<16} {:>10}  {:>10}  Status",
        "Factor", "Today", "Avg (90d)"
    );
    println!("  {:-<16} {:-<10}  {:-<10}  {:-<10}", "", "", "", "");

    for def in &defs {
        let Some(val) = today_val(def.col) else {
            continue;
        };
        let Some((mean, std)) = baseline_of(def.col, &recent_health) else {
            continue;
        };

        let factor = Factor {
            name: def.name,
            value: val,
            mean,
            std,
            direction: def.direction,
        };

        let (indicator, status) = factor.status();
        let z = factor.z_score();
        readiness_score += z;
        factor_count += 1;

        let val_s = (def.format_fn)(val);
        let mean_s = (def.format_fn)(mean);
        let unit = if def.unit.is_empty() {
            String::new()
        } else {
            format!(" {}", def.unit)
        };

        println!(
            "{} {:<16} {:>8}{:<3} {:>8}{:<3} {}",
            indicator, def.name, val_s, unit, mean_s, unit, status
        );

        factors.push(factor);
    }

    if factor_count == 0 {
        println!("\nNo health data for today.");
        return Ok(());
    }

    // Normalize score to 0-10 scale
    // z-scores typically range -2 to +2, with N factors summed
    let avg_z = readiness_score / factor_count as f64;
    // Map avg_z from [-2, +2] to [1, 10]
    let score_10 = ((avg_z + 2.0) / 4.0 * 9.0 + 1.0).clamp(1.0, 10.0);

    let (score_color, score_label) = if score_10 >= 7.0 {
        ("\x1b[32m", "Ready to push")
    } else if score_10 >= 4.0 {
        ("\x1b[33m", "Normal day")
    } else {
        ("\x1b[31m", "Recovery day")
    };

    println!(
        "\n  Readiness: {}{:.0}/10\x1b[0m — {}",
        score_color, score_10, score_label
    );

    // Training context
    println!("\n--- Training Context ---\n");

    let activities = data::load_activities(config)?
        .filter(data::type_in_list(RUNNING_TYPES))
        .select([
            col("activity_id"),
            col("start_time_local"),
            col("distance_m"),
        ])
        .sort(["start_time_local"], Default::default())
        .collect()?;

    let times = activities.column("start_time_local")?.datetime()?;
    let dists = activities.column("distance_m")?.f64()?;

    let mut km_7d = 0.0;
    let mut km_28d = 0.0;
    let mut runs_7d = 0u32;
    let mut last_run_date: Option<chrono::NaiveDate> = None;

    let d7_ago = today - chrono::Duration::days(7);
    let d28_ago = today - chrono::Duration::days(28);

    for i in 0..activities.height() {
        let Some(ts) = times.get(i) else { continue };
        let Some(date) =
            chrono::DateTime::from_timestamp_micros(ts).map(|dt| dt.naive_utc().date())
        else {
            continue;
        };
        let dist = dists.get(i).unwrap_or(0.0) / 1000.0;

        if date > d7_ago && date <= today {
            km_7d += dist;
            runs_7d += 1;
        }
        if date > d28_ago && date <= today {
            km_28d += dist;
        }
        if date <= today {
            last_run_date = Some(date);
        }
    }

    let days_off = last_run_date.map(|d| (today - d).num_days()).unwrap_or(0);

    println!("  7-day volume:  {:.0} km ({} runs)", km_7d, runs_7d);
    println!("  28-day volume: {:.0} km", km_28d);
    println!("  Days since last run: {}", days_off);

    // Historical CE lookup: what CE do you get in similar conditions?
    println!("\n--- Expected Performance ---\n");

    let details_dir = config.garmin_storage_path.join("activity_details");
    if !details_dir.exists() {
        println!(
            "  (No detail data — run `model-health fetch --only activity-details` for CE predictions)"
        );
        return Ok(());
    }

    // Load all run CEs with their day's health data
    let ids = activities.column("activity_id")?.i64()?;
    let mut historical: Vec<(f64, f64)> = Vec::new(); // (readiness_z, ce)

    let health_by_date: HashMap<chrono::NaiveDate, HashMap<String, f64>> = {
        let mut map = HashMap::new();
        let dates_col = daily.column("date")?.date()?;
        for i in 0..daily.height() {
            let Some(d) = dates_col.get(i) else { continue };
            let Some(date) = chrono::NaiveDate::from_ymd_opt(1970, 1, 1)
                .and_then(|epoch| epoch.checked_add_signed(chrono::Duration::days(d as i64)))
            else {
                continue;
            };
            let mut row_map = HashMap::new();
            for def in &defs {
                if let Ok(col) = daily.column(def.col)
                    && let Ok(f) = col.as_materialized_series().cast(&DataType::Float64)
                    && let Ok(ca) = f.f64()
                    && let Some(v) = ca.get(i)
                {
                    row_map.insert(def.col.to_string(), v);
                }
            }
            map.insert(date, row_map);
        }
        map
    };

    for i in 0..activities.height() {
        let Some(aid) = ids.get(i) else { continue };
        let Some(ts) = times.get(i) else { continue };
        let Some(date) =
            chrono::DateTime::from_timestamp_micros(ts).map(|dt| dt.naive_utc().date())
        else {
            continue;
        };

        let detail_path = details_dir.join(format!("{}.parquet", aid));
        if !detail_path.exists() {
            continue;
        }

        // Compute this run's readiness z-score from its day's health data
        let Some(health_row) = health_by_date.get(&date) else {
            continue;
        };

        let mut run_z_sum = 0.0;
        let mut run_z_count = 0;
        for factor in &factors {
            if let Some(&val) = health_row.get(
                defs.iter()
                    .find(|d| d.name == factor.name)
                    .map(|d| d.col)
                    .unwrap_or(""),
            ) && factor.std > 0.001
            {
                run_z_sum += (val - factor.mean) / factor.std * factor.direction;
                run_z_count += 1;
            }
        }

        if run_z_count < 2 {
            continue;
        }

        let run_avg_z = run_z_sum / run_z_count as f64;

        // Compute CE for this run
        let Some(rows) = fitness::load_steady_rows(&detail_path) else {
            continue;
        };
        if rows.len() < 30 {
            continue;
        }
        let n = rows.len() as f64;
        let avg_gap = rows.iter().map(|r| r.gap_speed).sum::<f64>() / n;
        let avg_hr = rows.iter().map(|r| r.heart_rate).sum::<f64>() / n;
        let temps: Vec<f64> = rows.iter().filter_map(|r| r.temperature).collect();
        let avg_temp = if temps.is_empty() {
            None
        } else {
            Some(temps.iter().sum::<f64>() / temps.len() as f64)
        };
        let ce_raw = avg_gap / avg_hr;
        let ce = match avg_temp {
            Some(t) => ce_raw - fitness::TEMP_COEFF * (t - fitness::TEMP_REF),
            None => ce_raw,
        };

        historical.push((run_avg_z, ce));
    }

    if historical.is_empty() {
        println!("  Not enough historical data for CE prediction.");
        return Ok(());
    }

    // Find runs with similar readiness z-score (within ±0.5)
    let similar: Vec<f64> = historical
        .iter()
        .filter(|(z, _)| (z - avg_z).abs() < 0.5)
        .map(|(_, ce)| *ce)
        .collect();

    let all_ce: Vec<f64> = historical.iter().map(|(_, ce)| *ce).collect();
    let overall_mean = all_ce.iter().sum::<f64>() / all_ce.len() as f64;

    if similar.len() >= 5 {
        let sim_mean = similar.iter().sum::<f64>() / similar.len() as f64;
        let mut sorted = similar.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p25 = sorted[sorted.len() / 4];
        let p75 = sorted[3 * sorted.len() / 4];

        println!(
            "  On days like today, your CE is typically: {:.4} (range {:.4}-{:.4})",
            sim_mean, p25, p75
        );
        println!("  Your overall average CE: {:.4}", overall_mean);

        let diff = sim_mean - overall_mean;
        if diff > 0.0005 {
            println!("  Today's conditions are \x1b[32mbetter than average\x1b[0m for running.");
        } else if diff < -0.0005 {
            println!("  Today's conditions are \x1b[31mworse than average\x1b[0m for running.");
        } else {
            println!("  Today's conditions are \x1b[33mtypical\x1b[0m for your running.");
        }
    } else {
        println!(
            "  Not enough similar days for CE prediction ({} matches, need 5).",
            similar.len()
        );
        println!("  Overall average CE: {:.4}", overall_mean);
    }

    // Recommendation
    println!("\n--- Recommendation ---\n");
    if score_10 >= 7.0 && days_off >= 2 {
        println!("  Good recovery + rested. Today is a good day for a hard effort or long run.");
    } else if score_10 >= 7.0 {
        println!("  Good recovery. A solid steady run should feel good today.");
    } else if score_10 >= 4.0 && days_off >= 1 {
        println!("  Normal recovery. An easy to moderate effort is appropriate.");
    } else if score_10 >= 4.0 {
        println!("  Normal recovery but ran recently. Keep it easy if you go.");
    } else if days_off == 0 {
        println!("  Recovery is low and you already ran today. Rest tomorrow.");
    } else {
        println!("  Recovery is below average. Consider a rest day or very easy short run.");
    }

    Ok(())
}
