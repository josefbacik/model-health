//! Daily health dashboard.
//!
//! Combines weight, nutrition, sleep, stress/recovery, blood pressure, and
//! training context into a single view over a configurable window. The key
//! insight: on rest days (no activity), avg_stress reflects external life
//! stress; on training days it's mixed. This distinction is surfaced explicitly.

use chrono::NaiveDate;
use polars::prelude::*;
use std::collections::HashMap;

use crate::config::Config;
use crate::data;
use crate::error::Result;
use crate::validation;

// ── Lightweight row structs for per-day data ────────────────────────────

struct ActivitySummary {
    activity_type: String,
    distance_m: f64,
    duration_sec: f64,
    avg_hr: Option<i32>,
    training_load: Option<f64>,
    calories: Option<i32>,
}

struct WeightReading {
    weight_kg: f64,
    body_fat: Option<f64>,
    muscle_mass: Option<f64>,
}

struct NutritionDay {
    consumed_calories: Option<i32>,
    protein_g: Option<f64>,
    carbs_g: Option<f64>,
    fat_g: Option<f64>,
}

struct BpReading {
    systolic: Option<i32>,
    diastolic: Option<i32>,
    pulse: Option<i32>,
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn opt_i32(df: &DataFrame, col: &str, row: usize) -> Option<i32> {
    df.column(col).ok()?.get(row).ok().and_then(|v| match v {
        AnyValue::Int32(n) => Some(n),
        AnyValue::Null => None,
        _ => None,
    })
}

fn opt_f64(df: &DataFrame, col: &str, row: usize) -> Option<f64> {
    df.column(col).ok()?.get(row).ok().and_then(|v| match v {
        AnyValue::Float64(n) => Some(n),
        AnyValue::Float32(n) => Some(n as f64),
        AnyValue::Int32(n) => Some(n as f64),
        AnyValue::Int64(n) => Some(n as f64),
        AnyValue::Null => None,
        _ => None,
    })
}

fn get_date(df: &DataFrame, row: usize) -> Option<NaiveDate> {
    df.column("date").ok()?.get(row).ok().and_then(|v| match v {
        AnyValue::Date(days) => chrono::NaiveDate::from_num_days_from_ce_opt(days + 719_163),
        _ => None,
    })
}

fn get_string(df: &DataFrame, col: &str, row: usize) -> Option<String> {
    df.column(col).ok()?.get(row).ok().and_then(|v| match v {
        AnyValue::String(s) => Some(s.to_string()),
        AnyValue::StringOwned(s) => Some(s.to_string()),
        AnyValue::Null => None,
        _ => None,
    })
}

fn format_duration(seconds: f64) -> String {
    let total = seconds.round() as u64;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{}:{:02}:{:02}", h, m, s)
    } else {
        format!("{}:{:02}", m, s)
    }
}

/// Load a dataset, returning None if not available (no data files yet).
fn try_load<F>(loader: F) -> Option<LazyFrame>
where
    F: FnOnce() -> Result<LazyFrame>,
{
    loader().ok()
}

// ── Entry point ─────────────────────────────────────────────────────────

pub fn run(config: &Config, days: u32) -> Result<()> {
    let today = chrono::Utc::now().date_naive();
    let from_date = today - chrono::Duration::days(days as i64 - 1);
    let d90_ago = today - chrono::Duration::days(90);

    // ── Load daily health (required) ────────────────────────────────
    // Collect first, then filter — predicate pushdown can panic when
    // some parquet files lack columns present in the union schema.
    let daily_all = {
        let lf = data::load_daily_health(config)?;
        let lf = validation::clean_daily_health(lf)?;
        lf.collect()?
    };

    let daily = daily_all
        .clone()
        .lazy()
        .filter(col("date").gt_eq(lit(from_date)))
        .sort(["date"], Default::default())
        .collect()?;

    // ── Load 90-day health for baselines ────────────────────────────
    let daily_90 = daily_all
        .lazy()
        .filter(col("date").gt_eq(lit(d90_ago)))
        .collect()?;

    // ── Load activities (once, derive both window map and 90-day set) ─
    let (activity_map, activity_dates_90) = {
        let mut map: HashMap<NaiveDate, Vec<ActivitySummary>> = HashMap::new();
        let mut dates_90: std::collections::HashSet<NaiveDate> = std::collections::HashSet::new();

        if let Some(lf) = try_load(|| data::load_activities(config)) {
            let adf = lf
                .select([
                    col("start_time_local"),
                    col("activity_type"),
                    col("distance_m"),
                    col("duration_sec"),
                    col("avg_hr"),
                    col("training_load"),
                    col("calories"),
                ])
                .sort(["start_time_local"], Default::default())
                .collect()?;

            for i in 0..adf.height() {
                let ts = adf
                    .column("start_time_local")
                    .ok()
                    .and_then(|c| c.get(i).ok());
                let date = match ts {
                    Some(AnyValue::Datetime(us, _, _)) => {
                        chrono::DateTime::from_timestamp_micros(us).map(|dt| dt.naive_utc().date())
                    }
                    _ => None,
                };
                let Some(date) = date else { continue };

                // Track all activity dates in the 90-day window for baseline
                if date >= d90_ago {
                    dates_90.insert(date);
                }

                if date < from_date || date > today {
                    continue;
                }

                let activity_type =
                    get_string(&adf, "activity_type", i).unwrap_or_else(|| "unknown".into());
                let distance_m = opt_f64(&adf, "distance_m", i).unwrap_or(0.0);
                let duration_sec = opt_f64(&adf, "duration_sec", i).unwrap_or(0.0);
                let avg_hr = opt_i32(&adf, "avg_hr", i);
                let training_load = opt_f64(&adf, "training_load", i);
                let calories = opt_i32(&adf, "calories", i);

                map.entry(date).or_default().push(ActivitySummary {
                    activity_type,
                    distance_m,
                    duration_sec,
                    avg_hr,
                    training_load,
                    calories,
                });
            }
        }
        (map, dates_90)
    };

    // ── 90-day rest-day stress baseline ──────────────────────────────
    let rest_day_baseline_stress: Option<f64> = {
        let mut rest_stresses = Vec::new();
        for i in 0..daily_90.height() {
            let Some(date) = get_date(&daily_90, i) else {
                continue;
            };
            if activity_dates_90.contains(&date) {
                continue;
            }
            if let Some(s) = opt_i32(&daily_90, "avg_stress", i) {
                rest_stresses.push(s as f64);
            }
        }
        if rest_stresses.len() >= 5 {
            Some(rest_stresses.iter().sum::<f64>() / rest_stresses.len() as f64)
        } else {
            None
        }
    };

    // ── Load sparse datasets ────────────────────────────────────────
    let weight_map: HashMap<NaiveDate, WeightReading> = {
        let mut map = HashMap::new();
        if let Some(lf) = try_load(|| data::load_weight(config))
            && let Ok(wdf) = lf
                .filter(col("date").gt_eq(lit(from_date)))
                .sort(["date"], Default::default())
                .collect()
        {
            for i in 0..wdf.height() {
                let Some(date) = get_date(&wdf, i) else {
                    continue;
                };
                let Some(wt) = opt_f64(&wdf, "weight_kg", i) else {
                    continue;
                };
                map.insert(
                    date,
                    WeightReading {
                        weight_kg: wt,
                        body_fat: opt_f64(&wdf, "body_fat", i),
                        muscle_mass: opt_f64(&wdf, "muscle_mass", i),
                    },
                );
            }
        }
        map
    };

    let nutrition_map: HashMap<NaiveDate, NutritionDay> = {
        let mut map = HashMap::new();
        if let Some(lf) = try_load(|| data::load_nutrition(config))
            && let Ok(ndf) = lf
                .filter(col("date").gt_eq(lit(from_date)))
                .sort(["date"], Default::default())
                .collect()
        {
            for i in 0..ndf.height() {
                let Some(date) = get_date(&ndf, i) else {
                    continue;
                };
                map.insert(
                    date,
                    NutritionDay {
                        consumed_calories: opt_i32(&ndf, "consumed_calories", i),
                        protein_g: opt_f64(&ndf, "protein_g", i),
                        carbs_g: opt_f64(&ndf, "carbs_g", i),
                        fat_g: opt_f64(&ndf, "fat_g", i),
                    },
                );
            }
        }
        map
    };

    let bp_map: HashMap<NaiveDate, BpReading> = {
        let mut map = HashMap::new();
        if let Some(lf) = try_load(|| data::load_blood_pressure(config))
            && let Ok(bdf) = lf
                .filter(col("date").gt_eq(lit(from_date)))
                .sort(["date"], Default::default())
                .collect()
        {
            for i in 0..bdf.height() {
                let Some(date) = get_date(&bdf, i) else {
                    continue;
                };
                map.insert(
                    date,
                    BpReading {
                        systolic: opt_i32(&bdf, "systolic", i),
                        diastolic: opt_i32(&bdf, "diastolic", i),
                        pulse: opt_i32(&bdf, "pulse", i),
                    },
                );
            }
        }
        map
    };

    // ── Load performance metrics (training readiness) ───────────────
    let readiness_map: HashMap<NaiveDate, i32> = {
        let mut map = HashMap::new();
        if let Some(lf) = try_load(|| data::load_performance_metrics(config))
            && let Ok(pdf) = lf.filter(col("date").gt_eq(lit(from_date))).collect()
        {
            for i in 0..pdf.height() {
                let Some(date) = get_date(&pdf, i) else {
                    continue;
                };
                if let Some(r) = opt_i32(&pdf, "training_readiness", i) {
                    map.insert(date, r);
                }
            }
        }
        map
    };

    // ── Build total-calories-burned map from daily_health ──────────
    let total_cal_map: HashMap<NaiveDate, i32> = {
        let mut map = HashMap::new();
        for i in 0..daily.height() {
            let Some(date) = get_date(&daily, i) else {
                continue;
            };
            if let Some(tc) = opt_i32(&daily, "total_calories", i) {
                map.insert(date, tc);
            }
        }
        map
    };

    // ── Header ──────────────────────────────────────────────────────
    println!("=== Daily Health Dashboard ===\n");
    println!("  Window: {} to {} ({} days)\n", from_date, today, days);

    // ── Section 1: Weight ───────────────────────────────────────────
    print_weight_section(from_date, today, &weight_map);

    // ── Section 2: Nutrition ────────────────────────────────────────
    print_nutrition_section(from_date, today, &nutrition_map, &total_cal_map);

    // ── Section 3: Sleep ────────────────────────────────────────────
    print_sleep_section(from_date, today, &daily);

    // ── Section 4: Stress & Recovery ────────────────────────────────
    print_stress_section(
        from_date,
        today,
        &daily,
        &activity_map,
        &readiness_map,
        rest_day_baseline_stress,
    );

    // ── Section 5: Blood Pressure ───────────────────────────────────
    print_bp_section(from_date, today, &bp_map);

    // ── Section 6: Training Context ─────────────────────────────────
    print_training_section(from_date, today, &activity_map);

    // ── Section 7: Insights ─────────────────────────────────────────
    print_insights(
        from_date,
        today,
        &weight_map,
        &nutrition_map,
        &daily,
        &activity_map,
        rest_day_baseline_stress,
        &total_cal_map,
    );

    Ok(())
}

// ── Output sections ─────────────────────────────────────────────────────

fn print_weight_section(
    from: NaiveDate,
    to: NaiveDate,
    weight_map: &HashMap<NaiveDate, WeightReading>,
) {
    println!("--- Weight ---\n");

    let mut readings: Vec<(&NaiveDate, &WeightReading)> = weight_map
        .iter()
        .filter(|(d, _)| **d >= from && **d <= to)
        .collect();
    readings.sort_by_key(|(d, _)| *d);

    if readings.is_empty() {
        println!("  No weight readings in this window.\n");
        return;
    }

    println!(
        "  {:<12} {:>8} {:>9} {:>10}",
        "Date", "Weight", "Body Fat", "Muscle"
    );
    println!("  {:-<12} {:-<8} {:-<9} {:-<10}", "", "", "", "");

    for (date, w) in &readings {
        let bf = w
            .body_fat
            .map(|f| format!("{:.1}%", f))
            .unwrap_or_else(|| "---".into());
        let mm = w
            .muscle_mass
            .map(|m| {
                if m > 1000.0 {
                    format!("{:.1} kg", m / 1000.0)
                } else {
                    format!("{:.1} kg", m)
                }
            })
            .unwrap_or_else(|| "---".into());
        println!(
            "  {:<12} {:>6.1} kg {:>9} {:>10}",
            date, w.weight_kg, bf, mm
        );
    }

    if readings.len() >= 2 {
        let first = readings.first().unwrap().1.weight_kg;
        let last = readings.last().unwrap().1.weight_kg;
        let change = last - first;
        let (color, arrow) = if change < -0.1 {
            ("\x1b[32m", "v") // green down = losing
        } else if change > 0.1 {
            ("\x1b[33m", "^") // yellow up
        } else {
            ("", "=")
        };
        println!("\n  Change: {}{}{:+.1} kg\x1b[0m\n", color, arrow, change);
    } else {
        println!();
    }
}

fn print_nutrition_section(
    from: NaiveDate,
    to: NaiveDate,
    nutrition_map: &HashMap<NaiveDate, NutritionDay>,
    total_cal_map: &HashMap<NaiveDate, i32>,
) {
    println!("--- Nutrition ---\n");

    let mut dates: Vec<&NaiveDate> = nutrition_map
        .keys()
        .filter(|d| **d >= from && **d <= to)
        .collect();
    dates.sort();

    if dates.is_empty() {
        println!("  No nutrition data logged in this window.\n");
        return;
    }

    println!(
        "  {:<12} {:>8} {:>8} {:>8} {:>6} {:>8}",
        "Date", "Calories", "Protein", "Carbs", "Fat", "Balance"
    );
    println!(
        "  {:-<12} {:-<8} {:-<8} {:-<8} {:-<6} {:-<8}",
        "", "", "", "", "", ""
    );

    let mut total_consumed = 0i64;
    let mut total_burned = 0i64;
    let mut balance_count = 0;

    for date in &dates {
        let n = &nutrition_map[*date];
        let cal_s = n
            .consumed_calories
            .map(|c| format!("{}", c))
            .unwrap_or_else(|| "---".into());
        let pro_s = n
            .protein_g
            .map(|p| format!("{:.0}g", p))
            .unwrap_or_else(|| "---".into());
        let carb_s = n
            .carbs_g
            .map(|c| format!("{:.0}g", c))
            .unwrap_or_else(|| "---".into());
        let fat_s = n
            .fat_g
            .map(|f| format!("{:.0}g", f))
            .unwrap_or_else(|| "---".into());

        let balance_s = match (n.consumed_calories, total_cal_map.get(*date)) {
            (Some(consumed), Some(&burned)) => {
                let bal = consumed - burned;
                total_consumed += consumed as i64;
                total_burned += burned as i64;
                balance_count += 1;
                let color = if bal < -500 {
                    "\x1b[31m"
                } else if bal < 0 {
                    "\x1b[33m"
                } else {
                    "\x1b[32m"
                };
                format!("{}{:+}\x1b[0m", color, bal)
            }
            _ => "---".into(),
        };

        println!(
            "  {:<12} {:>8} {:>8} {:>8} {:>6} {:>8}",
            date, cal_s, pro_s, carb_s, fat_s, balance_s
        );
    }

    // Averages
    let avg_cal: Option<f64> = {
        let vals: Vec<f64> = dates
            .iter()
            .filter_map(|d| nutrition_map[*d].consumed_calories.map(|c| c as f64))
            .collect();
        if vals.is_empty() {
            None
        } else {
            Some(vals.iter().sum::<f64>() / vals.len() as f64)
        }
    };
    let avg_pro: Option<f64> = {
        let vals: Vec<f64> = dates
            .iter()
            .filter_map(|d| nutrition_map[*d].protein_g)
            .collect();
        if vals.is_empty() {
            None
        } else {
            Some(vals.iter().sum::<f64>() / vals.len() as f64)
        }
    };

    println!();
    if let Some(ac) = avg_cal {
        print!("  Avg consumed: {:.0} cal", ac);
        if let Some(ap) = avg_pro {
            print!("   Avg protein: {:.0}g", ap);
        }
        if balance_count > 0 {
            let avg_bal = (total_consumed - total_burned) as f64 / balance_count as f64;
            let color = if avg_bal < -500.0 {
                "\x1b[31m"
            } else if avg_bal < 0.0 {
                "\x1b[33m"
            } else {
                "\x1b[32m"
            };
            print!("   Avg balance: {}{:+.0}\x1b[0m", color, avg_bal);
        }
        println!();
    }

    // Macro split
    {
        let total_p: f64 = dates
            .iter()
            .filter_map(|d| nutrition_map[*d].protein_g)
            .sum();
        let total_c: f64 = dates.iter().filter_map(|d| nutrition_map[*d].carbs_g).sum();
        let total_f: f64 = dates.iter().filter_map(|d| nutrition_map[*d].fat_g).sum();
        let total_cal_from_macros = total_p * 4.0 + total_c * 4.0 + total_f * 9.0;
        if total_cal_from_macros > 0.0 {
            println!(
                "  Macro split: P {:.0}% / C {:.0}% / F {:.0}%",
                total_p * 4.0 / total_cal_from_macros * 100.0,
                total_c * 4.0 / total_cal_from_macros * 100.0,
                total_f * 9.0 / total_cal_from_macros * 100.0,
            );
        }
    }
    println!();
}

fn print_sleep_section(from: NaiveDate, to: NaiveDate, daily: &DataFrame) {
    println!("--- Sleep ---\n");

    println!("  {:<12} {:>6} {:>6} Trend", "Date", "Hours", "Score");
    println!("  {:-<12} {:-<6} {:-<6} {:-<5}", "", "", "", "");

    let mut prev_score: Option<i32> = None;
    let mut total_hours = 0.0f64;
    let mut total_score = 0i64;
    let mut count_hours = 0;
    let mut count_score = 0;

    for i in 0..daily.height() {
        let Some(date) = get_date(daily, i) else {
            continue;
        };
        if date < from || date > to {
            continue;
        }

        let sleep_sec = opt_i32(daily, "sleep_seconds", i);
        let score = opt_i32(daily, "sleep_score", i);

        let hours_s = sleep_sec
            .map(|s| {
                let h = s as f64 / 3600.0;
                total_hours += h;
                count_hours += 1;
                format!("{:.1}", h)
            })
            .unwrap_or_else(|| "---".into());

        let score_s = score
            .map(|s| {
                total_score += s as i64;
                count_score += 1;
                format!("{}", s)
            })
            .unwrap_or_else(|| "---".into());

        let trend = match (score, prev_score) {
            (Some(cur), Some(prev)) => {
                if cur > prev + 2 {
                    "\x1b[32m^\x1b[0m"
                } else if cur < prev - 2 {
                    "\x1b[31mv\x1b[0m"
                } else {
                    "="
                }
            }
            _ => " ",
        };

        println!("  {:<12} {:>6} {:>6} {}", date, hours_s, score_s, trend);

        if score.is_some() {
            prev_score = score;
        }
    }

    println!();
    if count_hours > 0 {
        print!("  Avg: {:.1} hrs", total_hours / count_hours as f64);
    }
    if count_score > 0 {
        print!(
            "   Avg score: {:.0}",
            total_score as f64 / count_score as f64
        );
    }
    if count_hours > 0 || count_score > 0 {
        println!();
    }
    println!();
}

fn print_stress_section(
    from: NaiveDate,
    to: NaiveDate,
    daily: &DataFrame,
    activity_map: &HashMap<NaiveDate, Vec<ActivitySummary>>,
    readiness_map: &HashMap<NaiveDate, i32>,
    rest_day_baseline: Option<f64>,
) {
    println!("--- Stress & Recovery ---\n");

    println!(
        "  {:<12} {:>6} {:>4} {:>4} {:>10} {:>5}  Training",
        "Date", "Stress", "RHR", "HRV", "BB", "Ready"
    );
    println!(
        "  {:-<12} {:-<6} {:-<4} {:-<4} {:-<10} {:-<5}  {:-<16}",
        "", "", "", "", "", "", ""
    );

    let mut rest_stresses = Vec::new();
    let mut train_stresses = Vec::new();

    for i in 0..daily.height() {
        let Some(date) = get_date(daily, i) else {
            continue;
        };
        if date < from || date > to {
            continue;
        }

        let stress = opt_i32(daily, "avg_stress", i);
        let rhr = opt_i32(daily, "resting_hr", i);
        let hrv = opt_i32(daily, "hrv_last_night", i);
        let bb_start = opt_i32(daily, "body_battery_start", i);
        let bb_end = opt_i32(daily, "body_battery_end", i);

        let is_training = activity_map.contains_key(&date);

        if let Some(s) = stress {
            if is_training {
                train_stresses.push(s as f64);
            } else {
                rest_stresses.push(s as f64);
            }
        }

        // Color stress relative to rest-day baseline
        let stress_s = match stress {
            Some(s) => {
                let baseline = rest_day_baseline.unwrap_or(30.0);
                if !is_training && (s as f64) > baseline + 5.0 {
                    format!("\x1b[31m{:>6}\x1b[0m", s) // red: elevated rest-day stress
                } else if !is_training && (s as f64) < baseline - 5.0 {
                    format!("\x1b[32m{:>6}\x1b[0m", s) // green: low rest-day stress
                } else {
                    format!("{:>6}", s)
                }
            }
            None => format!("{:>6}", "---"),
        };

        let rhr_s = rhr
            .map(|r| format!("{:>4}", r))
            .unwrap_or_else(|| format!("{:>4}", "---"));
        let hrv_s = hrv
            .map(|h| format!("{:>4}", h))
            .unwrap_or_else(|| format!("{:>4}", "---"));

        let bb_s = match (bb_start, bb_end) {
            (Some(s), Some(e)) => {
                let end_color = if e >= 50 {
                    "\x1b[32m"
                } else if e >= 25 {
                    "\x1b[33m"
                } else {
                    "\x1b[31m"
                };
                format!("{:>3} > {}{:>3}\x1b[0m", s, end_color, e)
            }
            (Some(s), None) => format!("{:>3} > {:>3}", s, "---"),
            _ => format!("{:>10}", "---"),
        };

        let ready_s = readiness_map
            .get(&date)
            .map(|r| {
                let color = if *r >= 60 {
                    "\x1b[32m"
                } else if *r >= 30 {
                    "\x1b[33m"
                } else {
                    "\x1b[31m"
                };
                format!("{}{:>5}\x1b[0m", color, r)
            })
            .unwrap_or_else(|| format!("{:>5}", "---"));

        let training_s = if let Some(acts) = activity_map.get(&date) {
            let summaries: Vec<String> = acts
                .iter()
                .map(|a| {
                    if a.distance_m > 0.0 {
                        format!("{:.1}k {}", a.distance_m / 1000.0, a.activity_type)
                    } else {
                        a.activity_type.clone()
                    }
                })
                .collect();
            summaries.join(", ")
        } else {
            "REST".into()
        };

        println!(
            "  {:<12} {} {} {} {:>10} {}  {}",
            date, stress_s, rhr_s, hrv_s, bb_s, ready_s, training_s
        );
    }

    // Summary
    println!();
    let rest_avg = if !rest_stresses.is_empty() {
        Some(rest_stresses.iter().sum::<f64>() / rest_stresses.len() as f64)
    } else {
        None
    };
    let train_avg = if !train_stresses.is_empty() {
        Some(train_stresses.iter().sum::<f64>() / train_stresses.len() as f64)
    } else {
        None
    };

    if let Some(ra) = rest_avg {
        print!("  Rest-day avg stress: {:.0}", ra);
        if let Some(baseline) = rest_day_baseline {
            let diff = ra - baseline;
            let color = if diff > 5.0 {
                "\x1b[31m"
            } else if diff < -5.0 {
                "\x1b[32m"
            } else {
                ""
            };
            print!(
                "  (90-day baseline: {:.0}, {}{:+.0}\x1b[0m)",
                baseline, color, diff
            );
        }
        println!();
    }
    if let Some(ta) = train_avg {
        println!("  Training-day avg stress: {:.0}", ta);
    }
    if let (Some(ra), Some(ta)) = (rest_avg, train_avg) {
        println!("  Training stress delta: {:+.0}", ta - ra);
    }
    println!();
}

fn print_bp_section(from: NaiveDate, to: NaiveDate, bp_map: &HashMap<NaiveDate, BpReading>) {
    println!("--- Blood Pressure ---\n");

    let mut readings: Vec<(&NaiveDate, &BpReading)> = bp_map
        .iter()
        .filter(|(d, _)| **d >= from && **d <= to)
        .collect();
    readings.sort_by_key(|(d, _)| *d);

    if readings.is_empty() {
        println!("  No blood pressure readings in this window.\n");
        return;
    }

    println!("  {:<12} {:>8}/{:<9} {:>5}", "Date", "Sys", "Dia", "Pulse");
    println!("  {:-<12} {:-<18} {:-<5}", "", "", "");

    for (date, bp) in &readings {
        let sys = bp
            .systolic
            .map(|s| format!("{}", s))
            .unwrap_or_else(|| "---".into());
        let dia = bp
            .diastolic
            .map(|d| format!("{}", d))
            .unwrap_or_else(|| "---".into());
        let pulse = bp
            .pulse
            .map(|p| format!("{}", p))
            .unwrap_or_else(|| "---".into());
        println!("  {:<12} {:>8}/{:<9} {:>5}", date, sys, dia, pulse);
    }
    println!();
}

fn print_training_section(
    from: NaiveDate,
    to: NaiveDate,
    activity_map: &HashMap<NaiveDate, Vec<ActivitySummary>>,
) {
    println!("--- Training ---\n");

    // Flatten and sort by date
    let mut all: Vec<(NaiveDate, &ActivitySummary)> = Vec::new();
    for (date, acts) in activity_map {
        if *date >= from && *date <= to {
            for a in acts {
                all.push((*date, a));
            }
        }
    }
    all.sort_by_key(|(d, _)| *d);

    if all.is_empty() {
        println!("  No activities in this window.\n");
        return;
    }

    println!(
        "  {:<12} {:<18} {:>7} {:>9} {:>4} {:>6}",
        "Date", "Activity", "Dist", "Duration", "HR", "Load"
    );
    println!(
        "  {:-<12} {:-<18} {:-<7} {:-<9} {:-<4} {:-<6}",
        "", "", "", "", "", ""
    );

    let mut total_dist = 0.0f64;
    let mut total_load = 0.0f64;
    let mut total_cal = 0i64;

    for (date, a) in &all {
        let dist_s = if a.distance_m > 0.0 {
            total_dist += a.distance_m;
            format!("{:.1} km", a.distance_m / 1000.0)
        } else {
            "---".into()
        };
        let dur_s = format_duration(a.duration_sec);
        let hr_s = a
            .avg_hr
            .map(|h| format!("{}", h))
            .unwrap_or_else(|| "---".into());
        let load_s = a
            .training_load
            .map(|l| {
                total_load += l;
                format!("{:.0}", l)
            })
            .unwrap_or_else(|| "---".into());
        if let Some(c) = a.calories {
            total_cal += c as i64;
        }

        println!(
            "  {:<12} {:<18} {:>7} {:>9} {:>4} {:>6}",
            date, a.activity_type, dist_s, dur_s, hr_s, load_s
        );
    }

    println!();
    println!(
        "  Volume: {:.1} km across {} activities   Total load: {:.0}   Calories burned: {}",
        total_dist / 1000.0,
        all.len(),
        total_load,
        total_cal
    );
    println!();
}

#[allow(clippy::too_many_arguments)]
fn print_insights(
    from: NaiveDate,
    to: NaiveDate,
    weight_map: &HashMap<NaiveDate, WeightReading>,
    nutrition_map: &HashMap<NaiveDate, NutritionDay>,
    daily: &DataFrame,
    activity_map: &HashMap<NaiveDate, Vec<ActivitySummary>>,
    rest_day_baseline: Option<f64>,
    total_cal_map: &HashMap<NaiveDate, i32>,
) {
    println!("--- Insights ---\n");

    // Weight direction
    let mut weight_readings: Vec<(&NaiveDate, &WeightReading)> = weight_map
        .iter()
        .filter(|(d, _)| **d >= from && **d <= to)
        .collect();
    weight_readings.sort_by_key(|(d, _)| *d);

    if weight_readings.len() >= 2 {
        let first = weight_readings.first().unwrap().1.weight_kg;
        let last = weight_readings.last().unwrap().1.weight_kg;
        let change = last - first;
        let direction = if change < -0.3 {
            "\x1b[32mtrending down\x1b[0m"
        } else if change > 0.3 {
            "\x1b[33mtrending up\x1b[0m"
        } else {
            "stable"
        };
        println!("  Weight: {:+.1} kg over window ({})", change, direction);
    }

    // Calorie balance (using Garmin total daily burn)
    if !nutrition_map.is_empty() {
        let mut total_consumed = 0i64;
        let mut total_burned = 0i64;
        let mut count = 0;

        for (date, n) in nutrition_map {
            if *date >= from
                && *date <= to
                && let (Some(c), Some(&burned)) = (n.consumed_calories, total_cal_map.get(date))
            {
                total_consumed += c as i64;
                total_burned += burned as i64;
                count += 1;
            }
        }

        if count > 0 {
            let avg_bal = (total_consumed - total_burned) as f64 / count as f64;
            let label = if avg_bal < 0.0 { "deficit" } else { "surplus" };
            println!("  Nutrition: avg {:+.0} cal/day ({})", avg_bal, label);
        }
    }

    // Sleep trend: compare first half vs second half
    {
        let mut scores: Vec<(NaiveDate, i32)> = Vec::new();
        for i in 0..daily.height() {
            if let Some(date) = get_date(daily, i)
                && date >= from
                && date <= to
                && let Some(s) = opt_i32(daily, "sleep_score", i)
            {
                scores.push((date, s));
            }
        }
        if scores.len() >= 4 {
            let mid = scores.len() / 2;
            let first_avg = scores[..mid].iter().map(|(_, s)| *s as f64).sum::<f64>() / mid as f64;
            let second_avg = scores[mid..].iter().map(|(_, s)| *s as f64).sum::<f64>()
                / (scores.len() - mid) as f64;
            let diff = second_avg - first_avg;
            let trend = if diff > 3.0 {
                "\x1b[32mimproving\x1b[0m"
            } else if diff < -3.0 {
                "\x1b[31mdeclining\x1b[0m"
            } else {
                "stable"
            };
            println!(
                "  Sleep: avg score {:.0} ({})",
                scores.iter().map(|(_, s)| *s as f64).sum::<f64>() / scores.len() as f64,
                trend
            );
        }
    }

    // Rest-day stress vs baseline
    {
        let mut rest_stresses = Vec::new();
        for i in 0..daily.height() {
            if let Some(date) = get_date(daily, i)
                && date >= from
                && date <= to
                && !activity_map.contains_key(&date)
                && let Some(s) = opt_i32(daily, "avg_stress", i)
            {
                rest_stresses.push(s as f64);
            }
        }

        if !rest_stresses.is_empty() {
            let window_avg = rest_stresses.iter().sum::<f64>() / rest_stresses.len() as f64;
            if let Some(baseline) = rest_day_baseline {
                let diff = window_avg - baseline;
                if diff > 5.0 {
                    println!(
                        "  Stress: rest-day avg {:.0} vs baseline {:.0} — \x1b[31melevated, check external factors\x1b[0m",
                        window_avg, baseline
                    );
                } else if diff < -5.0 {
                    println!(
                        "  Stress: rest-day avg {:.0} vs baseline {:.0} — \x1b[32mbelow normal, good recovery environment\x1b[0m",
                        window_avg, baseline
                    );
                } else {
                    println!(
                        "  Stress: rest-day avg {:.0} (within normal range of baseline {:.0})",
                        window_avg, baseline
                    );
                }
            }
        }
    }

    // Training volume
    {
        let mut total_dist = 0.0f64;
        let mut count = 0usize;
        let mut total_load = 0.0f64;
        for (date, acts) in activity_map {
            if *date >= from && *date <= to {
                for a in acts {
                    total_dist += a.distance_m;
                    if let Some(l) = a.training_load {
                        total_load += l;
                    }
                    count += 1;
                }
            }
        }
        if count > 0 {
            println!(
                "  Training: {:.1} km across {} sessions, total load {:.0}",
                total_dist / 1000.0,
                count,
                total_load
            );
        }
    }

    println!();
}
