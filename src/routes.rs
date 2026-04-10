//! GPS-based route detection and per-route fitness tracking.
//!
//! Clusters running activities by start location (200m radius) and distance
//! (10% tolerance) to identify common routes, then tracks cardiac efficiency
//! on each route over time. Filters to steady-effort runs so interval workouts
//! on the same course don't contaminate the comparison.
//!
//! Run with `model-health routes` to see the report.

use chrono::Datelike;
use polars::prelude::*;
use std::collections::HashMap;

use crate::config::Config;
use crate::data;
use crate::data::RUNNING_TYPES;
use crate::error::Result;
use crate::fitness;

/// Maximum distance (meters) between start points to consider same route.
const START_RADIUS_M: f64 = 200.0;

/// Maximum fractional difference in total distance to consider same route.
const DISTANCE_TOLERANCE: f64 = 0.10;

/// Minimum number of steady runs on a route to include in the report.
const MIN_ROUTE_RUNS: usize = 5;

/// Maximum HR coefficient of variation for a "steady" run (same as fitness module).
const MAX_HR_CV_PCT: f64 = 8.5;

/// Haversine distance in meters between two lat/lon points.
fn haversine(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const R: f64 = 6_371_000.0;
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    R * 2.0 * a.sqrt().asin()
}

/// A detected route (cluster of runs from the same start with similar distance).
struct Route {
    start_lat: f64,
    start_lon: f64,
    avg_distance_km: f64,
    runs: Vec<RoutedRun>,
}

/// A single run on a detected route.
struct RoutedRun {
    date: chrono::NaiveDate,
    distance_km: f64,
    ce: f64,
    gap_speed: f64,
    avg_hr: f64,
    avg_temp: Option<f64>,
    location_name: String,
}

/// Number of days to consider "recent" for filtering routes.
const RECENT_DAYS: i64 = 90;

/// Entry point.
pub fn run(config: &Config, show_all: bool) -> Result<()> {
    let activities = data::load_activities(config)?
        .filter(data::type_in_list(RUNNING_TYPES))
        .filter(col("distance_m").gt(lit(3000)))
        .filter(col("start_lat").is_not_null())
        .select([
            col("activity_id"),
            col("start_time_local"),
            col("distance_m"),
            col("start_lat"),
            col("start_lon"),
            col("location_name"),
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

    let ids = activities.column("activity_id")?.i64()?;
    let times = activities.column("start_time_local")?.datetime()?;
    let dists = activities.column("distance_m")?.f64()?;
    let lats = activities.column("start_lat")?.f64()?;
    let lons = activities.column("start_lon")?.f64()?;
    let loc_names = activities.column("location_name")?;

    // Phase 1: Compute CE + HR CV for each run
    struct RunData {
        date: chrono::NaiveDate,
        distance_km: f64,
        start_lat: f64,
        start_lon: f64,
        ce: f64,
        gap_speed: f64,
        avg_hr: f64,
        avg_temp: Option<f64>,
        hr_cv: f64,
        location_name: String,
    }

    let mut all_runs: Vec<RunData> = Vec::new();

    for i in 0..activities.height() {
        let Some(aid) = ids.get(i) else { continue };
        let Some(ts) = times.get(i) else { continue };
        let Some(date) =
            chrono::DateTime::from_timestamp_micros(ts).map(|dt| dt.naive_utc().date())
        else {
            continue;
        };
        let Some(dist) = dists.get(i) else { continue };
        let Some(lat) = lats.get(i) else { continue };
        let Some(lon) = lons.get(i) else { continue };
        let loc = loc_names
            .get(i)
            .ok()
            .map(|v| v.to_string().trim_matches('"').to_string())
            .unwrap_or_default();

        let detail_path = details_dir.join(format!("{}.parquet", aid));
        if !detail_path.exists() {
            continue;
        }

        let Some(rows) = fitness::load_steady_rows(&detail_path) else {
            continue;
        };
        if rows.len() < 30 {
            continue;
        }

        // Compute CE
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

        // HR CV for steady-run filtering
        let hr_mean = avg_hr;
        let hr_var = rows
            .iter()
            .map(|r| (r.heart_rate - hr_mean).powi(2))
            .sum::<f64>()
            / n;
        let hr_cv = hr_var.sqrt() / hr_mean * 100.0;

        all_runs.push(RunData {
            date,
            distance_km: dist / 1000.0,
            start_lat: lat,
            start_lon: lon,
            ce,
            gap_speed: avg_gap,
            avg_hr,
            avg_temp,
            hr_cv,
            location_name: loc,
        });
    }

    if all_runs.is_empty() {
        println!("No runs with GPS + detail data.");
        return Ok(());
    }

    // Phase 2: Cluster into routes
    let mut routes: Vec<Route> = Vec::new();

    for run in &all_runs {
        // Skip interval workouts
        if run.hr_cv > MAX_HR_CV_PCT {
            continue;
        }

        // Find matching route index
        let match_idx = routes.iter().position(|route| {
            haversine(
                run.start_lat,
                run.start_lon,
                route.start_lat,
                route.start_lon,
            ) <= START_RADIUS_M
                && (run.distance_km - route.avg_distance_km).abs() / route.avg_distance_km
                    <= DISTANCE_TOLERANCE
        });

        let routed = RoutedRun {
            date: run.date,
            distance_km: run.distance_km,
            ce: run.ce,
            gap_speed: run.gap_speed,
            avg_hr: run.avg_hr,
            avg_temp: run.avg_temp,
            location_name: run.location_name.clone(),
        };

        if let Some(idx) = match_idx {
            routes[idx].runs.push(routed);
        } else {
            routes.push(Route {
                start_lat: run.start_lat,
                start_lon: run.start_lon,
                avg_distance_km: run.distance_km,
                runs: vec![routed],
            });
        }
    }

    // Filter to routes with enough runs
    routes.retain(|r| r.runs.len() >= MIN_ROUTE_RUNS);

    // Unless --all, only show routes with a run in the last RECENT_DAYS
    let cutoff = chrono::Utc::now().date_naive() - chrono::Duration::days(RECENT_DAYS);
    if !show_all {
        routes.retain(|r| r.runs.iter().any(|run| run.date >= cutoff));
    }

    routes.sort_by(|a, b| b.runs.len().cmp(&a.runs.len()));

    if routes.is_empty() {
        if show_all {
            println!(
                "No routes with {} or more steady runs found.",
                MIN_ROUTE_RUNS
            );
        } else {
            println!(
                "No routes with recent activity (last {} days). Use --all to see all routes.",
                RECENT_DAYS
            );
        }
        return Ok(());
    }

    // Phase 3: Report
    println!("=== Route Fitness Report ===\n");
    if show_all {
        println!(
            "Showing all {} routes with {}+ steady runs.\n",
            routes.len(),
            MIN_ROUTE_RUNS
        );
    } else {
        println!(
            "Showing {} routes with activity in the last {} days. Use --all for all routes.\n",
            routes.len(),
            RECENT_DAYS
        );
    }

    for (idx, route) in routes.iter().enumerate() {
        let runs = &route.runs;

        // Most common location name
        let mut name_counts: HashMap<&str, usize> = HashMap::new();
        for r in runs {
            *name_counts.entry(&r.location_name).or_default() += 1;
        }
        let route_name = name_counts
            .into_iter()
            .max_by_key(|(_, c)| *c)
            .map(|(n, _)| n)
            .unwrap_or("Unknown");

        let avg_dist = runs.iter().map(|r| r.distance_km).sum::<f64>() / runs.len() as f64;
        let first_date = runs.iter().map(|r| r.date).min().unwrap();
        let last_date = runs.iter().map(|r| r.date).max().unwrap();

        println!(
            "Route #{}: {:.1} km — \"{}\"",
            idx + 1,
            avg_dist,
            route_name,
        );
        println!(
            "  {} steady runs, {} to {}",
            runs.len(),
            first_date,
            last_date,
        );

        // Current vs historical CE
        let all_ce: Vec<f64> = runs.iter().map(|r| r.ce).collect();
        let mean_ce = all_ce.iter().sum::<f64>() / all_ce.len() as f64;

        let recent_n = 5.min(runs.len());
        let recent = &runs[runs.len() - recent_n..];
        let recent_ce = recent.iter().map(|r| r.ce).sum::<f64>() / recent.len() as f64;

        // Percentile
        let mut sorted = all_ce.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let below = sorted.iter().filter(|&&v| v < recent_ce).count();
        let pct = (below as f64 / sorted.len() as f64 * 100.0).round() as u32;

        // Best ever
        let best_ce = all_ce.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let trend_indicator = if recent_ce >= mean_ce {
            "\x1b[32m▲\x1b[0m"
        } else if recent_ce >= mean_ce - 0.0015 {
            "\x1b[33m-\x1b[0m"
        } else {
            "\x1b[31m▼\x1b[0m"
        };

        println!(
            "  Current CE (last {}): {:.4} {} ({}th pct)  |  avg: {:.4}  best: {:.4}",
            recent_n, recent_ce, trend_indicator, pct, mean_ce, best_ce,
        );

        // Yearly trend for this route
        let mut by_year: HashMap<i32, Vec<f64>> = HashMap::new();
        for r in runs {
            by_year.entry(r.date.year()).or_default().push(r.ce);
        }
        let mut years: Vec<i32> = by_year.keys().cloned().collect();
        years.sort();

        if years.len() > 1 {
            let year_strs: Vec<String> = years
                .iter()
                .map(|y| {
                    let vals = &by_year[y];
                    let avg = vals.iter().sum::<f64>() / vals.len() as f64;
                    format!("{}:{:.4}({})", y, avg, vals.len())
                })
                .collect();
            println!("  By year: {}", year_strs.join("  "));
        }

        // Last 5 runs on this route
        println!();
        for r in recent {
            let temp_s = r
                .avg_temp
                .map(|t| format!("{:.0}°C", t))
                .unwrap_or_else(|| "---".into());
            let indicator = if r.ce >= mean_ce {
                "\x1b[32m▲\x1b[0m"
            } else if r.ce >= mean_ce - 0.0015 {
                "\x1b[33m-\x1b[0m"
            } else {
                "\x1b[31m▼\x1b[0m"
            };
            println!(
                "  {} {} CE:{:.4} GAP:{} HR:{:.0} {}",
                indicator,
                r.date,
                r.ce,
                fitness::format_pace(r.gap_speed),
                r.avg_hr,
                temp_s,
            );
        }

        println!();
    }

    Ok(())
}
