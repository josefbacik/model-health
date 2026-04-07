//! Direct fetcher for Garmin health and performance data.
//! Fetches day by day and writes monthly-partitioned parquet files
//! compatible with the existing storage layout.

use chrono::{Datelike, Duration, NaiveDate, Utc};
use polars::prelude::*;
use std::collections::HashSet;
use std::time::Instant;

use garmin_connect::Client;

use crate::config::Config;
use crate::error::{AppError, Result};

pub async fn fetch_all(
    config: &Config,
    from: NaiveDate,
    to: Option<NaiveDate>,
    force: bool,
) -> Result<()> {
    println!("Authenticating...");
    let (_oauth1, oauth2) = crate::sync::authenticate().await?;
    let client = Client::new(oauth2);

    // Get display name (needed for some endpoints)
    let display_name = get_display_name(&client).await?;
    println!("Authenticated as {display_name}.");

    let to = to.unwrap_or_else(|| Utc::now().date_naive());
    let total_days = (to - from).num_days() + 1;

    // Load existing dates so we can skip them (unless --force re-fetches all)
    let existing_health = if force {
        HashSet::new()
    } else {
        load_existing_dates(config, "daily_health")?
    };
    let existing_perf = if force {
        HashSet::new()
    } else {
        load_existing_dates(config, "performance_metrics")?
    };
    let skip_count = existing_health.len().min(existing_perf.len());
    println!(
        "Fetching {} to {} ({} days, ~{} already synced{})",
        from,
        to,
        total_days,
        skip_count,
        if force { ", force re-fetch" } else { "" }
    );

    // Accumulate records per month, flush when the month changes
    let mut health_buf: Vec<serde_json::Value> = Vec::new();
    let mut perf_buf: Vec<serde_json::Value> = Vec::new();
    let mut current_month: Option<(i32, u32)> = None; // (year, month)

    let mut date = from;
    let mut fetched = 0i64;
    let mut skipped = 0i64;
    let mut errors = 0i64;
    let started = Instant::now();

    while date <= to {
        let ym = (date.year(), date.month());

        // Flush buffers when month changes
        if let Some(prev) = current_month
            && prev != ym
        {
            flush_month(config, prev, &health_buf, &perf_buf)?;
            health_buf.clear();
            perf_buf.clear();
        }
        current_month = Some(ym);

        let has_health = existing_health.contains(&date);
        let has_perf = existing_perf.contains(&date);
        let was_skipped = has_health && has_perf;

        if was_skipped {
            skipped += 1;
        } else {
            // Fetch health
            if !has_health {
                match fetch_daily_health(&client, &display_name, date).await {
                    Ok(record) => health_buf.push(record),
                    Err(e) => {
                        eprintln!("  Health {}: {}", date, e);
                        errors += 1;
                    }
                }
            }

            // Fetch performance
            if !has_perf {
                match fetch_performance(&client, &display_name, date).await {
                    Ok(record) => perf_buf.push(record),
                    Err(e) => {
                        eprintln!("  Perf {}: {}", date, e);
                        errors += 1;
                    }
                }
            }

            fetched += 1;
        }

        let processed = fetched + skipped;
        if processed % 30 == 0 || date == to {
            let (elapsed, eta) = elapsed_and_eta(started, processed, total_days);
            println!(
                "  {}/{} days ({}%), fetched: {}, skipped: {}, errors: {}  [{} elapsed, {} ETA]",
                processed,
                total_days,
                (processed * 100) / total_days,
                fetched,
                skipped,
                errors,
                elapsed,
                eta,
            );
        }

        // Rate limit — 500ms between API hits, none on cache hits
        if !was_skipped {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
        date += Duration::days(1);
    }

    // Flush remaining
    if let Some(ym) = current_month
        && (!health_buf.is_empty() || !perf_buf.is_empty())
    {
        flush_month(config, ym, &health_buf, &perf_buf)?;
    }

    println!(
        "Done. {} fetched, {} skipped, {} errors.",
        fetched, skipped, errors
    );

    // Weight & blood pressure fetch a date *range* per call, so they use a
    // different (month-based) incremental strategy than the per-day loop above.
    fetch_weight_and_bp(config, &client, from, to, force).await?;

    Ok(())
}

/// Fetch weight and blood pressure for the requested range, one month at a
/// time. Past months whose partition file already exists are skipped; the
/// current month is always re-fetched to pick up newly added measurements.
async fn fetch_weight_and_bp(
    config: &Config,
    client: &Client,
    from: NaiveDate,
    to: NaiveDate,
    force: bool,
) -> Result<()> {
    let today = Utc::now().date_naive();
    let existing_weight = if force {
        HashSet::new()
    } else {
        load_existing_months(config, "weight")?
    };
    let existing_bp = if force {
        HashSet::new()
    } else {
        load_existing_months(config, "blood_pressure")?
    };

    let months = iter_months(from, to);
    let total_months = months.len();
    println!(
        "Fetching weight & blood pressure ({} months)...",
        total_months
    );

    let mut total_weight_records = 0usize;
    let mut total_bp_records = 0usize;
    let mut weight_errors = 0usize;
    let mut bp_errors = 0usize;
    let mut processed = 0usize;
    let mut skipped_months = 0usize;
    let started = Instant::now();

    for (year, month) in months {
        let is_current = today.year() == year && today.month() == month;
        let (m_start, m_end) = month_bounds(year, month);
        let m_start = m_start.max(from);
        let m_end = m_end.min(to);
        let partition = format!("{:04}-{:02}", year, month);

        // Always re-fetch the current month so newly added measurements show
        // up. Past months are skipped if a partition file already exists.
        let fetch_weight = is_current || !existing_weight.contains(&(year, month));
        let fetch_bp = is_current || !existing_bp.contains(&(year, month));
        if !fetch_weight && !fetch_bp {
            skipped_months += 1;
        }

        // Weight
        if fetch_weight {
            match fetch_weight_range(client, m_start, m_end).await {
                Ok(records) if !records.is_empty() => {
                    total_weight_records += records.len();
                    let df = weight_records_to_df(&records)?;
                    // Dedup on (date, sample_pk, timestamp_gmt). sample_pk is
                    // the canonical unique ID for each measurement, but it
                    // can be null in fallback paths; timestamp_gmt is
                    // always non-null (synthesized from the epoch-ms `date`
                    // field) so the composite key is always discriminating.
                    write_parquet_partition(
                        config,
                        "weight",
                        &partition,
                        df,
                        &["date", "sample_pk", "timestamp_gmt"],
                    )?;
                }
                Ok(_) => {}
                Err(e) => {
                    eprintln!("  Weight {}: {}", partition, e);
                    weight_errors += 1;
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }

        // Blood pressure
        if fetch_bp {
            match fetch_bp_range(client, m_start, m_end).await {
                Ok(records) if !records.is_empty() => {
                    total_bp_records += records.len();
                    let df = bp_records_to_df(&records)?;
                    write_parquet_partition(
                        config,
                        "blood_pressure",
                        &partition,
                        df,
                        &["date", "timestamp_gmt"],
                    )?;
                }
                Ok(_) => {}
                Err(e) => {
                    eprintln!("  BP {}: {}", partition, e);
                    bp_errors += 1;
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }

        processed += 1;
        let (elapsed, eta) = elapsed_and_eta(started, processed as i64, total_months as i64);
        println!(
            "  {} {}/{} months, skipped: {}, weight total: {} rec, BP total: {} rec  [{} elapsed, {} ETA]",
            partition,
            processed,
            total_months,
            skipped_months,
            total_weight_records,
            total_bp_records,
            elapsed,
            eta,
        );
    }

    println!(
        "Weight: {} records, {} errors. BP: {} records, {} errors.",
        total_weight_records, weight_errors, total_bp_records, bp_errors
    );
    Ok(())
}

/// Iterate (year, month) tuples covering the inclusive [from, to] range.
/// Returns an empty vec if `from > to`.
fn iter_months(from: NaiveDate, to: NaiveDate) -> Vec<(i32, u32)> {
    if from > to {
        return Vec::new();
    }
    let mut out = Vec::new();
    let (mut y, mut m) = (from.year(), from.month());
    let end = (to.year(), to.month());
    loop {
        out.push((y, m));
        if (y, m) == end {
            break;
        }
        if m == 12 {
            y += 1;
            m = 1;
        } else {
            m += 1;
        }
    }
    out
}

/// First and last day of a given month.
fn month_bounds(year: i32, month: u32) -> (NaiveDate, NaiveDate) {
    let first = NaiveDate::from_ymd_opt(year, month, 1).expect("valid month");
    let (ny, nm) = if month == 12 {
        (year + 1, 1)
    } else {
        (year, month + 1)
    };
    let next_first = NaiveDate::from_ymd_opt(ny, nm, 1).expect("valid month");
    let last = next_first - Duration::days(1);
    (first, last)
}

/// Look at the entity's parquet directory and return the set of months
/// (year, month) that already have a partition file. Returns an empty set
/// if the directory doesn't exist (first run); other I/O errors are logged
/// at warn level and treated as "no cache" so the fetch can proceed.
fn load_existing_months(config: &Config, entity_dir: &str) -> Result<HashSet<(i32, u32)>> {
    let dir = config.garmin_storage_path.join(entity_dir);
    let mut set = HashSet::new();
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(set),
        Err(e) => {
            tracing::warn!(
                "load_existing_months: cannot read {}: {} — proceeding as if empty",
                dir.display(),
                e
            );
            return Ok(set);
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("parquet") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        // Expect "YYYY-MM"
        if stem.len() == 7
            && stem.as_bytes()[4] == b'-'
            && let (Ok(y), Ok(m)) = (stem[..4].parse::<i32>(), stem[5..].parse::<u32>())
        {
            set.insert((y, m));
        }
    }
    Ok(set)
}

/// Load the set of dates already present in parquet files for a given entity.
fn load_existing_dates(config: &Config, entity_dir: &str) -> Result<HashSet<NaiveDate>> {
    let pattern = config
        .garmin_storage_path
        .join(entity_dir)
        .join("*.parquet");
    let pattern_str = pattern.to_string_lossy().to_string();

    let lf = match LazyFrame::scan_parquet(&pattern_str, Default::default()) {
        Ok(lf) => lf,
        Err(_) => return Ok(HashSet::new()), // No files yet
    };

    let df = lf.select([col("date")]).collect()?;
    let dates = df.column("date")?;

    let mut set = HashSet::new();
    for i in 0..dates.len() {
        if let Ok(AnyValue::Date(days)) = dates.get(i)
            && let Some(d) = NaiveDate::from_num_days_from_ce_opt(days + 719_163)
        {
            set.insert(d);
        }
    }
    Ok(set)
}

/// Flush accumulated records to monthly-partitioned parquet files.
fn flush_month(
    config: &Config,
    (year, month): (i32, u32),
    health_records: &[serde_json::Value],
    perf_records: &[serde_json::Value],
) -> Result<()> {
    let partition = format!("{:04}-{:02}", year, month);

    if !health_records.is_empty() {
        let df = health_records_to_df(health_records)?;
        write_parquet_partition(config, "daily_health", &partition, df, &["date"])?;
    }
    if !perf_records.is_empty() {
        let df = perf_records_to_df(perf_records)?;
        write_parquet_partition(config, "performance_metrics", &partition, df, &["date"])?;
    }
    Ok(())
}

/// Write (or merge with) a monthly parquet partition file.
fn write_parquet_partition(
    config: &Config,
    entity_dir: &str,
    partition: &str,
    new_data: DataFrame,
    dedup_keys: &[&str],
) -> Result<()> {
    let dir = config.garmin_storage_path.join(entity_dir);
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{}.parquet", partition));
    write_or_merge_parquet(&path, new_data, dedup_keys)
}

/// Write a DataFrame to a parquet file, merging with existing data if present.
/// Handles schema mismatches by selecting only columns from the new data.
/// Deduplicates by `dedup_keys`, keeping the latest value, and sorts by all keys
/// (in order) so the output is deterministic across runs.
///
/// Writes go to a `.tmp` sibling file that is then atomically renamed into
/// place. This makes a torn write recoverable: a crashed run leaves a `.tmp`
/// file that the next run will overwrite, and the existing partition file is
/// never partial.
fn write_or_merge_parquet(
    path: &std::path::Path,
    new_data: DataFrame,
    dedup_keys: &[&str],
) -> Result<()> {
    assert!(!dedup_keys.is_empty(), "dedup_keys must not be empty");
    let dedup_cols: Vec<String> = dedup_keys.iter().map(|k| (*k).to_string()).collect();
    let sort_cols: Vec<String> = dedup_cols.clone();

    // If file exists, merge (deduplicate by keys)
    let merged = if path.exists() {
        let existing =
            LazyFrame::scan_parquet(path.to_string_lossy().as_ref(), Default::default())?
                .collect()?;
        // Select only the columns present in new_data to handle schema differences
        // (e.g., garmin-cli wrote extra columns like id, profile_id, raw_json)
        let our_cols: Vec<String> = new_data
            .get_column_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let existing_aligned = existing
            .lazy()
            .select(our_cols.iter().map(|c| col(c.as_str())).collect::<Vec<_>>())
            .collect()?;
        let combined = existing_aligned.vstack(&new_data)?;
        combined
            .lazy()
            .unique(Some(dedup_cols.clone()), UniqueKeepStrategy::Last)
            .sort(sort_cols.clone(), Default::default())
            .collect()?
    } else {
        new_data
            .lazy()
            .unique(Some(dedup_cols.clone()), UniqueKeepStrategy::Last)
            .sort(sort_cols.clone(), Default::default())
            .collect()?
    };

    // Atomic write: stage to a sibling .tmp file then rename.
    let tmp_path = path.with_extension("parquet.tmp");
    {
        let mut file = std::fs::File::create(&tmp_path)?;
        ParquetWriter::new(&mut file)
            .with_compression(ParquetCompression::Zstd(None))
            .finish(&mut merged.clone())?;
    }
    std::fs::rename(&tmp_path, path)?;

    Ok(())
}

fn health_records_to_df(records: &[serde_json::Value]) -> Result<DataFrame> {
    let mut dates = Vec::new();
    let mut steps = Vec::new();
    let mut step_goal = Vec::new();
    let mut total_cal = Vec::new();
    let mut active_cal = Vec::new();
    let mut bmr_cal = Vec::new();
    let mut resting_hr = Vec::new();
    let mut sleep_sec = Vec::new();
    let mut deep_sleep = Vec::new();
    let mut light_sleep = Vec::new();
    let mut rem_sleep = Vec::new();
    let mut sleep_score = Vec::new();
    let mut avg_stress = Vec::new();
    let mut max_stress = Vec::new();
    let mut bb_start = Vec::new();
    let mut bb_end = Vec::new();
    let mut hrv_weekly = Vec::new();
    let mut hrv_night = Vec::new();
    let mut hrv_status = Vec::new();
    let mut avg_resp = Vec::new();
    let mut avg_spo2 = Vec::new();
    let mut lowest_spo2 = Vec::new();
    let mut hydration = Vec::new();
    let mut mod_min = Vec::new();
    let mut vig_min = Vec::new();

    for r in records {
        dates.push(r["_date"].as_str().unwrap_or("").to_string());
        steps.push(oi32(r, "steps"));
        step_goal.push(oi32(r, "step_goal"));
        total_cal.push(oi32(r, "total_calories"));
        active_cal.push(oi32(r, "active_calories"));
        bmr_cal.push(oi32(r, "bmr_calories"));
        resting_hr.push(oi32(r, "resting_hr"));
        sleep_sec.push(oi32(r, "sleep_seconds"));
        deep_sleep.push(oi32(r, "deep_sleep_seconds"));
        light_sleep.push(oi32(r, "light_sleep_seconds"));
        rem_sleep.push(oi32(r, "rem_sleep_seconds"));
        sleep_score.push(oi32(r, "sleep_score"));
        avg_stress.push(oi32(r, "avg_stress"));
        max_stress.push(oi32(r, "max_stress"));
        bb_start.push(oi32(r, "body_battery_start"));
        bb_end.push(oi32(r, "body_battery_end"));
        hrv_weekly.push(oi32(r, "hrv_weekly_avg"));
        hrv_night.push(oi32(r, "hrv_last_night"));
        hrv_status.push(
            r.get("hrv_status")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        );
        avg_resp.push(r.get("avg_respiration").and_then(|v| v.as_f64()));
        avg_spo2.push(oi32(r, "avg_spo2"));
        lowest_spo2.push(oi32(r, "lowest_spo2"));
        hydration.push(oi32(r, "hydration_ml"));
        mod_min.push(oi32(r, "moderate_intensity_min"));
        vig_min.push(oi32(r, "vigorous_intensity_min"));
    }

    let date_series: Vec<Option<NaiveDate>> = dates
        .iter()
        .map(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
        .collect();

    let df = df!(
        "date" => &date_series,
        "steps" => &steps,
        "step_goal" => &step_goal,
        "total_calories" => &total_cal,
        "active_calories" => &active_cal,
        "bmr_calories" => &bmr_cal,
        "resting_hr" => &resting_hr,
        "sleep_seconds" => &sleep_sec,
        "deep_sleep_seconds" => &deep_sleep,
        "light_sleep_seconds" => &light_sleep,
        "rem_sleep_seconds" => &rem_sleep,
        "sleep_score" => &sleep_score,
        "avg_stress" => &avg_stress,
        "max_stress" => &max_stress,
        "body_battery_start" => &bb_start,
        "body_battery_end" => &bb_end,
        "hrv_weekly_avg" => &hrv_weekly,
        "hrv_last_night" => &hrv_night,
        "hrv_status" => &hrv_status,
        "avg_respiration" => &avg_resp,
        "avg_spo2" => &avg_spo2,
        "lowest_spo2" => &lowest_spo2,
        "hydration_ml" => &hydration,
        "moderate_intensity_min" => &mod_min,
        "vigorous_intensity_min" => &vig_min,
    )?;

    Ok(df)
}

fn perf_records_to_df(records: &[serde_json::Value]) -> Result<DataFrame> {
    let mut dates = Vec::new();
    let mut vo2max = Vec::new();
    let mut fitness_age = Vec::new();
    let mut training_readiness = Vec::new();
    let mut training_status = Vec::new();
    let mut race_5k = Vec::new();
    let mut race_10k = Vec::new();
    let mut race_half = Vec::new();
    let mut race_marathon = Vec::new();
    let mut endurance = Vec::new();
    let mut hill = Vec::new();

    for r in records {
        dates.push(r["_date"].as_str().unwrap_or("").to_string());
        vo2max.push(r.get("vo2max").and_then(|v| v.as_f64()));
        fitness_age.push(oi32(r, "fitness_age"));
        training_readiness.push(oi32(r, "training_readiness"));
        training_status.push(
            r.get("training_status")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        );
        race_5k.push(oi32(r, "race_5k_sec"));
        race_10k.push(oi32(r, "race_10k_sec"));
        race_half.push(oi32(r, "race_half_sec"));
        race_marathon.push(oi32(r, "race_marathon_sec"));
        endurance.push(oi32(r, "endurance_score"));
        hill.push(oi32(r, "hill_score"));
    }

    let date_series: Vec<Option<NaiveDate>> = dates
        .iter()
        .map(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
        .collect();

    let df = df!(
        "date" => &date_series,
        "vo2max" => &vo2max,
        "fitness_age" => &fitness_age,
        "training_readiness" => &training_readiness,
        "training_status" => &training_status,
        "race_5k_sec" => &race_5k,
        "race_10k_sec" => &race_10k,
        "race_half_sec" => &race_half,
        "race_marathon_sec" => &race_marathon,
        "endurance_score" => &endurance,
        "hill_score" => &hill,
    )?;

    Ok(df)
}

/// Hit each endpoint we use for one date and pretty-print the raw JSON. Used
/// to verify field names against what Garmin actually returns for this account.
pub async fn probe(date: NaiveDate) -> Result<()> {
    println!("Authenticating...");
    let (_oauth1, oauth2) = crate::sync::authenticate().await?;
    let client = Client::new(oauth2);
    let display_name = get_display_name(&client).await?;
    println!("Authenticated as {}.\n", display_name);

    let endpoints: Vec<(&str, String)> = vec![
        (
            "daily summary",
            format!(
                "/usersummary-service/usersummary/daily/{}?calendarDate={}",
                display_name, date
            ),
        ),
        (
            "sleep",
            format!(
                "/wellness-service/wellness/dailySleepData/{}?date={}",
                display_name, date
            ),
        ),
        ("hrv", format!("/hrv-service/hrv/{}", date)),
        (
            "vo2max (maxmet)",
            format!("/metrics-service/metrics/maxmet/daily/{}/{}", date, date),
        ),
        (
            "training readiness",
            format!("/metrics-service/metrics/trainingreadiness/{}", date),
        ),
        (
            "training status",
            format!(
                "/metrics-service/metrics/trainingstatus/aggregated/{}",
                date
            ),
        ),
        (
            "fitness age",
            format!("/fitnessage-service/fitnessage/{}", date),
        ),
        (
            "race predictions",
            format!(
                "/metrics-service/metrics/racepredictions/daily/{}?fromCalendarDate={}&toCalendarDate={}",
                display_name, date, date
            ),
        ),
        (
            "endurance score",
            format!(
                "/metrics-service/metrics/endurancescore?calendarDate={}",
                date
            ),
        ),
        (
            "hill score",
            format!("/metrics-service/metrics/hillscore?calendarDate={}", date),
        ),
    ];

    for (label, path) in endpoints {
        println!("==== {} ====", label);
        println!("GET {}", path);
        match client.get_json::<serde_json::Value>(&path).await {
            Ok(v) => {
                // Print top-level keys with type+sample for object responses,
                // or the array length + first element keys for array responses.
                summarize_json(&v);
                // Also pretty-print the full body for ground truth.
                println!(
                    "FULL: {}",
                    serde_json::to_string_pretty(&v).unwrap_or_default()
                );
            }
            Err(e) => println!("ERROR: {}", e),
        }
        println!();
    }

    Ok(())
}

fn summarize_json(v: &serde_json::Value) {
    match v {
        serde_json::Value::Object(map) => {
            println!("KEYS ({}): ", map.len());
            for (k, vv) in map {
                let kind = match vv {
                    serde_json::Value::Null => "null".to_string(),
                    serde_json::Value::Bool(b) => format!("bool={}", b),
                    serde_json::Value::Number(n) => format!("num={}", n),
                    serde_json::Value::String(s) => {
                        // Truncate at char boundaries to avoid panicking on
                        // multi-byte UTF-8 inputs.
                        let truncated: String = s.chars().take(60).collect();
                        format!("str=\"{}\"", truncated)
                    }
                    serde_json::Value::Array(a) => format!("array[{}]", a.len()),
                    serde_json::Value::Object(o) => format!("object{{{}}}", o.len()),
                };
                println!("  {} = {}", k, kind);
            }
        }
        serde_json::Value::Array(arr) => {
            println!("ARRAY len={}", arr.len());
            if let Some(first) = arr.first() {
                println!("  first element:");
                summarize_json(first);
            }
        }
        _ => println!("SCALAR: {:?}", v),
    }
}

// --- API fetch functions ---

async fn get_display_name(client: &Client) -> Result<String> {
    let profile: serde_json::Value = client
        .get_json("/userprofile-service/socialProfile")
        .await
        .map_err(|e| AppError::Sync(format!("Failed to get profile: {}", e)))?;

    profile
        .get("userName")
        .or_else(|| profile.get("displayName"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| AppError::Sync("Could not find display name in profile".to_string()))
}

async fn fetch_daily_health(
    client: &Client,
    display_name: &str,
    date: NaiveDate,
) -> Result<serde_json::Value> {
    let path = format!(
        "/usersummary-service/usersummary/daily/{}?calendarDate={}",
        display_name, date
    );
    let health: serde_json::Value = match client.get_json(&path).await {
        Ok(data) => data,
        Err(garmin_connect::Error::NotFound(_)) | Err(garmin_connect::Error::Api { .. }) => {
            serde_json::json!({})
        }
        Err(e) => return Err(AppError::Sync(e.to_string())),
    };

    // Sleep data
    let sleep_path = format!(
        "/wellness-service/wellness/dailySleepData/{}?date={}",
        display_name, date
    );
    let sleep: Option<serde_json::Value> = client.get_json(&sleep_path).await.ok();

    // HRV data
    let hrv_path = format!("/hrv-service/hrv/{}", date);
    let hrv: Option<serde_json::Value> = client.get_json(&hrv_path).await.ok();

    let (sleep_total, deep, light, rem, score) = parse_sleep(sleep.as_ref());
    let (hw, hn, hs) = parse_hrv(hrv.as_ref());

    Ok(serde_json::json!({
        "_date": date.to_string(),
        "steps": ji32(&health, "totalSteps"),
        "step_goal": ji32(&health, "dailyStepGoal"),
        "total_calories": ji32(&health, "totalKilocalories"),
        "active_calories": ji32(&health, "activeKilocalories"),
        "bmr_calories": ji32(&health, "bmrKilocalories"),
        "resting_hr": ji32(&health, "restingHeartRate"),
        "sleep_seconds": sleep_total.or_else(|| ji32(&health, "sleepingSeconds")),
        "deep_sleep_seconds": deep,
        "light_sleep_seconds": light,
        "rem_sleep_seconds": rem,
        "sleep_score": score,
        // Garmin uses -1 (and -2) as "no measurement" sentinels for stress.
        "avg_stress": ji32(&health, "averageStressLevel").filter(|&v| v >= 0),
        "max_stress": ji32(&health, "maxStressLevel").filter(|&v| v >= 0),
        // body_battery_start = value at wake (true start of day).
        // body_battery_end   = most recent reading at end of day.
        // (Previously this was incorrectly using `bodyBatteryChargedValue` /
        // `bodyBatteryDrainedValue`, which are *deltas*, not absolute values.)
        // We deliberately do NOT fall back to bodyBatteryHighestValue: the
        // day's peak isn't necessarily the wake value (could be later in the
        // day) and downstream features that compute drain would be wrong.
        "body_battery_start": ji32(&health, "bodyBatteryAtWakeTime"),
        "body_battery_end": ji32(&health, "bodyBatteryMostRecentValue"),
        "hrv_weekly_avg": hw,
        "hrv_last_night": hn,
        "hrv_status": hs,
        // Verified field names against /usersummary-service response shape.
        "avg_respiration": health.get("avgWakingRespirationValue").and_then(|v| v.as_f64()),
        "avg_spo2": ji32(&health, "averageSpo2"),
        "lowest_spo2": ji32(&health, "lowestSpo2"),
        // Hydration lives in a separate endpoint we're not currently calling.
        "hydration_ml": serde_json::Value::Null,
        "moderate_intensity_min": ji32(&health, "moderateIntensityMinutes"),
        "vigorous_intensity_min": ji32(&health, "vigorousIntensityMinutes"),
    }))
}

async fn fetch_performance(
    client: &Client,
    display_name: &str,
    date: NaiveDate,
) -> Result<serde_json::Value> {
    let vo2_path = format!("/metrics-service/metrics/maxmet/daily/{}/{}", date, date);
    let vo2_data: Option<serde_json::Value> = client.get_json(&vo2_path).await.ok();

    let vo2max = vo2_data.as_ref().and_then(|v| {
        first_entry(v)
            .and_then(|e| e.get("generic"))
            .and_then(|g| g.get("vo2MaxValue"))
            .and_then(|v| v.as_f64())
    });

    let fitness_age_from_vo2 = vo2_data.as_ref().and_then(|v| {
        first_entry(v)
            .and_then(|e| e.get("generic"))
            .and_then(|g| g.get("fitnessAge"))
            .and_then(|v| v.as_i64())
            .map(|v| v as i32)
    });

    let fa_path = format!("/fitnessage-service/fitnessage/{}", date);
    // The fitness-age endpoint returns a float (e.g. 39.35); round to nearest int.
    let fitness_age: Option<i32> = match client.get_json::<serde_json::Value>(&fa_path).await {
        Ok(data) => data
            .get("fitnessAge")
            .and_then(|v| v.as_f64())
            .map(|f| f.round() as i32)
            .or(fitness_age_from_vo2),
        Err(_) => fitness_age_from_vo2,
    };

    let race_path = format!(
        "/metrics-service/metrics/racepredictions/daily/{}?fromCalendarDate={}&toCalendarDate={}",
        display_name, date, date
    );
    let race: Option<serde_json::Value> = client.get_json(&race_path).await.ok();
    let re = race.as_ref().and_then(first_entry);

    let tr_path = format!("/metrics-service/metrics/trainingreadiness/{}", date);
    let tr: Option<serde_json::Value> = client.get_json(&tr_path).await.ok();
    let training_readiness = tr
        .as_ref()
        .and_then(first_entry)
        .and_then(|e| e.get("score"))
        .and_then(|v| v.as_i64())
        .map(|v| v as i32);

    let ts_path = format!(
        "/metrics-service/metrics/trainingstatus/aggregated/{}",
        date
    );
    let training_status: Option<String> = client
        .get_json::<serde_json::Value>(&ts_path)
        .await
        .ok()
        .and_then(|data| parse_training_status(&data, &date.to_string()));

    let es_path = format!(
        "/metrics-service/metrics/endurancescore?calendarDate={}",
        date
    );
    let endurance_score: Option<i32> = client
        .get_json::<serde_json::Value>(&es_path)
        .await
        .ok()
        .and_then(|d| {
            first_entry(&d)
                .and_then(|e| e.get("overallScore"))
                .and_then(|v| v.as_i64())
        })
        .map(|v| v as i32);

    let hs_path = format!("/metrics-service/metrics/hillscore?calendarDate={}", date);
    let hill_score: Option<i32> = client
        .get_json::<serde_json::Value>(&hs_path)
        .await
        .ok()
        .and_then(|d| {
            first_entry(&d)
                .and_then(|e| e.get("overallScore"))
                .and_then(|v| v.as_i64())
        })
        .map(|v| v as i32);

    Ok(serde_json::json!({
        "_date": date.to_string(),
        "vo2max": vo2max,
        "fitness_age": fitness_age,
        "training_readiness": training_readiness,
        "training_status": training_status,
        "race_5k_sec": re.and_then(|e| e.get("time5K")).and_then(|v| v.as_i64()).map(|v| v as i32),
        "race_10k_sec": re.and_then(|e| e.get("time10K")).and_then(|v| v.as_i64()).map(|v| v as i32),
        "race_half_sec": re.and_then(|e| e.get("timeHalfMarathon")).and_then(|v| v.as_i64()).map(|v| v as i32),
        "race_marathon_sec": re.and_then(|e| e.get("timeMarathon")).and_then(|v| v.as_i64()).map(|v| v as i32),
        "endurance_score": endurance_score,
        "hill_score": hill_score,
    }))
}

/// Fetch all weight measurements within [from, to]. Returns one record per
/// individual measurement (Garmin can store multiple weigh-ins per day).
async fn fetch_weight_range(
    client: &Client,
    from: NaiveDate,
    to: NaiveDate,
) -> Result<Vec<serde_json::Value>> {
    let path = format!(
        "/weight-service/weight/range/{}/{}?includeAll=true",
        from, to
    );
    let data: serde_json::Value = match client.get_json(&path).await {
        Ok(d) => d,
        Err(garmin_connect::Error::NotFound(_)) => return Ok(Vec::new()),
        Err(e) => return Err(AppError::Sync(e.to_string())),
    };

    let mut out = Vec::new();
    let summaries = data
        .get("dailyWeightSummaries")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    for summary in summaries {
        let summary_date = summary
            .get("summaryDate")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        // Prefer the full list if present; otherwise fall back to latestWeight.
        let metrics: Vec<serde_json::Value> = summary
            .get("allWeightMetrics")
            .and_then(|v| v.as_array())
            .cloned()
            .or_else(|| summary.get("latestWeight").map(|v| vec![v.clone()]))
            .unwrap_or_default();

        for m in metrics {
            // Each measurement carries its own calendarDate; fall back to the
            // summary date if missing.
            let date = m
                .get("calendarDate")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .or_else(|| summary_date.clone())
                .unwrap_or_default();

            // The Garmin weight endpoint puts the per-measurement timestamp in
            // a `date` field as epoch milliseconds (not `timestampGMT`).
            // Convert to an ISO string for storage.
            let ts_gmt = m
                .get("date")
                .and_then(|v| v.as_i64())
                .and_then(chrono::DateTime::<chrono::Utc>::from_timestamp_millis)
                .map(|dt| dt.format("%Y-%m-%dT%H:%M:%S").to_string())
                .or_else(|| {
                    m.get("timestampGMT")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                });

            out.push(serde_json::json!({
                "_date": date,
                "timestamp_gmt": ts_gmt,
                "sample_pk": m.get("samplePk").and_then(|v| v.as_i64()),
                "weight_grams": m.get("weight").and_then(|v| v.as_f64()),
                "bmi": m.get("bmi").and_then(|v| v.as_f64()),
                "body_fat": m.get("bodyFat").and_then(|v| v.as_f64()),
                "body_water": m.get("bodyWater").and_then(|v| v.as_f64()),
                "bone_mass": m.get("boneMass").and_then(|v| v.as_f64()),
                "muscle_mass": m.get("muscleMass").and_then(|v| v.as_f64()),
                "physique_rating": m.get("physiqueRating").and_then(|v| v.as_i64()),
                "visceral_fat": m.get("visceralFat").and_then(|v| v.as_f64()),
                "metabolic_age": m.get("metabolicAge").and_then(|v| v.as_i64()),
                "source_type": m.get("sourceType").and_then(|v| v.as_str()).map(|s| s.to_string()),
            }));
        }
    }

    Ok(out)
}

/// Fetch all blood pressure readings within [from, to].
async fn fetch_bp_range(
    client: &Client,
    from: NaiveDate,
    to: NaiveDate,
) -> Result<Vec<serde_json::Value>> {
    let path = format!(
        "/bloodpressure-service/bloodpressure/range/{}/{}?includeAll=true",
        from, to
    );
    let data: serde_json::Value = match client.get_json(&path).await {
        Ok(d) => d,
        Err(garmin_connect::Error::NotFound(_)) => return Ok(Vec::new()),
        Err(e) => return Err(AppError::Sync(e.to_string())),
    };

    // Two known shapes: { measurementSummaries: [{ measurements: [...] }] }
    // or a flat top-level array of measurements. Handle both.
    let mut out = Vec::new();

    let push_measurement = |out: &mut Vec<serde_json::Value>, m: &serde_json::Value| {
        let ts_local = m
            .get("measurementTimestampLocal")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        // Fall back to local if GMT is missing — `timestamp_gmt` is part of
        // the dedup key, so we need it to be non-null. Manual measurements
        // can be uploaded without timezone metadata.
        let ts_gmt = m
            .get("measurementTimestampGMT")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| ts_local.clone());
        // Derive a calendar date from the local timestamp ("YYYY-MM-DDTHH:MM:SS").
        let date = ts_local
            .as_deref()
            .or(ts_gmt.as_deref())
            .and_then(|s| s.split('T').next())
            .unwrap_or("")
            .to_string();

        out.push(serde_json::json!({
            "_date": date,
            "timestamp_gmt": ts_gmt,
            "timestamp_local": ts_local,
            "systolic": m.get("systolic").and_then(|v| v.as_i64()),
            "diastolic": m.get("diastolic").and_then(|v| v.as_i64()),
            "pulse": m.get("pulse").and_then(|v| v.as_i64()),
            "category": m.get("category").and_then(|v| v.as_i64()),
            "category_name": m.get("categoryName").and_then(|v| v.as_str()).map(|s| s.to_string()),
            "notes": m.get("notes").and_then(|v| v.as_str()).map(|s| s.to_string()),
            "source_type": m.get("sourceType").and_then(|v| v.as_str()).map(|s| s.to_string()),
            "version": m.get("version").and_then(|v| v.as_i64()),
        }));
    };

    if let Some(summaries) = data.get("measurementSummaries").and_then(|v| v.as_array()) {
        for s in summaries {
            if let Some(measurements) = s.get("measurements").and_then(|v| v.as_array()) {
                for m in measurements {
                    push_measurement(&mut out, m);
                }
            }
        }
    } else if let Some(arr) = data.as_array() {
        for m in arr {
            push_measurement(&mut out, m);
        }
    }

    Ok(out)
}

fn weight_records_to_df(records: &[serde_json::Value]) -> Result<DataFrame> {
    let mut dates = Vec::new();
    let mut ts_gmt = Vec::new();
    let mut sample_pk = Vec::new();
    let mut weight_g = Vec::new();
    let mut bmi = Vec::new();
    let mut body_fat = Vec::new();
    let mut body_water = Vec::new();
    let mut bone_mass = Vec::new();
    let mut muscle_mass = Vec::new();
    let mut physique = Vec::new();
    let mut visceral = Vec::new();
    let mut metabolic_age = Vec::new();
    let mut source = Vec::new();

    for r in records {
        dates.push(r["_date"].as_str().unwrap_or("").to_string());
        ts_gmt.push(
            r.get("timestamp_gmt")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        );
        sample_pk.push(r.get("sample_pk").and_then(|v| v.as_i64()));
        weight_g.push(r.get("weight_grams").and_then(|v| v.as_f64()));
        bmi.push(r.get("bmi").and_then(|v| v.as_f64()));
        body_fat.push(r.get("body_fat").and_then(|v| v.as_f64()));
        body_water.push(r.get("body_water").and_then(|v| v.as_f64()));
        bone_mass.push(r.get("bone_mass").and_then(|v| v.as_f64()));
        muscle_mass.push(r.get("muscle_mass").and_then(|v| v.as_f64()));
        physique.push(
            r.get("physique_rating")
                .and_then(|v| v.as_i64())
                .map(|v| v as i32),
        );
        visceral.push(r.get("visceral_fat").and_then(|v| v.as_f64()));
        metabolic_age.push(
            r.get("metabolic_age")
                .and_then(|v| v.as_i64())
                .map(|v| v as i32),
        );
        source.push(
            r.get("source_type")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        );
    }

    // Convert grams to kg for convenience while keeping the raw column too.
    let weight_kg: Vec<Option<f64>> = weight_g.iter().map(|w| w.map(|g| g / 1000.0)).collect();

    let date_series: Vec<Option<NaiveDate>> = dates
        .iter()
        .map(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
        .collect();

    let df = df!(
        "date" => &date_series,
        "timestamp_gmt" => &ts_gmt,
        "sample_pk" => &sample_pk,
        "weight_kg" => &weight_kg,
        "weight_grams" => &weight_g,
        "bmi" => &bmi,
        "body_fat" => &body_fat,
        "body_water" => &body_water,
        "bone_mass" => &bone_mass,
        "muscle_mass" => &muscle_mass,
        "physique_rating" => &physique,
        "visceral_fat" => &visceral,
        "metabolic_age" => &metabolic_age,
        "source_type" => &source,
    )?;

    Ok(df)
}

fn bp_records_to_df(records: &[serde_json::Value]) -> Result<DataFrame> {
    let mut dates = Vec::new();
    let mut ts_gmt = Vec::new();
    let mut ts_local = Vec::new();
    let mut systolic = Vec::new();
    let mut diastolic = Vec::new();
    let mut pulse = Vec::new();
    let mut category = Vec::new();
    let mut category_name = Vec::new();
    let mut notes = Vec::new();
    let mut source = Vec::new();
    let mut version = Vec::new();

    for r in records {
        dates.push(r["_date"].as_str().unwrap_or("").to_string());
        ts_gmt.push(
            r.get("timestamp_gmt")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        );
        ts_local.push(
            r.get("timestamp_local")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        );
        systolic.push(r.get("systolic").and_then(|v| v.as_i64()).map(|v| v as i32));
        diastolic.push(
            r.get("diastolic")
                .and_then(|v| v.as_i64())
                .map(|v| v as i32),
        );
        pulse.push(r.get("pulse").and_then(|v| v.as_i64()).map(|v| v as i32));
        category.push(r.get("category").and_then(|v| v.as_i64()).map(|v| v as i32));
        category_name.push(
            r.get("category_name")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        );
        notes.push(
            r.get("notes")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        );
        source.push(
            r.get("source_type")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        );
        version.push(r.get("version").and_then(|v| v.as_i64()).map(|v| v as i32));
    }

    let date_series: Vec<Option<NaiveDate>> = dates
        .iter()
        .map(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
        .collect();

    let df = df!(
        "date" => &date_series,
        "timestamp_gmt" => &ts_gmt,
        "timestamp_local" => &ts_local,
        "systolic" => &systolic,
        "diastolic" => &diastolic,
        "pulse" => &pulse,
        "category" => &category,
        "category_name" => &category_name,
        "notes" => &notes,
        "source_type" => &source,
        "version" => &version,
    )?;

    Ok(df)
}

// --- Helpers ---

/// Format a number of seconds as "HhMMm", "MmSSs", or "SSs".
fn format_secs(secs: u64) -> String {
    if secs >= 3600 {
        format!("{}h{:02}m", secs / 3600, (secs % 3600) / 60)
    } else if secs >= 60 {
        format!("{}m{:02}s", secs / 60, secs % 60)
    } else {
        format!("{}s", secs)
    }
}

/// Compute (elapsed, eta) strings for a fetch loop. ETA is a linear
/// extrapolation from observed throughput so it adapts to real API latency.
fn elapsed_and_eta(started: Instant, processed: i64, total: i64) -> (String, String) {
    // Sample once so the elapsed and ETA calculations are consistent.
    let elapsed_dur = started.elapsed();
    let elapsed_secs_f = elapsed_dur.as_secs_f64();
    let eta = if processed > 0 && processed < total {
        let per_unit = elapsed_secs_f / processed as f64;
        // Clamp to >=0 to defend against the impossible-but-cheap-to-guard case
        // where `total - processed` is somehow negative (would otherwise wrap
        // when cast to u64).
        let remaining = ((total - processed) as f64 * per_unit).max(0.0);
        format_secs(remaining as u64)
    } else if processed >= total {
        "0s".to_string()
    } else {
        "?".to_string()
    };
    (format_secs(elapsed_dur.as_secs()), eta)
}

fn ji32(v: &serde_json::Value, key: &str) -> Option<i32> {
    v.get(key)
        .and_then(|v| v.as_i64().or_else(|| v.as_f64().map(|f| f as i64)))
        .map(|v| v as i32)
}

fn oi32(v: &serde_json::Value, key: &str) -> Option<i32> {
    v.get(key).and_then(|v| v.as_i64()).map(|v| v as i32)
}

fn first_entry(v: &serde_json::Value) -> Option<&serde_json::Value> {
    if let Some(arr) = v.as_array() {
        arr.first()
    } else {
        Some(v)
    }
}

type SleepFields = (
    Option<i32>,
    Option<i32>,
    Option<i32>,
    Option<i32>,
    Option<i32>,
);

fn parse_sleep(data: Option<&serde_json::Value>) -> SleepFields {
    let data = match data {
        Some(d) => d.get("dailySleepDTO").unwrap_or(d),
        None => return (None, None, None, None, None),
    };
    let deep = ji32(data, "deepSleepSeconds");
    let light = ji32(data, "lightSleepSeconds");
    let rem = ji32(data, "remSleepSeconds");
    let total = ji32(data, "sleepTimeSeconds").or_else(|| match (deep, light, rem) {
        (Some(d), Some(l), Some(r)) => Some(d + l + r),
        _ => None,
    });
    let score = data
        .get("sleepScores")
        .and_then(|s| s.get("overall"))
        .and_then(|o| o.get("value"))
        .and_then(|v| v.as_i64())
        .map(|v| v as i32);
    (total, deep, light, rem, score)
}

fn parse_hrv(data: Option<&serde_json::Value>) -> (Option<i32>, Option<i32>, Option<String>) {
    let summary = match data {
        Some(d) => match d.get("hrvSummary") {
            Some(s) => s,
            None => return (None, None, None),
        },
        None => return (None, None, None),
    };
    let weekly = ji32(summary, "weeklyAvg");
    let night = ji32(summary, "lastNight").or_else(|| ji32(summary, "lastNightAvg"));
    let status = summary
        .get("status")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    (weekly, night, status)
}

fn parse_training_status(data: &serde_json::Value, date_str: &str) -> Option<String> {
    // Primary path: the shape is:
    //   mostRecentTrainingStatus.latestTrainingStatusData.{deviceId}.trainingStatusFeedbackPhrase
    // The inner key is the device ID (e.g. "3497096021"), not the date — so we
    // iterate values and require the calendarDate to match. We do NOT fall
    // back to "any device" because each entry's calendarDate may differ from
    // the requested date and we'd silently mis-attribute it.
    if let Some(by_device) = data
        .get("mostRecentTrainingStatus")
        .and_then(|s| s.get("latestTrainingStatusData"))
        .and_then(|d| d.as_object())
        && let Some(entry) = by_device
            .values()
            .find(|v| v.get("calendarDate").and_then(|d| d.as_str()) == Some(date_str))
        && let Some(phrase) = entry
            .get("trainingStatusFeedbackPhrase")
            .and_then(|v| v.as_str())
    {
        return Some(phrase.to_string());
    }

    // Fallback: older shape with a top-level history array.
    if let Some(history) = data.get("trainingStatusHistory").and_then(|h| h.as_array()) {
        for entry in history {
            if entry.get("calendarDate").and_then(|d| d.as_str()) == Some(date_str)
                && let Some(phrase) = entry
                    .get("trainingStatusFeedbackPhrase")
                    .and_then(|v| v.as_str())
            {
                return Some(phrase.to_string());
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_health_df(dates: &[&str], steps: &[Option<i32>]) -> DataFrame {
        let date_vals: Vec<Option<NaiveDate>> = dates
            .iter()
            .map(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
            .collect();
        df!(
            "date" => &date_vals,
            "steps" => steps,
            "resting_hr" => &vec![Option::<i32>::None; dates.len()],
        )
        .unwrap()
    }

    #[test]
    fn test_merge_with_no_existing_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.parquet");

        let new = make_health_df(&["2024-01-01", "2024-01-02"], &[Some(5000), Some(6000)]);
        write_or_merge_parquet(&path, new, &["date"]).unwrap();

        let result = LazyFrame::scan_parquet(path.to_string_lossy().as_ref(), Default::default())
            .unwrap()
            .collect()
            .unwrap();
        assert_eq!(result.height(), 2);
    }

    #[test]
    fn test_merge_deduplicates_by_date() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.parquet");

        // Write initial data
        let first = make_health_df(&["2024-01-01", "2024-01-02"], &[Some(5000), Some(6000)]);
        write_or_merge_parquet(&path, first, &["date"]).unwrap();

        // Write overlapping data — 01-02 should be updated, 01-03 added
        let second = make_health_df(&["2024-01-02", "2024-01-03"], &[Some(7000), Some(8000)]);
        write_or_merge_parquet(&path, second, &["date"]).unwrap();

        let result = LazyFrame::scan_parquet(path.to_string_lossy().as_ref(), Default::default())
            .unwrap()
            .collect()
            .unwrap();
        assert_eq!(result.height(), 3); // 3 unique dates
        // The 01-02 value should be the newer one (7000)
        let steps = result.column("steps").unwrap();
        // Sorted by date, so row 1 is 01-02
        assert_eq!(steps.get(1).unwrap(), AnyValue::Int32(7000));
    }

    #[test]
    fn test_merge_with_extra_columns_in_existing() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.parquet");

        // Simulate garmin-cli's wider schema with extra columns
        let date_vals: Vec<Option<NaiveDate>> = vec![NaiveDate::from_ymd_opt(2024, 1, 1)];
        let existing = df!(
            "id" => &[Option::<i64>::None],
            "profile_id" => &[1i32],
            "date" => &date_vals,
            "steps" => &[Some(5000i32)],
            "resting_hr" => &[Some(60i32)],
            "raw_json" => &[Some("{}")],
        )
        .unwrap();

        let mut file = std::fs::File::create(&path).unwrap();
        ParquetWriter::new(&mut file)
            .finish(&mut existing.clone())
            .unwrap();

        // Now merge with our narrower schema (no id, profile_id, raw_json)
        let new = make_health_df(&["2024-01-02"], &[Some(6000)]);
        write_or_merge_parquet(&path, new, &["date"]).unwrap();

        // Should succeed and have 2 rows with our columns only
        let result = LazyFrame::scan_parquet(path.to_string_lossy().as_ref(), Default::default())
            .unwrap()
            .collect()
            .unwrap();
        assert_eq!(result.height(), 2);
        assert_eq!(result.width(), 3); // date, steps, resting_hr
        assert!(!result.schema().contains("id"));
        assert!(!result.schema().contains("raw_json"));
    }

    #[test]
    fn test_health_records_to_df() {
        let records = vec![serde_json::json!({
            "_date": "2024-01-15",
            "steps": 8000,
            "resting_hr": 58,
            "sleep_seconds": 28800,
        })];

        let df = health_records_to_df(&records).unwrap();
        assert_eq!(df.height(), 1);
        assert!(df.schema().contains("date"));
        assert!(df.schema().contains("steps"));
        assert!(df.schema().contains("resting_hr"));

        let steps = df.column("steps").unwrap().get(0).unwrap();
        assert_eq!(steps, AnyValue::Int32(8000));
    }

    #[test]
    fn test_perf_records_to_df() {
        let records = vec![serde_json::json!({
            "_date": "2024-01-15",
            "vo2max": 48.5,
            "training_readiness": 72,
        })];

        let df = perf_records_to_df(&records).unwrap();
        assert_eq!(df.height(), 1);
        assert!(df.schema().contains("vo2max"));
        assert!(df.schema().contains("training_readiness"));
    }

    #[test]
    fn test_iter_months_basic() {
        // Same month
        let m = iter_months(
            NaiveDate::from_ymd_opt(2024, 3, 5).unwrap(),
            NaiveDate::from_ymd_opt(2024, 3, 25).unwrap(),
        );
        assert_eq!(m, vec![(2024, 3)]);

        // Across year boundary
        let m = iter_months(
            NaiveDate::from_ymd_opt(2023, 11, 15).unwrap(),
            NaiveDate::from_ymd_opt(2024, 2, 10).unwrap(),
        );
        assert_eq!(m, vec![(2023, 11), (2023, 12), (2024, 1), (2024, 2)]);
    }

    #[test]
    fn test_iter_months_inverted_range_returns_empty() {
        // Regression guard: previous version looped forever when from > to.
        let m = iter_months(
            NaiveDate::from_ymd_opt(2024, 6, 1).unwrap(),
            NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        );
        assert!(m.is_empty());
    }

    #[test]
    fn test_parse_training_status_returns_none_for_wrong_date() {
        // Regression guard: previous version returned an arbitrary device's
        // entry as a fallback, mis-attributing the status to the wrong date.
        let data = serde_json::json!({
            "mostRecentTrainingStatus": {
                "latestTrainingStatusData": {
                    "3497096021": {
                        "calendarDate": "2026-04-05",
                        "trainingStatusFeedbackPhrase": "RECOVERY_2"
                    }
                }
            }
        });
        // Asking for a date the device doesn't have data for: must return None,
        // NOT the 2026-04-05 entry.
        assert_eq!(parse_training_status(&data, "2026-04-06"), None);
        // Same data, asking for the matching date: should succeed.
        assert_eq!(
            parse_training_status(&data, "2026-04-05"),
            Some("RECOVERY_2".to_string())
        );
    }

    #[test]
    fn test_parse_training_status_history_fallback() {
        // The older shape uses a flat trainingStatusHistory array.
        let data = serde_json::json!({
            "trainingStatusHistory": [
                {"calendarDate": "2024-01-15", "trainingStatusFeedbackPhrase": "PRODUCTIVE_1"},
                {"calendarDate": "2024-01-16", "trainingStatusFeedbackPhrase": "MAINTAINING_1"}
            ]
        });
        assert_eq!(
            parse_training_status(&data, "2024-01-16"),
            Some("MAINTAINING_1".to_string())
        );
        assert_eq!(parse_training_status(&data, "2024-01-17"), None);
    }

    #[test]
    fn test_summarize_json_handles_multibyte_strings() {
        // Regression guard: previous version sliced bytes [..60] which panics
        // on multi-byte UTF-8 boundaries.
        let v = serde_json::json!({
            "key": "🚀".repeat(40), // 4-byte chars; well past byte 60
        });
        // Just needs to not panic.
        summarize_json(&v);
    }
}
