//! Direct fetcher for Garmin health and performance data.
//! Fetches day by day and writes monthly-partitioned parquet files
//! compatible with the existing storage layout.

use chrono::{Datelike, Duration, NaiveDate, Utc};
use polars::prelude::*;
use std::collections::HashSet;

use garmin_connect::Client;

use crate::config::Config;
use crate::error::{AppError, Result};

pub async fn fetch_all(config: &Config, from: NaiveDate, to: Option<NaiveDate>) -> Result<()> {
    println!("Authenticating...");
    let (_oauth1, oauth2) = crate::sync::authenticate().await?;
    let client = Client::new(oauth2);

    // Get display name (needed for some endpoints)
    let display_name = get_display_name(&client).await?;
    println!("Authenticated as {display_name}.");

    let to = to.unwrap_or_else(|| Utc::now().date_naive());
    let total_days = (to - from).num_days() + 1;

    // Load existing dates so we can skip them
    let existing_health = load_existing_dates(config, "daily_health")?;
    let existing_perf = load_existing_dates(config, "performance_metrics")?;
    let skip_count = existing_health.len().min(existing_perf.len());
    println!(
        "Fetching {} to {} ({} days, ~{} already synced)",
        from, to, total_days, skip_count
    );

    // Accumulate records per month, flush when the month changes
    let mut health_buf: Vec<serde_json::Value> = Vec::new();
    let mut perf_buf: Vec<serde_json::Value> = Vec::new();
    let mut current_month: Option<(i32, u32)> = None; // (year, month)

    let mut date = from;
    let mut fetched = 0i64;
    let mut skipped = 0i64;
    let mut errors = 0i64;

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

        if has_health && has_perf {
            date += Duration::days(1);
            skipped += 1;
            continue;
        }

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
        if fetched % 30 == 0 || date == to {
            println!(
                "  {}/{} days ({}%), skipped: {}, errors: {}",
                fetched + skipped,
                total_days,
                ((fetched + skipped) * 100) / total_days,
                skipped,
                errors
            );
        }

        // Rate limit — 500ms between days
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
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
    Ok(())
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
        write_parquet_partition(config, "daily_health", &partition, df)?;
    }
    if !perf_records.is_empty() {
        let df = perf_records_to_df(perf_records)?;
        write_parquet_partition(config, "performance_metrics", &partition, df)?;
    }
    Ok(())
}

/// Write (or merge with) a monthly parquet partition file.
fn write_parquet_partition(
    config: &Config,
    entity_dir: &str,
    partition: &str,
    new_data: DataFrame,
) -> Result<()> {
    let dir = config.garmin_storage_path.join(entity_dir);
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{}.parquet", partition));
    write_or_merge_parquet(&path, new_data)
}

/// Write a DataFrame to a parquet file, merging with existing data if present.
/// Handles schema mismatches by selecting only columns from the new data.
/// Deduplicates by "date" column, keeping the latest value.
fn write_or_merge_parquet(path: &std::path::Path, new_data: DataFrame) -> Result<()> {
    // If file exists, merge (deduplicate by date)
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
            .unique(Some(vec!["date".into()]), UniqueKeepStrategy::Last)
            .sort(["date"], Default::default())
            .collect()?
    } else {
        new_data
    };

    let mut file = std::fs::File::create(path)?;
    ParquetWriter::new(&mut file)
        .with_compression(ParquetCompression::Zstd(None))
        .finish(&mut merged.clone())?;

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
        "avg_stress": ji32(&health, "averageStressLevel"),
        "max_stress": ji32(&health, "maxStressLevel"),
        "body_battery_start": ji32(&health, "bodyBatteryChargedValue"),
        "body_battery_end": ji32(&health, "bodyBatteryDrainedValue"),
        "hrv_weekly_avg": hw,
        "hrv_last_night": hn,
        "hrv_status": hs,
        "avg_respiration": health.get("averageRespirationValue").and_then(|v| v.as_f64()),
        "avg_spo2": ji32(&health, "averageSpo2Value"),
        "lowest_spo2": ji32(&health, "lowestSpo2Value"),
        "hydration_ml": ji32(&health, "hydrationIntakeGoal"),
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
    let fitness_age: Option<i32> = match client.get_json::<serde_json::Value>(&fa_path).await {
        Ok(data) => data
            .get("fitnessAge")
            .and_then(|v| v.as_i64())
            .map(|v| v as i32),
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

// --- Helpers ---

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
    if let Some(sd) = data
        .get("mostRecentTrainingStatus")
        .and_then(|s| s.get("latestTrainingStatusData"))
        .and_then(|d| d.get(date_str))
    {
        return sd
            .get("trainingStatusFeedbackPhrase")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
    }
    if let Some(history) = data.get("trainingStatusHistory").and_then(|h| h.as_array()) {
        for entry in history {
            if entry.get("calendarDate").and_then(|d| d.as_str()) == Some(date_str) {
                return entry
                    .get("trainingStatusFeedbackPhrase")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
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
        write_or_merge_parquet(&path, new).unwrap();

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
        write_or_merge_parquet(&path, first).unwrap();

        // Write overlapping data — 01-02 should be updated, 01-03 added
        let second = make_health_df(&["2024-01-02", "2024-01-03"], &[Some(7000), Some(8000)]);
        write_or_merge_parquet(&path, second).unwrap();

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
        write_or_merge_parquet(&path, new).unwrap();

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
}
