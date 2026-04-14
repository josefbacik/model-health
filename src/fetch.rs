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

/// Categories that `fetch_all` can run independently. The user can pass one
/// or more on the CLI to skip the categories they don't need (the per-day
/// daily-health loop is the slow one — over an hour for a multi-year backfill
/// — so being able to run just `activities` for a few minutes is the main
/// motivating use case).
///
/// Note: daily-health and performance-metrics share a per-day fetch loop and
/// can't be split independently. Same for weight + blood-pressure (which
/// share a per-month loop). Hence the bundled category names below.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum FetchCategory {
    /// Daily health summary + performance metrics (per-day loop, slow).
    DailyHealth,
    /// Weight + blood pressure measurements (per-month loop, fast).
    WeightBp,
    /// Activities (paginated list, fast). Required after editing event
    /// types in Garmin Connect — pair with `--force` for that case.
    Activities,
    /// Activity details + splits (per-activity fetch, slow for backfill).
    /// Requires activities to be fetched first.
    ActivityDetails,
    /// Nutrition / food logging (per-day fetch, fast — only days with data).
    Nutrition,
}

impl FetchCategory {
    /// True if `selected` is empty (== "everything") or contains `self`.
    fn is_enabled(self, selected: &[FetchCategory]) -> bool {
        selected.is_empty() || selected.contains(&self)
    }
}

pub async fn fetch_all(
    config: &Config,
    from: NaiveDate,
    to: Option<NaiveDate>,
    force: bool,
    only: &[FetchCategory],
) -> Result<()> {
    println!("Authenticating...");
    let (_oauth1, oauth2) = crate::sync::authenticate().await?;
    let client = Client::new(oauth2);

    // Get display name (needed for some endpoints)
    let display_name = get_display_name(&client).await?;
    println!("Authenticated as {display_name}.");

    let to = to.unwrap_or_else(|| Utc::now().date_naive());
    let total_days = (to - from).num_days() + 1;

    if FetchCategory::DailyHealth.is_enabled(only) {
        fetch_daily_loop(config, &client, &display_name, from, to, total_days, force).await?;
    } else {
        println!("Skipping daily health (not selected by --only).");
    }

    // Weight & blood pressure fetch a date *range* per call, so they use a
    // different (month-based) incremental strategy than the per-day loop above.
    if FetchCategory::WeightBp.is_enabled(only) {
        fetch_weight_and_bp(config, &client, from, to, force).await?;
    } else {
        println!("Skipping weight & blood pressure (not selected by --only).");
    }

    // Activities are paginated (newest-first) from a single list endpoint and
    // partitioned by ISO week.
    if FetchCategory::Activities.is_enabled(only) {
        fetch_activities(config, &client, from, to, force).await?;
    } else {
        println!("Skipping activities (not selected by --only).");
    }

    // Activity details + splits: per-activity fetch of time-series and lap data.
    // Depends on activities being on disk (needs activity_ids).
    if FetchCategory::ActivityDetails.is_enabled(only) {
        fetch_activity_details_and_splits(config, &client, force).await?;
    } else {
        println!("Skipping activity details (not selected by --only).");
    }

    // Nutrition / food logging: per-day fetch of daily macro summary.
    if FetchCategory::Nutrition.is_enabled(only) {
        fetch_nutrition(config, &client, from, to, force).await?;
    } else {
        println!("Skipping nutrition (not selected by --only).");
    }

    Ok(())
}

/// Per-day daily-health + performance-metrics fetch loop. Lifted out of
/// `fetch_all` so it can be conditionally skipped via `--only`.
async fn fetch_daily_loop(
    config: &Config,
    client: &Client,
    display_name: &str,
    from: NaiveDate,
    to: NaiveDate,
    total_days: i64,
    force: bool,
) -> Result<()> {
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
                match fetch_daily_health(client, display_name, date).await {
                    Ok(record) => health_buf.push(record),
                    Err(e) => {
                        eprintln!("  Health {}: {}", date, e);
                        errors += 1;
                    }
                }
            }

            // Fetch performance
            if !has_perf {
                match fetch_performance(client, display_name, date).await {
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

// ---------------------------------------------------------------------------
// Activities
// ---------------------------------------------------------------------------
//
// Garmin's activity list endpoint is paginated and returns activities
// newest-first. We:
//   1. Page through it from offset 0
//   2. Stop when an activity's start date is older than `from`, or the page
//      comes back empty
//   3. Bucket activities by ISO week (matching the existing storage layout
//      written by the previous garmin-cli writer)
//   4. Skip weeks already on disk unless `--force` is set, but always
//      re-fetch the *current* week (the one containing today) so newly-
//      logged activities flow through on each run.
//
// Note: this is the only entity in the fetch pipeline that requires a full
// re-fetch under `--force` to pick up *edits* — the user can re-tag an old
// activity in Garmin Connect (e.g. set Event Type → Race) and the only way
// for that change to land in the local parquet is via `--force`. The
// alternative (always re-fetching every week) was rejected as too API-heavy
// for what is, in practice, an occasional need.

/// Top-level activity fetch. Pages the list endpoint, groups by ISO week,
/// and writes one parquet partition per week through the existing
/// `write_parquet_partition` helper.
async fn fetch_activities(
    config: &Config,
    client: &Client,
    from: NaiveDate,
    to: NaiveDate,
    force: bool,
) -> Result<()> {
    let today = Utc::now().date_naive();
    let existing_weeks = if force {
        HashSet::new()
    } else {
        load_existing_weeks(config)?
    };

    println!(
        "Fetching activities from {} to {}{}...",
        from,
        to,
        if force { " (force)" } else { "" }
    );

    // Paginate the list endpoint, stopping at the lower bound.
    let raw = match fetch_activities_range(client, from).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  Activity fetch failed: {}", e);
            return Ok(());
        }
    };

    if raw.is_empty() {
        println!("  No activities returned.");
        return Ok(());
    }

    // Partition by ISO week derived from the activity's local start date.
    // Activities outside [from, to] are dropped here. The paginator already
    // stops on `from`, but `to` filtering still happens (the user can request
    // a historical window like 2018-01..2018-12).
    let mut by_week: std::collections::BTreeMap<(i32, u32), Vec<serde_json::Value>> =
        std::collections::BTreeMap::new();
    let mut considered = 0usize;
    let mut out_of_range = 0usize;
    for act in raw {
        let local = match parse_garmin_local_datetime(act.get("startTimeLocal")) {
            Some(dt) => dt,
            None => continue,
        };
        let date = local.date();
        if date < from || date > to {
            out_of_range += 1;
            continue;
        }
        considered += 1;
        let week = date.iso_week();
        by_week
            .entry((week.year(), week.week()))
            .or_default()
            .push(act);
    }

    println!(
        "  Fetched {} activities in range ({} dropped as out-of-range), grouped into {} weeks.",
        considered,
        out_of_range,
        by_week.len()
    );

    // Write each week's partition. Skip weeks we already have unless --force,
    // but always rewrite the current week.
    let mut written = 0usize;
    let mut skipped = 0usize;
    for ((year, week), records) in by_week {
        let is_current = {
            let cur = today.iso_week();
            cur.year() == year && cur.week() == week
        };
        if !force && !is_current && existing_weeks.contains(&(year, week)) {
            skipped += 1;
            continue;
        }

        let partition = format!("{:04}-W{:02}", year, week);
        let df = activities_records_to_df(&records)?;
        write_parquet_partition(config, "activities", &partition, df, &["activity_id"])?;
        written += 1;
    }

    println!(
        "Activities: {} weeks written, {} weeks skipped (already on disk).",
        written, skipped
    );
    Ok(())
}

/// Page through Garmin's activity list endpoint until either the page comes
/// back empty or the activities returned are older than `from`. Returns the
/// raw JSON for every activity encountered (caller filters by date range).
///
/// Progress tracking is done in *date space* (days walked back from today
/// toward `from`) rather than activity count, since we don't know the total
/// activity count up front. This lets `elapsed_and_eta` produce a meaningful
/// ETA in the same shape as the daily-health and weight/BP loops.
async fn fetch_activities_range(
    client: &Client,
    from: NaiveDate,
) -> Result<Vec<serde_json::Value>> {
    const PAGE_SIZE: usize = 50;
    let mut out: Vec<serde_json::Value> = Vec::new();
    let mut start = 0usize;
    let started = Instant::now();

    let today = Utc::now().date_naive();
    let total_days_back = (today - from).num_days().max(1);
    // Track the earliest activity date seen across all pages so we can
    // compute "how much of the requested range we've walked through".
    let mut oldest_seen: Option<NaiveDate> = None;

    loop {
        let path = format!(
            "/activitylist-service/activities/search/activities?start={}&limit={}",
            start, PAGE_SIZE
        );
        let page: serde_json::Value = match client.get_json(&path).await {
            Ok(v) => v,
            Err(garmin_connect::Error::NotFound(_)) => break,
            Err(e) => return Err(AppError::Sync(e.to_string())),
        };
        let arr = match page.as_array() {
            Some(a) => a.clone(),
            None => break,
        };
        if arr.is_empty() {
            break;
        }

        // Walk the page once: update `oldest_seen` and detect whether the
        // entire page predates `from` (which means we're done paginating).
        let mut any_in_range = false;
        for act in &arr {
            let Some(dt) = parse_garmin_local_datetime(act.get("startTimeLocal")) else {
                continue;
            };
            let date = dt.date();
            if date >= from {
                any_in_range = true;
            }
            oldest_seen = Some(match oldest_seen {
                Some(prev) if prev <= date => prev,
                _ => date,
            });
        }

        let n = arr.len();
        out.extend(arr);
        start += n;

        // Progress in matching shape: covered_days / total_days_back, with
        // elapsed + ETA, plus the per-loop counters and most-recent boundary.
        let covered_days = oldest_seen
            .map(|d| (today - d).num_days().max(0))
            .unwrap_or(0);
        let (elapsed, eta) = elapsed_and_eta(started, covered_days, total_days_back);
        let pct = if total_days_back > 0 {
            (covered_days * 100) / total_days_back
        } else {
            0
        };
        println!(
            "  {}/{} days back ({}%), {} activities fetched, oldest: {}  [{} elapsed, {} ETA]",
            covered_days,
            total_days_back,
            pct,
            out.len(),
            oldest_seen
                .map(|d| d.to_string())
                .unwrap_or_else(|| "?".to_string()),
            elapsed,
            eta,
        );

        if !any_in_range {
            // Every activity on this page predates `from` — stop paginating.
            break;
        }
        if n < PAGE_SIZE {
            // Short page = end of list.
            break;
        }

        // Rate limit between pages.
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }

    Ok(out)
}

/// Build the activities DataFrame from raw API records. Mirrors the existing
/// parquet schema (29 columns) so `vstack` in `write_or_merge_parquet` lines
/// up correctly. Notably, populates `calories`, `avg_hr`, `max_hr` which the
/// previous garmin-cli writer was leaving 100% null even though they're in
/// the API response.
fn activities_records_to_df(records: &[serde_json::Value]) -> Result<DataFrame> {
    let mut activity_id: Vec<Option<i64>> = Vec::new();
    let mut profile_id: Vec<Option<i32>> = Vec::new();
    let mut activity_name: Vec<Option<String>> = Vec::new();
    let mut activity_type: Vec<Option<String>> = Vec::new();
    let mut start_local_us: Vec<Option<i64>> = Vec::new();
    let mut start_gmt_us: Vec<Option<i64>> = Vec::new();
    let mut duration_sec: Vec<Option<f64>> = Vec::new();
    let mut distance_m: Vec<Option<f64>> = Vec::new();
    let mut calories: Vec<Option<i32>> = Vec::new();
    let mut avg_hr: Vec<Option<i32>> = Vec::new();
    let mut max_hr: Vec<Option<i32>> = Vec::new();
    let mut avg_speed: Vec<Option<f64>> = Vec::new();
    let mut max_speed: Vec<Option<f64>> = Vec::new();
    let mut elevation_gain: Vec<Option<f64>> = Vec::new();
    let mut elevation_loss: Vec<Option<f64>> = Vec::new();
    let mut avg_cadence: Vec<Option<f64>> = Vec::new();
    let mut avg_power: Vec<Option<i32>> = Vec::new();
    let mut normalized_power: Vec<Option<i32>> = Vec::new();
    let mut training_effect: Vec<Option<f64>> = Vec::new();
    let mut training_load: Vec<Option<f64>> = Vec::new();
    let mut start_lat: Vec<Option<f64>> = Vec::new();
    let mut start_lon: Vec<Option<f64>> = Vec::new();
    let mut end_lat: Vec<Option<f64>> = Vec::new();
    let mut end_lon: Vec<Option<f64>> = Vec::new();
    let mut ground_contact_time: Vec<Option<f64>> = Vec::new();
    let mut vertical_oscillation: Vec<Option<f64>> = Vec::new();
    let mut stride_length: Vec<Option<f64>> = Vec::new();
    let mut location_name: Vec<Option<String>> = Vec::new();
    let mut raw_json: Vec<Option<String>> = Vec::new();

    for r in records {
        activity_id.push(r.get("activityId").and_then(|v| v.as_i64()));
        // profile_id is constant 2 across all 1517 existing rows for this user;
        // hard-code it to keep the schema stable. The field doesn't appear in
        // the activity list response under any obvious key.
        profile_id.push(Some(2));
        activity_name.push(
            r.get("activityName")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        );
        activity_type.push(
            r.get("activityType")
                .and_then(|v| v.get("typeKey"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        );
        start_local_us.push(
            parse_garmin_local_datetime(r.get("startTimeLocal"))
                .map(|dt| dt.and_utc().timestamp_micros()),
        );
        start_gmt_us.push(
            parse_garmin_local_datetime(r.get("startTimeGMT"))
                .map(|dt| dt.and_utc().timestamp_micros()),
        );
        duration_sec.push(r.get("duration").and_then(|v| v.as_f64()));
        distance_m.push(r.get("distance").and_then(|v| v.as_f64()));
        // calories/HR were 100% null in the previous writer but are in raw_json.
        calories.push(jf32_to_i32(r, "calories"));
        avg_hr.push(jf32_to_i32(r, "averageHR"));
        max_hr.push(jf32_to_i32(r, "maxHR"));
        avg_speed.push(r.get("averageSpeed").and_then(|v| v.as_f64()));
        max_speed.push(r.get("maxSpeed").and_then(|v| v.as_f64()));
        elevation_gain.push(r.get("elevationGain").and_then(|v| v.as_f64()));
        elevation_loss.push(r.get("elevationLoss").and_then(|v| v.as_f64()));
        // Garmin uses different cadence keys per sport; try the running one,
        // then fall back to biking. The double-precision (`maxDoubleCadence`)
        // variant is for max only — we want the average, so we don't need it.
        avg_cadence.push(
            r.get("averageRunningCadenceInStepsPerMinute")
                .and_then(|v| v.as_f64())
                .or_else(|| {
                    r.get("averageBikingCadenceInRevsPerMinute")
                        .and_then(|v| v.as_f64())
                }),
        );
        avg_power.push(jf32_to_i32(r, "avgPower"));
        normalized_power.push(jf32_to_i32(r, "normPower"));
        training_effect.push(r.get("aerobicTrainingEffect").and_then(|v| v.as_f64()));
        training_load.push(r.get("activityTrainingLoad").and_then(|v| v.as_f64()));
        start_lat.push(r.get("startLatitude").and_then(|v| v.as_f64()));
        start_lon.push(r.get("startLongitude").and_then(|v| v.as_f64()));
        end_lat.push(r.get("endLatitude").and_then(|v| v.as_f64()));
        end_lon.push(r.get("endLongitude").and_then(|v| v.as_f64()));
        ground_contact_time.push(r.get("avgGroundContactTime").and_then(|v| v.as_f64()));
        vertical_oscillation.push(r.get("avgVerticalOscillation").and_then(|v| v.as_f64()));
        stride_length.push(r.get("avgStrideLength").and_then(|v| v.as_f64()));
        location_name.push(
            r.get("locationName")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        );
        // Preserve the original JSON exactly so race-tag detection
        // (`eventType.typeKey == "race"`) keeps working from the parquet.
        raw_json.push(serde_json::to_string(r).ok());
    }

    // Build datetime[μs] series explicitly so the dtype matches the existing
    // parquet (the df! macro defaults to nanoseconds, which would break
    // vstack on merge).
    let local_series =
        Int64Chunked::from_iter_options("start_time_local".into(), start_local_us.into_iter())
            .into_datetime(TimeUnit::Microseconds, None)
            .into_series();
    let gmt_series =
        Int64Chunked::from_iter_options("start_time_gmt".into(), start_gmt_us.into_iter())
            .into_datetime(TimeUnit::Microseconds, None)
            .into_series();

    let df = df!(
        "activity_id" => &activity_id,
        "profile_id" => &profile_id,
        "activity_name" => &activity_name,
        "activity_type" => &activity_type,
        "duration_sec" => &duration_sec,
        "distance_m" => &distance_m,
        "calories" => &calories,
        "avg_hr" => &avg_hr,
        "max_hr" => &max_hr,
        "avg_speed" => &avg_speed,
        "max_speed" => &max_speed,
        "elevation_gain" => &elevation_gain,
        "elevation_loss" => &elevation_loss,
        "avg_cadence" => &avg_cadence,
        "avg_power" => &avg_power,
        "normalized_power" => &normalized_power,
        "training_effect" => &training_effect,
        "training_load" => &training_load,
        "start_lat" => &start_lat,
        "start_lon" => &start_lon,
        "end_lat" => &end_lat,
        "end_lon" => &end_lon,
        "ground_contact_time" => &ground_contact_time,
        "vertical_oscillation" => &vertical_oscillation,
        "stride_length" => &stride_length,
        "location_name" => &location_name,
        "raw_json" => &raw_json,
    )?;

    // Append the two datetime columns and reorder to match the existing schema.
    let mut df = df;
    df.with_column(local_series)?;
    df.with_column(gmt_series)?;
    let ordered = df.select([
        "activity_id",
        "profile_id",
        "activity_name",
        "activity_type",
        "start_time_local",
        "start_time_gmt",
        "duration_sec",
        "distance_m",
        "calories",
        "avg_hr",
        "max_hr",
        "avg_speed",
        "max_speed",
        "elevation_gain",
        "elevation_loss",
        "avg_cadence",
        "avg_power",
        "normalized_power",
        "training_effect",
        "training_load",
        "start_lat",
        "start_lon",
        "end_lat",
        "end_lon",
        "ground_contact_time",
        "vertical_oscillation",
        "stride_length",
        "location_name",
        "raw_json",
    ])?;

    Ok(ordered)
}

// ---------------------------------------------------------------------------
// Activity Details + Splits
// ---------------------------------------------------------------------------
//
// Per-activity fetch of time-series metrics (HR, cadence, speed, altitude,
// GPS, running dynamics) and lap/split data. Stored as one parquet file per
// activity_id under activity_details/ and activity_splits/.
//
// Incremental: file existence = already fetched. Activities that return 404
// or empty data get a zero-row parquet with the correct schema.

/// Retry an async closure on `garmin_connect::Error::RateLimited` with
/// exponential backoff. Returns the first non-429 result, or the last
/// error after `max_retries` attempts.
async fn retry_on_rate_limit<F, Fut, T>(
    max_retries: u32,
    mut f: F,
) -> std::result::Result<T, garmin_connect::Error>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = std::result::Result<T, garmin_connect::Error>>,
{
    let mut attempt = 0;
    loop {
        match f().await {
            Err(garmin_connect::Error::RateLimited) if attempt < max_retries => {
                attempt += 1;
                let backoff = std::time::Duration::from_secs(1 << attempt); // 2s, 4s, 8s, ...
                eprintln!(
                    "  Rate limited — backing off {}s (attempt {}/{})",
                    backoff.as_secs(),
                    attempt,
                    max_retries
                );
                tokio::time::sleep(backoff).await;
            }
            result => return result,
        }
    }
}

/// Build a column-index mapping from the Garmin `metricDescriptors` array.
/// Returns a map from metric key string (e.g. "directHeartRate") to its
/// position in each row's `metrics` array.
fn build_metric_index(
    descriptors: &[serde_json::Value],
) -> std::collections::HashMap<String, usize> {
    let mut map = std::collections::HashMap::new();
    for desc in descriptors {
        if let (Some(key), Some(idx)) = (
            desc.get("key").and_then(|k| k.as_str()),
            desc.get("metricsIndex").and_then(|v| v.as_u64()),
        ) {
            map.insert(key.to_string(), idx as usize);
        }
    }
    map
}

/// Parse the activity details JSON response into a DataFrame.
///
/// The response contains:
/// - `metricDescriptors`: array describing which metric is at which index
/// - `activityDetailMetrics`: array of `{ metrics: [f64...] }` rows
/// - `geoPolylineDTO.polyline`: array of `{ lat, lon, altitude, ... }` GPS points
///
/// GPS points are linearly interpolated onto the metrics timestamps by
/// matching on the cumulative distance or elapsed time.
fn parse_activity_details(activity_id: i64, json: &serde_json::Value) -> Result<DataFrame> {
    let descriptors = json
        .get("metricDescriptors")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let metrics_rows = json
        .get("activityDetailMetrics")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let idx = build_metric_index(&descriptors);

    // Pre-allocate column vectors
    let n = metrics_rows.len();
    let mut activity_ids = vec![activity_id; n];
    let mut timestamp_ms_col: Vec<Option<i64>> = Vec::with_capacity(n);
    let mut elapsed_sec_col: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut heart_rate_col: Vec<Option<i32>> = Vec::with_capacity(n);
    let mut cadence_col: Vec<Option<i32>> = Vec::with_capacity(n);
    let mut speed_col: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut altitude_col: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut distance_col: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut power_col: Vec<Option<i32>> = Vec::with_capacity(n);
    let mut temperature_col: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut gct_col: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut vo_col: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut stride_col: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut gcb_col: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut resp_col: Vec<Option<f64>> = Vec::with_capacity(n);

    // Helper to extract a metric from a row's metrics array
    let get_f64 = |row: &serde_json::Value, key: &str| -> Option<f64> {
        let i = idx.get(key)?;
        row.get("metrics")
            .and_then(|m| m.as_array())
            .and_then(|a| a.get(*i))
            .and_then(|v| v.as_f64())
    };

    for row in &metrics_rows {
        // Timestamp: directTimestamp gives epoch ms
        let ts_ms = get_f64(row, "directTimestamp").map(|v| v as i64);
        timestamp_ms_col.push(ts_ms);

        // Elapsed duration in seconds (accounts for auto-pause)
        let ts = get_f64(row, "sumElapsedDuration");
        elapsed_sec_col.push(ts);

        let hr = get_f64(row, "directHeartRate").map(|v| v.round() as i32);
        heart_rate_col.push(hr);

        // Running cadence (directRunCadence), cycling cadence, or
        // directDoubleCadence (used by some watch faces / CIQ apps)
        let cadence = get_f64(row, "directRunCadence")
            .or_else(|| get_f64(row, "bikeCadence"))
            .or_else(|| get_f64(row, "directDoubleCadence"))
            .map(|v| v.round() as i32);
        cadence_col.push(cadence);

        speed_col.push(get_f64(row, "directSpeed"));
        altitude_col.push(get_f64(row, "directElevation"));
        distance_col.push(get_f64(row, "sumDistance"));
        power_col.push(get_f64(row, "directPower").map(|v| v.round() as i32));
        temperature_col.push(get_f64(row, "directAirTemperature"));
        gct_col.push(get_f64(row, "directGroundContactTime"));
        vo_col.push(get_f64(row, "directVerticalOscillation"));
        stride_col.push(get_f64(row, "directStrideLength"));
        gcb_col.push(get_f64(row, "directGroundContactBalanceLeft"));
        resp_col.push(get_f64(row, "directRespirationRate"));
    }

    // GPS: lat/lon come directly from the metrics array (directLatitude,
    // directLongitude). No polyline interpolation needed.
    let mut lat_col: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut lon_col: Vec<Option<f64>> = Vec::with_capacity(n);
    for row in &metrics_rows {
        lat_col.push(get_f64(row, "directLatitude"));
        lon_col.push(get_f64(row, "directLongitude"));
    }

    // If there are no rows, we still need a zero-row DF with the right schema
    if n == 0 {
        activity_ids.clear();
    }

    let df = DataFrame::new(vec![
        Column::new("activity_id".into(), &activity_ids),
        Column::new("timestamp_ms".into(), &timestamp_ms_col),
        Column::new("elapsed_sec".into(), &elapsed_sec_col),
        Column::new("heart_rate".into(), &heart_rate_col),
        Column::new("cadence".into(), &cadence_col),
        Column::new("speed".into(), &speed_col),
        Column::new("altitude".into(), &altitude_col),
        Column::new("distance".into(), &distance_col),
        Column::new("power".into(), &power_col),
        Column::new("temperature".into(), &temperature_col),
        Column::new("ground_contact_time".into(), &gct_col),
        Column::new("vertical_oscillation".into(), &vo_col),
        Column::new("stride_length".into(), &stride_col),
        Column::new("ground_contact_balance".into(), &gcb_col),
        Column::new("respiration_rate".into(), &resp_col),
        Column::new("lat".into(), &lat_col),
        Column::new("lon".into(), &lon_col),
    ])?;

    Ok(df)
}

/// Parse the activity splits JSON response into a DataFrame.
fn parse_activity_splits(activity_id: i64, json: &serde_json::Value) -> Result<DataFrame> {
    // The splits endpoint can return splits under various keys
    let splits = json
        .get("lapDTOs")
        .or_else(|| json.get("splitDTOs"))
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let n = splits.len();
    let activity_ids = vec![activity_id; n];
    let mut split_number_col: Vec<i32> = Vec::with_capacity(n);
    let mut split_type_col: Vec<Option<String>> = Vec::with_capacity(n);
    let mut distance_m_col: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut duration_sec_col: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut avg_speed_col: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut avg_hr_col: Vec<Option<i32>> = Vec::with_capacity(n);
    let mut max_hr_col: Vec<Option<i32>> = Vec::with_capacity(n);
    let mut avg_cadence_col: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut avg_power_col: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut elev_gain_col: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut elev_loss_col: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut start_lat_col: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut start_lon_col: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut avg_gct_col: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut avg_vo_col: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut avg_stride_col: Vec<Option<f64>> = Vec::with_capacity(n);

    for (i, split) in splits.iter().enumerate() {
        split_number_col.push(
            split
                .get("lapIndex")
                .and_then(|v| v.as_i64())
                .map(|v| v as i32)
                .unwrap_or((i + 1) as i32),
        );

        // Garmin uses "intensityType" (e.g. "INTERVAL", "REST")
        split_type_col.push(
            split
                .get("intensityType")
                .or_else(|| split.get("splitType"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        );

        distance_m_col.push(split.get("distance").and_then(|v| v.as_f64()));
        duration_sec_col.push(split.get("duration").and_then(|v| v.as_f64()));
        avg_speed_col.push(split.get("averageSpeed").and_then(|v| v.as_f64()));

        avg_hr_col.push(
            split
                .get("averageHR")
                .and_then(|v| v.as_f64())
                .map(|v| v.round() as i32),
        );
        max_hr_col.push(
            split
                .get("maxHR")
                .and_then(|v| v.as_f64())
                .map(|v| v.round() as i32),
        );

        avg_cadence_col.push(split.get("averageRunCadence").and_then(|v| v.as_f64()));
        avg_power_col.push(split.get("averagePower").and_then(|v| v.as_f64()));
        elev_gain_col.push(split.get("elevationGain").and_then(|v| v.as_f64()));
        elev_loss_col.push(split.get("elevationLoss").and_then(|v| v.as_f64()));
        start_lat_col.push(split.get("startLatitude").and_then(|v| v.as_f64()));
        start_lon_col.push(split.get("startLongitude").and_then(|v| v.as_f64()));
        avg_gct_col.push(split.get("groundContactTime").and_then(|v| v.as_f64()));
        avg_vo_col.push(split.get("verticalOscillation").and_then(|v| v.as_f64()));
        avg_stride_col.push(split.get("strideLength").and_then(|v| v.as_f64()));
    }

    let df = DataFrame::new(vec![
        Column::new("activity_id".into(), &activity_ids),
        Column::new("split_number".into(), &split_number_col),
        Column::new("split_type".into(), &split_type_col),
        Column::new("distance_m".into(), &distance_m_col),
        Column::new("duration_sec".into(), &duration_sec_col),
        Column::new("avg_speed".into(), &avg_speed_col),
        Column::new("avg_hr".into(), &avg_hr_col),
        Column::new("max_hr".into(), &max_hr_col),
        Column::new("avg_cadence".into(), &avg_cadence_col),
        Column::new("avg_power".into(), &avg_power_col),
        Column::new("elevation_gain".into(), &elev_gain_col),
        Column::new("elevation_loss".into(), &elev_loss_col),
        Column::new("start_lat".into(), &start_lat_col),
        Column::new("start_lon".into(), &start_lon_col),
        Column::new("avg_ground_contact_time".into(), &avg_gct_col),
        Column::new("avg_vertical_oscillation".into(), &avg_vo_col),
        Column::new("avg_stride_length".into(), &avg_stride_col),
    ])?;

    Ok(df)
}

/// Write a per-activity parquet file (atomic via .tmp + rename).
fn write_activity_parquet(
    dir: &std::path::Path,
    activity_id: i64,
    df: &mut DataFrame,
) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("{}.parquet", activity_id));
    let tmp_path = path.with_extension("parquet.tmp");
    {
        let mut file = std::fs::File::create(&tmp_path)?;
        ParquetWriter::new(&mut file)
            .with_compression(ParquetCompression::Zstd(None))
            .finish(df)?;
    }
    std::fs::rename(&tmp_path, &path)?;
    Ok(())
}

/// Load all activity_ids from existing per-activity parquet files in a directory.
/// File names are `{activity_id}.parquet`.
fn load_existing_activity_ids(dir: &std::path::Path) -> HashSet<i64> {
    let mut set = HashSet::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return set,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("parquet") {
            continue;
        }
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str())
            && let Ok(id) = stem.parse::<i64>()
        {
            set.insert(id);
        }
    }
    set
}

/// Load all activity_ids from the activities parquet files on disk.
fn load_all_activity_ids(config: &Config) -> Result<Vec<i64>> {
    let dir = config.garmin_storage_path.join("activities");
    if !crate::data::dir_has_parquet(&dir) {
        return Err(AppError::Data(
            "No activities on disk. Run `model-health fetch --from <date>` first.".to_string(),
        ));
    }

    let pattern = dir.join("*.parquet");
    let lf = LazyFrame::scan_parquet(
        pattern.to_string_lossy().as_ref(),
        ScanArgsParquet {
            allow_missing_columns: true,
            ..Default::default()
        },
    )?;

    let df = lf.select([col("activity_id")]).collect()?;
    let ids: Vec<i64> = df
        .column("activity_id")?
        .i64()?
        .into_no_null_iter()
        .collect();

    Ok(ids)
}

/// Fetch activity details and splits for all activities not yet fetched.
async fn fetch_activity_details_and_splits(
    config: &Config,
    client: &Client,
    force: bool,
) -> Result<()> {
    let all_ids = match load_all_activity_ids(config) {
        Ok(ids) => ids,
        Err(e) => {
            eprintln!("  Skipping activity details: {}", e);
            return Ok(());
        }
    };

    let details_dir = config.garmin_storage_path.join("activity_details");
    let splits_dir = config.garmin_storage_path.join("activity_splits");

    let existing = if force {
        HashSet::new()
    } else {
        // Check details dir only — both are always written together
        load_existing_activity_ids(&details_dir)
    };

    let mut todo: Vec<i64> = all_ids
        .into_iter()
        .filter(|id| !existing.contains(id))
        .collect();
    todo.sort(); // oldest first (activity IDs are monotonically increasing)

    if todo.is_empty() {
        println!(
            "Activity details: all {} activities already fetched.",
            existing.len()
        );
        return Ok(());
    }

    println!(
        "Fetching activity details for {} activities ({} already on disk){}...",
        todo.len(),
        existing.len(),
        if force { " (force)" } else { "" },
    );

    let started = Instant::now();
    let total = todo.len() as i64;
    let mut fetched = 0i64;
    let mut errors = 0u32;

    for &activity_id in &todo {
        fetched += 1;
        let (elapsed, eta) = elapsed_and_eta(started, fetched, total);
        print!(
            "\r  Activity {}/{} (id={})  [{} elapsed, {} ETA]    ",
            fetched, total, activity_id, elapsed, eta
        );

        // Fetch details
        let details_path = format!(
            "/activity-service/activity/{}/details?maxChartSize=100000&maxPolylineSize=100000",
            activity_id
        );
        let details_result =
            retry_on_rate_limit(3, || client.get_json::<serde_json::Value>(&details_path)).await;

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        // Fetch splits
        let splits_path = format!("/activity-service/activity/{}/splits", activity_id);
        let splits_result =
            retry_on_rate_limit(3, || client.get_json::<serde_json::Value>(&splits_path)).await;

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        // Classify results: Ok => data, NotFound => no data (write zero-row),
        // other error => skip entirely (don't write, so it gets retried next run)
        let details_json = match details_result {
            Ok(v) => Some(v),
            Err(garmin_connect::Error::NotFound(_)) => None,
            Err(e) => {
                errors += 1;
                eprintln!(
                    "\n  Warning: details fetch failed for {}: {} (will retry next run)",
                    activity_id, e
                );
                continue;
            }
        };
        let splits_json = match splits_result {
            Ok(v) => Some(v),
            Err(garmin_connect::Error::NotFound(_)) => None,
            Err(e) => {
                errors += 1;
                eprintln!(
                    "\n  Warning: splits fetch failed for {}: {} (will retry next run)",
                    activity_id, e
                );
                continue;
            }
        };

        // Parse and write details (zero-row if 404 / no data)
        let mut details_df = match &details_json {
            Some(json) => parse_activity_details(activity_id, json)?,
            None => {
                parse_activity_details(activity_id, &serde_json::Value::Object(Default::default()))?
            }
        };
        write_activity_parquet(&details_dir, activity_id, &mut details_df)?;

        // Parse and write splits (zero-row if 404 / no data)
        let mut splits_df = match &splits_json {
            Some(json) => parse_activity_splits(activity_id, json)?,
            None => {
                parse_activity_splits(activity_id, &serde_json::Value::Object(Default::default()))?
            }
        };
        write_activity_parquet(&splits_dir, activity_id, &mut splits_df)?;
    }

    println!();
    println!(
        "  Done: {} activities fetched in {}{}",
        total,
        format_secs(started.elapsed().as_secs()),
        if errors > 0 {
            format!(" ({} errors)", errors)
        } else {
            String::new()
        }
    );

    Ok(())
}

/// List the ISO weeks already represented as `activities/YYYY-Wnn.parquet`
/// files on disk.
fn load_existing_weeks(config: &Config) -> Result<HashSet<(i32, u32)>> {
    let dir = config.garmin_storage_path.join("activities");
    let mut set = HashSet::new();
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(set),
        Err(e) => {
            tracing::warn!(
                "load_existing_weeks: cannot read {}: {} — proceeding as if empty",
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
        // Expect "YYYY-Wnn"
        if stem.len() == 8
            && &stem.as_bytes()[4..6] == b"-W"
            && let (Ok(y), Ok(w)) = (stem[..4].parse::<i32>(), stem[6..].parse::<u32>())
        {
            set.insert((y, w));
        }
    }
    Ok(set)
}

/// Parse a Garmin "YYYY-MM-DD HH:MM:SS" string (used for both startTimeLocal
/// and startTimeGMT) into a NaiveDateTime. Returns None on missing/malformed
/// input — caller treats that as "skip this activity".
fn parse_garmin_local_datetime(v: Option<&serde_json::Value>) -> Option<chrono::NaiveDateTime> {
    let s = v.and_then(|v| v.as_str())?;
    chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S").ok()
}

/// Garmin's activity list returns numeric fields (calories, HR) as f64 even
/// when they're conceptually integers. Convert to Option<i32> via the i64
/// rounding path so we don't lose values that happen to be whole.
fn jf32_to_i32(v: &serde_json::Value, key: &str) -> Option<i32> {
    v.get(key)
        .and_then(|v| v.as_f64())
        .map(|f| f.round() as i32)
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
    let dir = config.garmin_storage_path.join(entity_dir);
    // Polars' scan_parquet returns Ok for a glob that matches zero files and
    // only fails later at collect() with "expected at least 1 source", so
    // short-circuit before scanning when nothing is on disk yet.
    if !crate::data::dir_has_parquet(&dir) {
        return Ok(HashSet::new());
    }

    let pattern = dir.join("*.parquet");
    let pattern_str = pattern.to_string_lossy().to_string();
    let lf = LazyFrame::scan_parquet(&pattern_str, Default::default())?;
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
        // Align schemas: select columns present in new_data, adding null
        // columns for any that exist in new_data but not in the old file
        // (e.g., newly added fields like consumed_calories).
        let our_cols: Vec<String> = new_data
            .get_column_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let existing_col_names: HashSet<String> = existing
            .get_column_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let select_exprs: Vec<Expr> = our_cols
            .iter()
            .map(|c| {
                if existing_col_names.contains(c) {
                    col(c.as_str())
                } else {
                    // New column not in old file — fill with null, matching the
                    // dtype from new_data so vstack succeeds.
                    let dtype = new_data.column(c).unwrap().dtype().clone();
                    lit(NULL).cast(dtype).alias(c)
                }
            })
            .collect();
        let existing_aligned = existing.lazy().select(select_exprs).collect()?;
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
    let mut consumed_cal = Vec::new();

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
        consumed_cal.push(oi32(r, "consumed_calories"));
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
        "consumed_calories" => &consumed_cal,
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

/// Probe the activity detail + splits endpoints for a single activity ID.
/// Used to inspect the actual API response structure before writing parsers.
pub async fn probe_activity(activity_id: i64, max_chart_size: u32) -> Result<()> {
    println!("Authenticating...");
    let (_oauth1, oauth2) = crate::sync::authenticate().await?;
    let client = Client::new(oauth2);
    println!("Authenticated.\n");

    let endpoints: Vec<(&str, String)> = vec![
        (
            "activity details",
            format!(
                "/activity-service/activity/{}/details?maxChartSize={}&maxPolylineSize={}",
                activity_id, max_chart_size, max_chart_size
            ),
        ),
        (
            "activity splits",
            format!("/activity-service/activity/{}/splits", activity_id),
        ),
    ];

    for (label, path) in endpoints {
        println!("==== {} ====", label);
        println!("GET {}", path);
        match client.get_json::<serde_json::Value>(&path).await {
            Ok(v) => {
                summarize_json(&v);
                println!(
                    "FULL: {}",
                    serde_json::to_string_pretty(&v).unwrap_or_default()
                );
            }
            Err(e) => println!("ERROR: {}", e),
        }
        println!();
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
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
        "consumed_calories": ji32(&health, "consumedKilocalories"),
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

// ---------------------------------------------------------------------------
// Nutrition / food logging
// ---------------------------------------------------------------------------

/// Fetch daily nutrition summaries for the requested date range.
/// Uses a per-day loop similar to daily health, but only stores the daily
/// macro totals (calories, protein, carbs, fat) — not individual food entries.
async fn fetch_nutrition(
    config: &Config,
    client: &Client,
    from: NaiveDate,
    to: NaiveDate,
    force: bool,
) -> Result<()> {
    let existing = if force {
        HashSet::new()
    } else {
        load_existing_dates(config, "nutrition")?
    };

    let today = Utc::now().date_naive();
    let mut date = from;
    let total_days = (to - from).num_days() + 1;
    let mut processed = 0i64;
    let mut total_records = 0usize;
    let mut errors = 0usize;
    let mut skipped = 0usize;
    let started = Instant::now();

    // Accumulate records per month, then flush.
    let mut current_month: Option<(i32, u32)> = None;
    let mut month_records: Vec<serde_json::Value> = Vec::new();

    println!("Fetching nutrition ({} days)...", total_days);

    while date <= to {
        let ym = (date.year(), date.month());
        let is_today = date == today;

        // Flush previous month when we cross a boundary.
        if let Some(prev) = current_month
            && prev != ym
            && !month_records.is_empty()
        {
            let partition = format!("{:04}-{:02}", prev.0, prev.1);
            let df = nutrition_records_to_df(&month_records)?;
            write_parquet_partition(config, "nutrition", &partition, df, &["date"])?;
            month_records.clear();
        }
        current_month = Some(ym);

        if !is_today && existing.contains(&date) {
            skipped += 1;
        } else {
            match fetch_nutrition_day(client, date).await {
                Ok(Some(record)) => {
                    total_records += 1;
                    month_records.push(record);
                }
                Ok(None) => {} // no food logged that day
                Err(e) => {
                    eprintln!("  Nutrition {}: {}", date, e);
                    errors += 1;
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }

        processed += 1;
        if processed % 30 == 0 || date == to {
            let (elapsed, eta) = elapsed_and_eta(started, processed, total_days);
            println!(
                "  {}: {}/{} days, {} records, {} skipped  [{} elapsed, {} ETA]",
                date, processed, total_days, total_records, skipped, elapsed, eta,
            );
        }

        date += Duration::days(1);
    }

    // Flush remaining month.
    if let Some(ym) = current_month
        && !month_records.is_empty()
    {
        let partition = format!("{:04}-{:02}", ym.0, ym.1);
        let df = nutrition_records_to_df(&month_records)?;
        write_parquet_partition(config, "nutrition", &partition, df, &["date"])?;
    }

    println!("Nutrition: {} records, {} errors.", total_records, errors);
    Ok(())
}

/// Fetch the daily nutrition summary for a single date.
/// Returns None if no food was logged that day (404 or empty data).
async fn fetch_nutrition_day(
    client: &Client,
    date: NaiveDate,
) -> Result<Option<serde_json::Value>> {
    let path = format!("/nutrition-service/food/logs/{}", date);
    let data: serde_json::Value = match client.get_json(&path).await {
        Ok(d) => d,
        Err(garmin_connect::Error::NotFound(_)) => return Ok(None),
        Err(garmin_connect::Error::Api { status: 405, .. }) => return Ok(None),
        Err(e) => return Err(AppError::Sync(e.to_string())),
    };

    let daily = match data.get("dailyNutritionContent") {
        Some(d) if d.is_object() => d,
        _ => return Ok(None),
    };

    let calories = daily
        .get("calories")
        .and_then(|v| v.as_i64())
        .map(|v| v as i32);

    // If no calories logged, treat as no data for the day.
    if calories.is_none() || calories == Some(0) {
        return Ok(None);
    }

    Ok(Some(serde_json::json!({
        "_date": date.to_string(),
        "consumed_calories": calories,
        "protein_g": daily.get("protein").and_then(|v| v.as_f64()),
        "carbs_g": daily.get("carbs").and_then(|v| v.as_f64()),
        "fat_g": daily.get("fat").and_then(|v| v.as_f64()),
    })))
}

/// Convert a batch of nutrition JSON records into a DataFrame.
fn nutrition_records_to_df(records: &[serde_json::Value]) -> Result<DataFrame> {
    let mut dates = Vec::new();
    let mut consumed_cal = Vec::new();
    let mut protein = Vec::new();
    let mut carbs = Vec::new();
    let mut fat = Vec::new();

    for r in records {
        dates.push(r["_date"].as_str().unwrap_or("").to_string());
        consumed_cal.push(oi32(r, "consumed_calories"));
        protein.push(r.get("protein_g").and_then(|v| v.as_f64()));
        carbs.push(r.get("carbs_g").and_then(|v| v.as_f64()));
        fat.push(r.get("fat_g").and_then(|v| v.as_f64()));
    }

    let date_series: Vec<Option<NaiveDate>> = dates
        .iter()
        .map(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
        .collect();

    let df = df!(
        "date" => &date_series,
        "consumed_calories" => &consumed_cal,
        "protein_g" => &protein,
        "carbs_g" => &carbs,
        "fat_g" => &fat,
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
