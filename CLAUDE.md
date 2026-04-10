# model-health

Rust CLI tool that fetches personal health/performance data from Garmin Connect,
stores it as partitioned Parquet files, and runs ML models + analysis on it.

## Storage Paths

- **Garmin data:** `~/Library/Application Support/garmin/` (configurable via `config.toml`)
- **Derived data / models:** `~/Library/Application Support/model-health/`
- **Config:** `~/Library/Application Support/model-health/config.toml` (via `dirs::config_dir()`)

## Data Entities and Schemas

All data is stored as Parquet files, partitioned by month (or week for activities).

### daily_health (monthly partitioned, ~3400 rows, 2017-01 to present)
Primary date column: `date` (Date)

| Column | Type | Notes |
|--------|------|-------|
| steps, step_goal | i32 | |
| total_calories, active_calories, bmr_calories | i32 | |
| resting_hr | i32 | |
| sleep_seconds | i32 | Total sleep; Garmin records sleep against the **morning you wake up** |
| deep_sleep_seconds, light_sleep_seconds, rem_sleep_seconds | i32 | Sleep stages |
| sleep_score | i32 | Garmin sleep quality score |
| avg_stress, max_stress | i32 | |
| body_battery_start, body_battery_end | i32 | Morning start / evening end |
| hrv_weekly_avg, hrv_last_night | i32 | |
| hrv_status | String | Categorical |
| avg_respiration | f64 | |
| avg_spo2, lowest_spo2 | i32 | |
| hydration_ml | i32 | |
| moderate_intensity_min, vigorous_intensity_min | i32 | |

### performance_metrics (monthly partitioned, same date range as daily_health)
Primary date column: `date` (Date)

| Column | Type |
|--------|------|
| vo2max | f64 |
| fitness_age | i32 |
| training_readiness | i32 |
| training_status | String |
| race_5k_sec, race_10k_sec, race_half_sec, race_marathon_sec | i32 |
| endurance_score, hill_score | i32 |

### activities (weekly partitioned, ~1500 rows)
Primary date column: `start_time_local` (Datetime[us])

| Column | Type | Notes |
|--------|------|-------|
| activity_id | i64 | |
| activity_name, activity_type | String | |
| start_time_local, start_time_gmt | Datetime[us] | |
| duration_sec | f64 | |
| distance_m | f64 | |
| calories, avg_hr, max_hr | i32 | |
| avg_speed, max_speed | f64 | |
| elevation_gain, elevation_loss | f64 | |
| avg_cadence, avg_power, normalized_power | f64/i32 | |
| training_effect, training_load | f64 | |
| start_lat, start_lon, end_lat, end_lon | f64 | |
| ground_contact_time, vertical_oscillation, stride_length | f64 | Running dynamics |
| location_name | String | |
| raw_json | String | Full original JSON |

### weight (monthly partitioned, ~60 rows)
Primary date column: `date` (Date)

Columns: weight_kg, weight_grams, bmi, body_fat, body_water, bone_mass,
muscle_mass, physique_rating (i32), visceral_fat, metabolic_age (i32),
source_type (String)

### blood_pressure (monthly partitioned, ~37 rows)
Primary date column: `date` (Date)

Columns: systolic, diastolic, pulse (i32), category (i32), category_name (String),
notes (String), timestamp_gmt, timestamp_local (String)

## CLI Commands

```
fetch --from DATE [--to DATE] [--force] [--only <category>]
    Categories: daily-health, weight-bp, activities
probe --date DATE          # dump raw Garmin JSON (debugging)
status                     # data coverage summary
profile                    # column-level data quality stats
train [--target TARGET]    # train Random Forest model
predict [--target TARGET]  # predict using latest model
races                      # race retrospectives + good-vs-bad contrast
decompose                  # OLS decomposition of stress/RHR into training vs life
injury-risk                # tiered warning system (red/yellow/volume)
```

**ML targets:** next_day_resting_hr (default), next_day_sleep_hours, next_day_steps, next_day_hrv

## Key Conventions

- Sleep data is recorded against the day you **wake up**, not the day you fall asleep.
  If you took medication on Monday and want to see its effect on Monday night's sleep,
  look at Tuesday's `sleep_seconds`/`rem_sleep_seconds`.
- `features.rs` derives `sleep_hours` (sleep_seconds / 3600), rolling averages, distance
  windows (7d/28d/90d), etc.
- `decompose.rs` writes `decomposed_health.parquet` to `data_dir` (not `garmin_storage_path`)
  because it's derived data.
- The project uses Polars (lazy frames) for all data operations.
- Python with `polars` is available for ad-hoc analysis against the parquet files.
  `numpy` and `scipy` are **not** installed.

## Building

Standard Rust project: `cargo build`, `cargo run -- <command>`.
