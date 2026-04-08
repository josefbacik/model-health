# model-health

> **Status:** Work in progress. Expect breaking changes, rough edges, and missing features.

A Rust CLI for fetching personal health and performance data from Garmin Connect and training simple ML models on it (e.g. predicting next-day resting heart rate).

## What it does

- Fetches daily health stats, performance metrics, activities, weight, and blood pressure from Garmin Connect and stores them locally as Parquet.
- Profiles and summarizes the local dataset.
- Builds features and trains classical ML models (via [smartcore](https://smartcore.rs/)) against configurable targets.
- Runs predictions from the latest trained model.

## Requirements

- Rust (edition 2024)
- A sibling checkout of [`garmin-connect`](https://github.com/) at `../garmin-connect` (referenced as a path dependency in `Cargo.toml`)
- A Garmin Connect account

## Build

```sh
cargo build --release
```

## Usage

```sh
# Fetch data for a date range
model-health fetch --from 2024-01-01 --to 2024-12-31

# Re-fetch (upsert) dates already cached locally
model-health fetch --from 2024-01-01 --force

# Show coverage and sync status
model-health status

# Profile data quality (columns, null rates, stats)
model-health profile

# Train a model for a given target
model-health train --target next_day_resting_hr

# Predict using the latest trained model
model-health predict --target next_day_resting_hr

# Dump raw JSON from each Garmin endpoint for one date (debugging)
model-health probe --date 2024-06-01
```

## Configuration

On first run, defaults are used. An optional config file may be placed at:

```
$XDG_CONFIG_HOME/model-health/config.toml
```

Supported keys:

```toml
garmin_storage_path = "/path/to/garmin/parquet/store"
data_dir            = "/path/to/model-health/data"
default_target      = "next_day_resting_hr"
min_training_days   = 60
```

## License

MIT — see [LICENSE](LICENSE).
