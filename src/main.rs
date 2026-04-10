mod config;
mod data;
mod decompose;
mod error;
mod features;
mod fetch;
mod fitness;
mod injury_risk;
mod model;
mod races;
mod sync;
mod validation;

use chrono::NaiveDate;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "model-health",
    about = "Train ML models on your Garmin health data"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Fetch health + performance data from Garmin Connect
    Fetch {
        /// Start date (YYYY-MM-DD)
        #[arg(long)]
        from: NaiveDate,
        /// End date (YYYY-MM-DD, defaults to today)
        #[arg(long)]
        to: Option<NaiveDate>,
        /// Re-fetch dates even if already cached locally. Note: this is an
        /// *upsert* — fetched rows overwrite existing ones (dedup-keep-last),
        /// but rows in the existing parquet that the API no longer returns
        /// for that range are preserved. For a true clean rebuild, delete
        /// the relevant subdirectory of the garmin storage path before
        /// running.
        #[arg(long)]
        force: bool,
        /// Restrict fetch to one or more categories. Repeat the flag to
        /// select multiple. When omitted, all categories run (default).
        ///
        /// Categories:
        ///   daily-health      per-day daily summary + performance metrics (slow)
        ///   weight-bp         monthly weight + blood pressure (fast)
        ///   activities        paginated activity list (fast)
        ///   activity-details  per-activity time-series + splits (slow for backfill)
        ///
        /// Example: `model-health fetch --from 2017-01-01 --force --only activities`
        #[arg(long, value_enum)]
        only: Vec<fetch::FetchCategory>,
    },
    /// Dump raw JSON from each Garmin endpoint for a single date (debugging)
    Probe {
        /// Date to probe (YYYY-MM-DD)
        #[arg(long)]
        date: NaiveDate,
    },
    /// Dump raw JSON from the activity detail + splits endpoints (debugging)
    ProbeActivity {
        /// Activity ID to probe
        #[arg(long)]
        id: i64,
        /// Max chart size to request (controls time-series resolution)
        #[arg(long, default_value = "100000")]
        max_chart_size: u32,
    },
    /// Show data coverage and sync status
    Status,
    /// Profile data quality (column names, null rates, stats)
    Profile,
    /// Train a model
    Train {
        /// Prediction target
        #[arg(long, default_value = "next_day_resting_hr")]
        target: String,
    },
    /// Run prediction using the latest model
    Predict {
        /// Prediction target
        #[arg(long, default_value = "next_day_resting_hr")]
        target: String,
    },
    /// Race retrospective: per-race report cards plus a contrast of what
    /// separated good races (close to PR) from bad ones in each distance bucket.
    Races,
    /// POC: decompose stress / RHR into training-explained vs external
    /// (residual) components via per-target OLS regression on trailing
    /// training-load features. Console-only — does not write any new
    /// data files.
    Decompose,
    /// Injury risk assessment: shows recent training load and physiological
    /// signals with a tiered warning system (red / yellow / volume) based on
    /// historical injury-gap analysis.
    InjuryRisk,
    /// Cardiac efficiency fitness report: grade-adjusted speed/HR trends,
    /// current state, and training drivers.
    Fitness,
    /// Cardiac drift analysis: how much does pace:HR efficiency drop from
    /// the first to second half of each run? Lower drift = better aerobic base.
    Drift,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    let config = config::Config::load()?;

    match cli.command {
        Commands::Fetch {
            from,
            to,
            force,
            only,
        } => {
            fetch::fetch_all(&config, from, to, force, &only).await?;
        }
        Commands::Probe { date } => {
            fetch::probe(date).await?;
        }
        Commands::ProbeActivity { id, max_chart_size } => {
            fetch::probe_activity(id, max_chart_size).await?;
        }
        Commands::Status => {
            println!("Garmin storage: {}", config.garmin_storage_path.display());
            println!("Model data: {}", config.data_dir.display());
            println!();

            type Loader = Box<dyn Fn(&config::Config) -> error::Result<polars::prelude::LazyFrame>>;
            let datasets: Vec<(&str, &str, Loader)> = vec![
                ("Daily Health", "date", Box::new(data::load_daily_health)),
                (
                    "Performance",
                    "date",
                    Box::new(data::load_performance_metrics),
                ),
                (
                    "Activities",
                    "start_time_local",
                    Box::new(data::load_activities),
                ),
                ("Weight", "date", Box::new(data::load_weight)),
                (
                    "Blood Pressure",
                    "date",
                    Box::new(data::load_blood_pressure),
                ),
            ];

            for (name, date_col, loader) in &datasets {
                match loader(&config) {
                    Ok(lf) => match data::summarize(lf, date_col) {
                        Ok(s) if s.row_count > 0 => {
                            let range = match (&s.min_date, &s.max_date) {
                                (Some(min), Some(max)) => format!("{} to {}", min, max),
                                _ => "n/a".to_string(),
                            };
                            println!("{name:<20} {:<8} rows    range: {}", s.row_count, range);
                        }
                        _ => println!("{name:<20} no data yet"),
                    },
                    Err(_) => println!("{name:<20} no data yet"),
                }
            }
        }
        Commands::Profile => {
            data::profile_data(&config)?;
        }
        Commands::Train { target } => {
            model::train(&config, &target)?;
        }
        Commands::Predict { target } => {
            model::predict(&config, &target)?;
        }
        Commands::Races => {
            races::run(&config)?;
        }
        Commands::Decompose => {
            decompose::run(&config)?;
        }
        Commands::InjuryRisk => {
            injury_risk::run(&config)?;
        }
        Commands::Fitness => {
            fitness::run(&config)?;
        }
        Commands::Drift => {
            fitness::drift(&config)?;
        }
    }

    Ok(())
}
