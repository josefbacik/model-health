mod config;
mod data;
mod error;
mod features;
mod model;
mod sync;

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
    /// Sync data from Garmin Connect
    Sync {
        /// Start date for backfill (YYYY-MM-DD)
        #[arg(long)]
        from: Option<NaiveDate>,
        /// End date (YYYY-MM-DD)
        #[arg(long)]
        to: Option<NaiveDate>,
        /// Force re-download of existing data
        #[arg(long)]
        force: bool,
    },
    /// Show sync status and data coverage
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
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    let config = config::Config::load()?;

    match cli.command {
        Commands::Sync { from, to, force } => {
            let stats = sync::run_sync(&config, from, to, force).await?;
            println!("{stats}");
        }
        Commands::Status => {
            println!("Config: {}", config::Config::config_path().display());
            println!("Garmin storage: {}", config.garmin_storage_path.display());
            println!("Model data: {}", config.data_dir.display());
            println!();
            // Quick summary
            match data::load_daily_health(&config) {
                Ok(lf) => match lf.collect() {
                    Ok(df) => println!("Daily health records: {}", df.height()),
                    Err(e) => println!("Daily health: error loading ({e})"),
                },
                Err(_) => println!("Daily health: no data yet"),
            }
            match data::load_activities(&config) {
                Ok(lf) => match lf.collect() {
                    Ok(df) => println!("Activity records: {}", df.height()),
                    Err(e) => println!("Activities: error loading ({e})"),
                },
                Err(_) => println!("Activities: no data yet"),
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
    }

    Ok(())
}
