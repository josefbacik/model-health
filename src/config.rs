use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::{AppError, Result};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_garmin_storage")]
    pub garmin_storage_path: PathBuf,

    #[serde(default = "default_data_dir")]
    pub data_dir: PathBuf,

    #[serde(default = "default_target")]
    pub default_target: String,

    #[serde(default = "default_min_days")]
    pub min_training_days: usize,
}

fn default_garmin_storage() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("garmin")
}

fn default_data_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("model-health")
}

fn default_target() -> String {
    "next_day_resting_hr".to_string()
}

fn default_min_days() -> usize {
    60
}

impl Default for Config {
    fn default() -> Self {
        Self {
            garmin_storage_path: default_garmin_storage(),
            data_dir: default_data_dir(),
            default_target: default_target(),
            min_training_days: default_min_days(),
        }
    }
}

impl Config {
    pub fn load() -> Result<Self> {
        let config_path = Self::config_path();
        if config_path.exists() {
            let contents = std::fs::read_to_string(&config_path)?;
            let config: Config = toml::from_str(&contents)?;
            Ok(config)
        } else {
            Ok(Config::default())
        }
    }

    #[allow(dead_code)]
    pub fn save(&self) -> Result<()> {
        let config_path = Self::config_path();
        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let contents = toml::to_string_pretty(self).map_err(|e| AppError::Config(e.to_string()))?;
        std::fs::write(&config_path, contents)?;
        Ok(())
    }

    pub fn config_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("model-health")
            .join("config.toml")
    }

    pub fn models_dir(&self) -> PathBuf {
        self.data_dir.join("models")
    }
}
