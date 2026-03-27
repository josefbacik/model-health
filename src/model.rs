use std::path::Path;

use chrono::Utc;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use smartcore::ensemble::random_forest_regressor::{
    RandomForestRegressor, RandomForestRegressorParameters,
};
use smartcore::linalg::basic::matrix::DenseMatrix;
use tracing::info;

use crate::config::Config;
use crate::error::{AppError, Result};
use crate::features;

type RfModel = RandomForestRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>;

/// Metadata about a trained model.
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub target: String,
    pub feature_names: Vec<String>,
    pub n_training_rows: usize,
    pub date_range: (String, String),
    pub metrics: EvalMetrics,
    pub trained_at: String,
    pub hyperparams: Hyperparams,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hyperparams {
    pub n_trees: usize,
    pub max_depth: u16,
    pub min_samples_split: usize,
}

impl Default for Hyperparams {
    fn default() -> Self {
        Self {
            n_trees: 200,
            max_depth: 15,
            min_samples_split: 5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalMetrics {
    pub mae: f64,
    pub rmse: f64,
    pub r_squared: f64,
}

impl std::fmt::Display for EvalMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MAE: {:.2}, RMSE: {:.2}, R²: {:.3}",
            self.mae, self.rmse, self.r_squared
        )
    }
}

/// Convert a Polars DataFrame into a SmartCore DenseMatrix.
/// Extracts the given feature columns as f64 and builds a row-major matrix.
fn dataframe_to_matrix(df: &DataFrame, feature_cols: &[String]) -> Result<DenseMatrix<f64>> {
    let n_rows = df.height();
    let n_cols = feature_cols.len();

    // Build row-by-row as slices for from_2d_array
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n_rows);
    for row_idx in 0..n_rows {
        let mut row = Vec::with_capacity(n_cols);
        for col_name in feature_cols {
            let col = df
                .column(col_name.as_str())
                .map_err(|e| AppError::Model(format!("Column '{col_name}' not found: {e}")))?;
            let ca = col
                .f64()
                .map_err(|e| AppError::Model(format!("Column '{col_name}' is not f64: {e}")))?;
            let val = ca.get(row_idx).ok_or_else(|| {
                AppError::Model(format!("Null value at row {row_idx}, column '{col_name}'"))
            })?;
            row.push(val);
        }
        rows.push(row);
    }

    let row_refs: Vec<&[f64]> = rows.iter().map(|r| r.as_slice()).collect();
    DenseMatrix::from_2d_array(&row_refs)
        .map_err(|e| AppError::Model(format!("Failed to create matrix: {e}")))
}

/// Extract a target column as Vec<f64>.
fn column_to_vec(col: &Column) -> Result<Vec<f64>> {
    let ca = col
        .f64()
        .map_err(|e| AppError::Model(format!("Target is not f64: {e}")))?;
    Ok(ca.into_no_null_iter().collect())
}

/// Compute evaluation metrics for predictions vs actuals.
fn compute_metrics(actual: &[f64], predicted: &[f64]) -> EvalMetrics {
    let n = actual.len() as f64;
    let mean_actual: f64 = actual.iter().sum::<f64>() / n;

    let mae: f64 = actual
        .iter()
        .zip(predicted.iter())
        .map(|(a, p)| (a - p).abs())
        .sum::<f64>()
        / n;

    let mse: f64 = actual
        .iter()
        .zip(predicted.iter())
        .map(|(a, p)| (a - p).powi(2))
        .sum::<f64>()
        / n;

    let ss_res: f64 = actual
        .iter()
        .zip(predicted.iter())
        .map(|(a, p)| (a - p).powi(2))
        .sum();

    let ss_tot: f64 = actual.iter().map(|a| (a - mean_actual).powi(2)).sum();

    let r_squared = if ss_tot > 0.0 {
        1.0 - (ss_res / ss_tot)
    } else {
        0.0
    };

    EvalMetrics {
        mae,
        rmse: mse.sqrt(),
        r_squared,
    }
}

fn build_rf_params(params: &Hyperparams) -> RandomForestRegressorParameters {
    RandomForestRegressorParameters::default()
        .with_n_trees(params.n_trees)
        .with_max_depth(params.max_depth)
        .with_min_samples_split(params.min_samples_split)
}

/// Time-series cross-validation with expanding window.
fn time_series_cv(
    df: &DataFrame,
    feature_cols: &[String],
    target_col: &str,
    params: &Hyperparams,
    min_train_days: usize,
    test_window: usize,
) -> Result<EvalMetrics> {
    let n = df.height();
    let mut all_actual = Vec::new();
    let mut all_predicted = Vec::new();

    let mut train_end = min_train_days;
    while train_end + test_window <= n {
        let test_end = (train_end + test_window).min(n);

        let train_df = df.slice(0, train_end);
        let test_df = df.slice(train_end as i64, test_end - train_end);

        let train_x = dataframe_to_matrix(&train_df, feature_cols)?;
        let train_y = column_to_vec(train_df.column(target_col)?)?;
        let test_x = dataframe_to_matrix(&test_df, feature_cols)?;
        let test_y = column_to_vec(test_df.column(target_col)?)?;

        let model = RandomForestRegressor::fit(&train_x, &train_y, build_rf_params(params))
            .map_err(|e| AppError::Model(format!("Training failed: {e}")))?;

        let preds = model
            .predict(&test_x)
            .map_err(|e| AppError::Model(format!("Prediction failed: {e}")))?;

        all_actual.extend_from_slice(&test_y);
        all_predicted.extend_from_slice(&preds);

        train_end += test_window;
    }

    if all_actual.is_empty() {
        return Err(AppError::Model(
            "Not enough data for cross-validation".to_string(),
        ));
    }

    let metrics = compute_metrics(&all_actual, &all_predicted);
    info!(
        folds = (n - min_train_days) / test_window,
        total_test_samples = all_actual.len(),
        %metrics,
        "Cross-validation complete"
    );

    Ok(metrics)
}

/// Train a Random Forest model and save it to disk.
pub fn train(config: &Config, target: &str) -> Result<()> {
    crate::data::validate_training_data(config)?;

    let (df, feature_names) = features::build_feature_matrix(config, target)?;
    let n_rows = df.height();
    info!(
        rows = n_rows,
        features = feature_names.len(),
        "Training data ready"
    );

    let params = Hyperparams::default();

    println!("Running time-series cross-validation...");
    let cv_metrics = time_series_cv(&df, &feature_names, target, &params, 30, 7)?;
    println!("Cross-validation results: {cv_metrics}");

    println!("Training final model on all {n_rows} rows...");
    let x = dataframe_to_matrix(&df, &feature_names)?;
    let y = column_to_vec(df.column(target)?)?;

    let model = RandomForestRegressor::fit(&x, &y, build_rf_params(&params))
        .map_err(|e| AppError::Model(format!("Training failed: {e}")))?;

    // Get date range
    let dates = df.column("date")?;
    let min_date = dates
        .min_reduce()
        .map_err(|e| AppError::Model(e.to_string()))?;
    let max_date = dates
        .max_reduce()
        .map_err(|e| AppError::Model(e.to_string()))?;

    let metadata = ModelMetadata {
        target: target.to_string(),
        feature_names: feature_names.clone(),
        n_training_rows: n_rows,
        date_range: (
            format!("{:?}", min_date.value()),
            format!("{:?}", max_date.value()),
        ),
        metrics: cv_metrics,
        trained_at: Utc::now().to_rfc3339(),
        hyperparams: params,
    };

    let model_dir = config.models_dir().join(target);
    save_model(&model, &metadata, &model_dir)?;

    println!("Model saved to {}", model_dir.display());
    Ok(())
}

/// Run prediction using the latest trained model.
pub fn predict(config: &Config, target: &str) -> Result<()> {
    let model_dir = config.models_dir().join(target);
    let (model, metadata) = load_model(&model_dir)?;

    let (df, _feature_names) = features::build_feature_matrix(config, target)?;

    if df.height() == 0 {
        return Err(AppError::Model(
            "No data available for prediction".to_string(),
        ));
    }

    let last_row = df.tail(Some(1));
    let x = dataframe_to_matrix(&last_row, &metadata.feature_names)?;

    let prediction = model
        .predict(&x)
        .map_err(|e| AppError::Model(format!("Prediction failed: {e}")))?;

    let date = last_row
        .column("date")?
        .get(0)
        .map_err(|e| AppError::Model(e.to_string()))?;

    println!(
        "Prediction for {} -> {}: {:.1} (model MAE: {:.1})",
        date, metadata.target, prediction[0], metadata.metrics.mae
    );

    Ok(())
}

fn save_model(model: &RfModel, metadata: &ModelMetadata, dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;

    let model_path = dir.join("latest.json");
    let model_json = serde_json::to_string(model)?;
    std::fs::write(&model_path, model_json)?;

    let meta_path = dir.join("latest_meta.json");
    let meta_json = serde_json::to_string_pretty(metadata)?;
    std::fs::write(&meta_path, meta_json)?;

    Ok(())
}

fn load_model(dir: &Path) -> Result<(RfModel, ModelMetadata)> {
    let model_path = dir.join("latest.json");
    if !model_path.exists() {
        return Err(AppError::Model(format!(
            "No trained model found at {}. Run `model-health train` first.",
            dir.display()
        )));
    }

    let model_json = std::fs::read_to_string(&model_path)?;
    let model: RfModel = serde_json::from_str(&model_json)?;

    let meta_path = dir.join("latest_meta.json");
    let meta_json = std::fs::read_to_string(&meta_path)?;
    let metadata: ModelMetadata = serde_json::from_str(&meta_json)?;

    Ok((model, metadata))
}
