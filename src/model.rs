use std::path::Path;

use chrono::{Duration, NaiveDate, Utc};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use smartcore::ensemble::random_forest_regressor::{
    RandomForestRegressor, RandomForestRegressorParameters,
};
use smartcore::linalg::basic::matrix::DenseMatrix;
use tracing::{info, warn};

use crate::config::Config;
use crate::error::{AppError, Result};
use crate::features;

type RfModel = RandomForestRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>;

/// Fixed seed for permutation importance shuffles (reproducibility).
const PERMUTATION_SEED: u64 = 0xC0FFEE_u64;

/// Metadata about a trained model.
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub target: String,
    pub feature_names: Vec<String>,
    pub n_training_rows: usize,
    pub date_range: (String, String),
    pub cv_results: CvResults,
    pub feature_importance: Vec<(String, f64)>,
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

/// Results of time-series cross-validation for the model plus baselines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CvResults {
    pub model: EvalMetrics,
    /// Persistence (naive) baseline: predict tomorrow = today's value of the
    /// source column. `None` if the source column isn't in the feature list.
    pub persistence: Option<EvalMetrics>,
    /// Mean baseline: predict the training-fold target mean.
    pub mean: EvalMetrics,
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

/// Extract a single named column from a DataFrame as Vec<f64>.
fn df_column_as_f64(df: &DataFrame, name: &str) -> Result<Vec<f64>> {
    column_to_vec(df.column(name)?)
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

/// Infer the persistence-baseline source column name from a target name,
/// by stripping the `next_day_` prefix. Returns `None` if the prefix is absent.
fn persistence_source_for(target: &str) -> Option<&str> {
    target.strip_prefix("next_day_")
}

/// Time-series cross-validation with expanding window. Also computes the
/// persistence and mean baselines over the same folds.
fn time_series_cv(
    df: &DataFrame,
    feature_cols: &[String],
    target_col: &str,
    params: &Hyperparams,
    min_train_days: usize,
    test_window: usize,
) -> Result<CvResults> {
    let n = df.height();
    let mut all_actual = Vec::new();
    let mut all_predicted = Vec::new();
    let mut all_mean_pred = Vec::new();
    let mut all_persistence_pred: Vec<f64> = Vec::new();

    // Determine persistence source column. If the inferred name isn't in the
    // feature list we just skip the persistence baseline.
    let persistence_src = persistence_source_for(target_col).and_then(|src| {
        if feature_cols.iter().any(|f| f == src) {
            Some(src.to_string())
        } else {
            warn!(
                target = target_col,
                source = src,
                "Persistence baseline source column not in feature list; skipping"
            );
            None
        }
    });

    let mut train_end = min_train_days;
    let mut folds = 0usize;
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

        // Mean baseline: predict the training fold's target mean for each test row.
        let train_mean = if train_y.is_empty() {
            0.0
        } else {
            train_y.iter().sum::<f64>() / train_y.len() as f64
        };
        all_mean_pred.extend(std::iter::repeat_n(train_mean, test_y.len()));

        // Persistence baseline: use today's value of the source column as the
        // prediction for tomorrow's target. Row N's source column holds
        // today's value and its target column holds tomorrow's actual
        // (features.rs shifts the target by -1).
        if let Some(src) = &persistence_src {
            let src_vals = df_column_as_f64(&test_df, src)?;
            all_persistence_pred.extend_from_slice(&src_vals);
        }

        all_actual.extend_from_slice(&test_y);
        all_predicted.extend_from_slice(&preds);

        train_end += test_window;
        folds += 1;
    }

    if all_actual.is_empty() {
        return Err(AppError::Model(
            "Not enough data for cross-validation".to_string(),
        ));
    }

    let model_metrics = compute_metrics(&all_actual, &all_predicted);
    let mean_metrics = compute_metrics(&all_actual, &all_mean_pred);
    let persistence_metrics = if persistence_src.is_some() {
        Some(compute_metrics(&all_actual, &all_persistence_pred))
    } else {
        None
    };

    info!(
        folds,
        total_test_samples = all_actual.len(),
        %model_metrics,
        "Cross-validation complete"
    );

    Ok(CvResults {
        model: model_metrics,
        persistence: persistence_metrics,
        mean: mean_metrics,
    })
}

/// Simple deterministic LCG for reproducible shuffles without adding a
/// dependency on the `rand` crate.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        // XOR with a non-zero magic constant (the 64-bit golden-ratio
        // increment from SplitMix) so that a caller passing seed=0 doesn't
        // land on the LCG's zero fixed point.
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        // Constants from Numerical Recipes.
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Uniform integer in [0, n). n must be > 0.
    fn gen_range(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Fisher-Yates shuffle using the deterministic LCG.
fn shuffle_in_place<T>(slice: &mut [T], rng: &mut Lcg) {
    let n = slice.len();
    if n < 2 {
        return;
    }
    for i in (1..n).rev() {
        let j = rng.gen_range(i + 1);
        slice.swap(i, j);
    }
}

/// Compute permutation feature importance on a held-out slice. For each
/// feature, the column values are shuffled and the MAE is re-measured; the
/// importance is `shuffled_mae - baseline_mae` (larger = more important).
fn permutation_importance(
    model: &RfModel,
    eval_df: &DataFrame,
    feature_cols: &[String],
    target_col: &str,
) -> Result<Vec<(String, f64)>> {
    let baseline_x = dataframe_to_matrix(eval_df, feature_cols)?;
    let y_true = column_to_vec(eval_df.column(target_col)?)?;

    let baseline_preds = model
        .predict(&baseline_x)
        .map_err(|e| AppError::Model(format!("Prediction failed: {e}")))?;
    let baseline_mae = compute_metrics(&y_true, &baseline_preds).mae;

    // Extract feature data as columns once up front.
    let n_rows = eval_df.height();
    let mut columns: Vec<Vec<f64>> = Vec::with_capacity(feature_cols.len());
    for name in feature_cols {
        columns.push(df_column_as_f64(eval_df, name)?);
    }

    let mut rng = Lcg::new(PERMUTATION_SEED);
    let mut importances: Vec<(String, f64)> = Vec::with_capacity(feature_cols.len());

    for (feat_idx, feat_name) in feature_cols.iter().enumerate() {
        // Build a shuffled version of this one column.
        let mut shuffled = columns[feat_idx].clone();
        shuffle_in_place(&mut shuffled, &mut rng);

        // Rebuild the row-major matrix with only this column replaced.
        let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n_rows);
        for row_idx in 0..n_rows {
            let mut row = Vec::with_capacity(feature_cols.len());
            for (col_idx, col_vals) in columns.iter().enumerate() {
                if col_idx == feat_idx {
                    row.push(shuffled[row_idx]);
                } else {
                    row.push(col_vals[row_idx]);
                }
            }
            rows.push(row);
        }
        let row_refs: Vec<&[f64]> = rows.iter().map(|r| r.as_slice()).collect();
        let shuffled_x = DenseMatrix::from_2d_array(&row_refs)
            .map_err(|e| AppError::Model(format!("Failed to build matrix: {e}")))?;

        let preds = model
            .predict(&shuffled_x)
            .map_err(|e| AppError::Model(format!("Prediction failed: {e}")))?;
        let shuffled_mae = compute_metrics(&y_true, &preds).mae;

        importances.push((feat_name.clone(), shuffled_mae - baseline_mae));
    }

    importances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(importances)
}

/// Extract a Polars Date column value at a given row as NaiveDate.
/// Polars stores Date values as days-since-unix-epoch (1970-01-01),
/// and `NaiveDate::from_num_days_from_ce_opt` wants days-since-CE, so we
/// offset by 719_163.
fn date_at(df: &DataFrame, row: usize) -> Result<NaiveDate> {
    let col = df.column("date")?;
    let av = col
        .get(row)
        .map_err(|e| AppError::Model(format!("Failed to read date: {e}")))?;
    match av {
        AnyValue::Date(days) => NaiveDate::from_num_days_from_ce_opt(days + 719_163)
            .ok_or_else(|| AppError::Model(format!("Invalid date value: {days}"))),
        other => Err(AppError::Model(format!(
            "Expected Date column, got {other:?}"
        ))),
    }
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
    let cv_results = time_series_cv(&df, &feature_names, target, &params, 30, 7)?;
    println!("Cross-validation results:");
    println!("  Model:       {}", cv_results.model);
    match &cv_results.persistence {
        Some(p) => println!("  Persistence: {p}"),
        None => println!("  Persistence: (skipped; source column not in features)"),
    }
    println!("  Mean:        {}", cv_results.mean);

    println!("Training final model on all {n_rows} rows...");
    let x = dataframe_to_matrix(&df, &feature_names)?;
    let y = column_to_vec(df.column(target)?)?;

    let model = RandomForestRegressor::fit(&x, &y, build_rf_params(&params))
        .map_err(|e| AppError::Model(format!("Training failed: {e}")))?;

    // Permutation importance on the last 20% of the data.
    let importance_start = ((n_rows as f64) * 0.8) as i64;
    let importance_len = n_rows.saturating_sub(importance_start as usize);
    let feature_importance = if importance_len >= 2 {
        let eval_df = df.slice(importance_start, importance_len);
        println!(
            "Computing permutation importance on {} held-out rows...",
            eval_df.height()
        );
        let importances = permutation_importance(&model, &eval_df, &feature_names, target)?;
        println!("Top features (MAE increase when shuffled):");
        for (name, score) in importances.iter().take(10) {
            println!("  {name:<30} {score:+.4}");
        }
        importances
    } else {
        warn!("Not enough rows for permutation importance; skipping");
        Vec::new()
    };

    // Get date range as ISO YYYY-MM-DD. The feature matrix is sorted by date
    // in features::build_feature_matrix, so row 0 is the earliest.
    let min_date = date_at(&df, 0)?;
    let max_date = date_at(&df, n_rows - 1)?;

    let metadata = ModelMetadata {
        target: target.to_string(),
        feature_names: feature_names.clone(),
        n_training_rows: n_rows,
        date_range: (min_date.to_string(), max_date.to_string()),
        cv_results,
        feature_importance,
        trained_at: Utc::now().to_rfc3339(),
        hyperparams: params,
    };

    let model_dir = config.models_dir().join(target);
    save_model(&model, &metadata, &model_dir)?;

    println!("Model saved to {}", model_dir.display());
    Ok(())
}

/// Run prediction using the latest trained model. Forecasts the day AFTER
/// the most recent row of features.
pub fn predict(config: &Config, target: &str) -> Result<()> {
    let model_dir = config.models_dir().join(target);
    let (model, metadata) = load_model(&model_dir)?;

    // build_prediction_features returns the latest row of features without
    // any target column — the row whose answer isn't yet known, i.e. the
    // input for tomorrow's forecast. Pass the trained model's feature list
    // so prediction stays aligned with whatever survived training-time
    // pruning.
    let (df, _feature_names) =
        features::build_prediction_features(config, &metadata.feature_names)?;

    if df.height() == 0 {
        return Err(AppError::Model(
            "No data available for prediction".to_string(),
        ));
    }

    let x = dataframe_to_matrix(&df, &metadata.feature_names)?;

    let prediction = model
        .predict(&x)
        .map_err(|e| AppError::Model(format!("Prediction failed: {e}")))?;

    let feature_date = date_at(&df, df.height() - 1)?;
    let forecast_date = feature_date + Duration::days(1);

    println!(
        "Forecast for {} (using features from {}) -> {}: {:.1} (model CV MAE: {:.2})",
        forecast_date, feature_date, metadata.target, prediction[0], metadata.cv_results.model.mae
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
