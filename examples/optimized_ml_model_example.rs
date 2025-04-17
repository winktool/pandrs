use pandrs::column::{Column, Float64Column, StringColumn};
use pandrs::error::Error as PandRSError;
use pandrs::ml::metrics::classification::{accuracy_score, precision_score, recall_score, f1_score};
use pandrs::ml::metrics::regression::{mean_squared_error, r2_score};
use pandrs::ml::models::{SupervisedModel, LinearRegression, LogisticRegression};
use pandrs::ml::models::model_persistence::ModelPersistence;
use pandrs::ml::pipeline::Pipeline;
use pandrs::ml::preprocessing::{StandardScaler, PolynomialFeatures};
use pandrs::OptimizedDataFrame;
use rand::Rng;

fn main() -> Result<(), PandRSError> {
    println!("Example of PandRS Model Training and Evaluation (OptimizedDataFrame version)");
    println!("==========================================");
    
    // Regression model example
    regression_example()?;
    
    // Classification model example
    classification_example()?;
    
    // Model selection and evaluation example
    model_selection_example()?;
    
    // Model persistence example
    model_persistence_example()?;
    
    Ok(())
}

// Regression model example
fn regression_example() -> Result<(), PandRSError> {
    println!("\n==== Regression Model Example ====");
    
    // Generate regression data
    let reg_df = create_regression_data()?;
    println!("Regression Data Sample:");
    println!("{:?}", reg_df);
    
    // Split into training and test data (mock implementation)
    let train_size = (reg_df.row_count() as f64 * 0.7) as usize;
    let test_size = reg_df.row_count() - train_size;
    println!("Training Data Size: {}, Test Data Size: {}", train_size, test_size);
    
    // Normally, the data would be split, but here we use the original data
    let train_df = &reg_df;
    let test_df = &reg_df;
    
    // List of features
    let features = vec!["feature1", "feature2", "feature3"];
    let target = "target";
    
    // Create and train a linear regression model
    let mut model = LinearRegression::new();
    model.fit(&train_df, target, &features)?;
    
    // Display coefficients and intercept
    println!("\nLinear Regression Model Results:");
    println!("Coefficients: {:?}", model.coefficients());
    println!("Intercept: {}", model.intercept());
    
    // Predict on test data
    let predictions = model.predict(&test_df)?;
    
    // Model evaluation
    // Convert column view to f64 array
    let y_true: Vec<f64> = {
        // First, get the target column
        let target_col = test_df.column(target)?;
        let float_col = target_col.as_float64().ok_or(PandRSError::Type("Expected float column".to_string()))?;
        
        (0..test_df.row_count())
            .filter_map(|i| float_col.get(i).ok().flatten())
            .collect()
    };
    
    let mse = mean_squared_error(&y_true, &predictions)?;
    let r2 = r2_score(&y_true, &predictions)?;
    
    println!("\nModel Evaluation:");
    println!("MSE: {}", mse);
    println!("R^2: {}", r2);
    
    // Feature engineering and regression using pipeline
    println!("\nRegression using Feature Engineering Pipeline:");
    
    let mut pipeline = Pipeline::new();
    
    // Add polynomial features
    pipeline.add_transformer(PolynomialFeatures::new(vec![features[0].to_string(), features[1].to_string()], 2, false));
    
    // Standardize features
    pipeline.add_transformer(StandardScaler::new(vec![
        features[0].to_string(), features[1].to_string(), features[2].to_string(),
        format!("{}_{}^2", features[0], features[0]),
        format!("{}_{}", features[0], features[1]),
        format!("{}_{}^2", features[1], features[1]),
    ]));
    
    // Apply pipeline
    let transformed_train_df = pipeline.fit_transform(&train_df)?;
    let transformed_test_df = pipeline.transform(&test_df)?;
    
    // Get new feature list
    let poly_features: Vec<&str> = transformed_train_df.column_names().iter()
        .filter(|&name| name != target)
        .map(|s| s.as_str())
        .collect();
    
    // Linear regression with transformed data
    let mut poly_model = LinearRegression::new();
    poly_model.fit(&transformed_train_df, target, &poly_features)?;
    
    // Predict on test data
    let poly_predictions = poly_model.predict(&transformed_test_df)?;
    
    // Model evaluation
    // Convert column view to f64 array
    let poly_y_true: Vec<f64> = {
        // First, get the target column
        let target_col = transformed_test_df.column(target)?;
        let float_col = target_col.as_float64().ok_or(PandRSError::Type("Expected float column".to_string()))?;
        
        (0..transformed_test_df.row_count())
            .filter_map(|i| float_col.get(i).ok().flatten())
            .collect()
    };
    
    let poly_mse = mean_squared_error(&poly_y_true, &poly_predictions)?;
    let poly_r2 = r2_score(&poly_y_true, &poly_predictions)?;
    
    println!("Evaluation of Linear Regression with Polynomial Features:");
    println!("MSE: {}", poly_mse);
    println!("R^2: {}", poly_r2);
    
    Ok(())
}

// Classification model example
fn classification_example() -> Result<(), PandRSError> {
    println!("\n==== Classification Model Example ====");
    
    // Generate classification data
    let cls_df = create_classification_data()?;
    println!("Classification Data Sample:");
    println!("{:?}", cls_df);
    
    // Split into training and test data (mock implementation)
    let train_size = (cls_df.row_count() as f64 * 0.7) as usize;
    let test_size = cls_df.row_count() - train_size;
    println!("Training Data Size: {}, Test Data Size: {}", train_size, test_size);
    
    // Normally, the data would be split, but here we use the original data
    let train_df = &cls_df;
    let test_df = &cls_df;
    
    // List of features
    let features = vec!["feature1", "feature2"];
    let target = "target";
    
    // Create and train a logistic regression model
    let mut model = LogisticRegression::new(0.1, 1000, 1e-5);
    model.fit(&train_df, target, &features)?;
    
    // Display coefficients and intercept
    println!("\nLogistic Regression Model Results:");
    println!("Coefficients: {:?}", model.coefficients());
    println!("Intercept: {}", model.intercept());
    
    // Predict on test data
    let predictions = model.predict(&test_df)?;
    
    // Model evaluation
    // Convert column view to bool array
    let y_true: Vec<bool> = {
        // First, get the target column
        let target_col = test_df.column(target)?;
        let string_col = target_col.as_string().ok_or(PandRSError::Type("Expected string column".to_string()))?;
        
        (0..test_df.row_count())
            .filter_map(|i| {
                string_col.get(i).ok().flatten().map(|s| s == "1")
            })
            .collect()
    };
    
    // Convert predictions from f64 to bool
    let pred_bool: Vec<bool> = predictions.iter().map(|&val| val > 0.5).collect();
    
    let accuracy = accuracy_score(&y_true, &pred_bool)?;
    let precision = precision_score(&y_true, &pred_bool)?;
    let recall = recall_score(&y_true, &pred_bool)?;
    let f1 = f1_score(&y_true, &pred_bool)?;
    
    println!("\nModel Evaluation:");
    println!("Accuracy: {}", accuracy);
    println!("Precision: {}", precision);
    println!("Recall: {}", recall);
    println!("F1 Score: {}", f1);
    
    // Probability predictions
    let proba_df = model.predict_proba(&test_df)?;
    println!("\nProbability Predictions Sample:");
    
    // Display only the first 5 rows
    println!("Probability Predictions (First 5 Rows):");
    for i in 0..proba_df.row_count().min(5) {
        // Get column data
        if let (Ok(col0), Ok(col1)) = (proba_df.column("prob_0"), proba_df.column("prob_1")) {
            // Convert to float64
            if let (Some(fcol0), Some(fcol1)) = (col0.as_float64(), col1.as_float64()) {
                // Get values for each row
                if let (Ok(Some(prob_0)), Ok(Some(prob_1))) = (fcol0.get(i), fcol1.get(i)) {
                    println!("Row {}: prob_0={:.4}, prob_1={:.4}", i, prob_0, prob_1);
                }
            }
        }
    }
    
    Ok(())
}

// Model selection and evaluation example
fn model_selection_example() -> Result<(), PandRSError> {
    println!("\n==== Model Selection and Evaluation Example ====");
    
    // Generate regression data
    let _reg_df = create_regression_data()?;
    
    // Model evaluation using cross-validation
    println!("\nCross-Validation (5-fold) Results:");
    println!("Note: LinearRegression does not implement the Clone trait, so cross-validation cannot be performed");
    
    // The following code is disabled because the Clone trait is required
    // let model = LinearRegression::new();
    // let _features = vec!["feature1", "feature2", "feature3"];
    // let _target = "target";
    // let cv_scores = cross_val_score(&model, &_reg_df, _target, &_features, 5)?;
    // println!("Scores for each fold: {:?}", cv_scores);
    // println!("Average Score: {}", cv_scores.iter().sum::<f64>() / cv_scores.len() as f64);
    
    Ok(())
}

// Model persistence example
fn model_persistence_example() -> Result<(), PandRSError> {
    println!("\n==== Model Persistence Example ====");
    
    // Generate regression data
    let reg_df = create_regression_data()?;
    
    // Use the original DataFrame directly
    let df_to_use = &reg_df;
    
    // List of features
    let features = vec!["feature1", "feature2", "feature3"];
    let target = "target";
    
    // Train the model
    let mut model = LinearRegression::new();
    model.fit(df_to_use, target, &features)?;
    
    // Save the model
    let model_path = "/tmp/linear_regression_model.json";
    model.save_model(model_path)?;
    println!("Model saved: {}", model_path);
    
    // Load the model
    let loaded_model = LinearRegression::load_model(model_path)?;
    println!("Model loaded");
    
    // Verify the parameters of the loaded model
    println!("Loaded Model Coefficients: {:?}", loaded_model.coefficients());
    println!("Loaded Model Intercept: {}", loaded_model.intercept());
    
    // Verify predictions
    let orig_pred = model.predict(df_to_use)?;
    let loaded_pred = loaded_model.predict(df_to_use)?;
    
    // Extract only the first 5 elements
    let orig_pred_sample: Vec<&f64> = orig_pred.iter().take(5).collect();
    let loaded_pred_sample: Vec<&f64> = loaded_pred.iter().take(5).collect();
    
    println!("Original Model Predictions: {:?}", orig_pred_sample);
    println!("Loaded Model Predictions: {:?}", loaded_pred_sample);
    
    Ok(())
}

// Generate regression data
fn create_regression_data() -> Result<OptimizedDataFrame, PandRSError> {
    let mut rng = rand::rng();
    
    // Generate 100 rows of data
    let n = 100;
    
    // 3 features
    let feature1: Vec<f64> = (0..n).map(|_| rng.random_range(-10.0..10.0)).collect();
    let feature2: Vec<f64> = (0..n).map(|_| rng.random_range(0.0..100.0)).collect();
    let feature3: Vec<f64> = (0..n).map(|_| rng.random_range(-5.0..15.0)).collect();
    
    // Target variable with a linear relationship: y = 2*x1 + 0.5*x2 - 1.5*x3 + noise
    let target: Vec<f64> = feature1
        .iter()
        .zip(feature2.iter())
        .zip(feature3.iter())
        .map(|((x1, x2), x3)| {
            2.0 * x1 + 0.5 * x2 - 1.5 * x3 + rng.random_range(-5.0..5.0)
        })
        .collect();
    
    // Create OptimizedDataFrame
    let mut df = OptimizedDataFrame::new();
    
    // Add features
    let feature1_col = Float64Column::with_name(feature1, "feature1");
    df.add_column("feature1", Column::Float64(feature1_col))?;
    
    let feature2_col = Float64Column::with_name(feature2, "feature2");
    df.add_column("feature2", Column::Float64(feature2_col))?;
    
    let feature3_col = Float64Column::with_name(feature3, "feature3");
    df.add_column("feature3", Column::Float64(feature3_col))?;
    
    // Add target variable
    let target_col = Float64Column::with_name(target, "target");
    df.add_column("target", Column::Float64(target_col))?;
    
    Ok(df)
}

// Generate classification data
fn create_classification_data() -> Result<OptimizedDataFrame, PandRSError> {
    let mut rng = rand::rng();
    
    // Generate 100 rows of data
    let n = 100;
    
    // 2 features
    let feature1: Vec<f64> = (0..n).map(|_| rng.random_range(-5.0..5.0)).collect();
    let feature2: Vec<f64> = (0..n).map(|_| rng.random_range(-5.0..5.0)).collect();
    
    // Binary classification using a logistic model
    // P(y=1) = sigmoid(1.5*x1 - 2*x2)
    let target: Vec<String> = feature1
        .iter()
        .zip(feature2.iter())
        .map(|(x1, x2)| {
            let z = 1.5 * x1 - 2.0 * x2;
            let p = 1.0 / (1.0 + (-z).exp());
            
            if rng.random::<f64>() < p {
                "1".to_string()
            } else {
                "0".to_string()
            }
        })
        .collect();
    
    // Create OptimizedDataFrame
    let mut df = OptimizedDataFrame::new();
    
    // Add features
    let feature1_col = Float64Column::with_name(feature1, "feature1");
    df.add_column("feature1", Column::Float64(feature1_col))?;
    
    let feature2_col = Float64Column::with_name(feature2, "feature2");
    df.add_column("feature2", Column::Float64(feature2_col))?;
    
    // Add target variable
    let target_col = StringColumn::with_name(target, "target");
    df.add_column("target", Column::String(target_col))?;
    
    Ok(df)
}