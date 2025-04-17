// Minimal sample example (OptimizedDataFrame version)
use pandrs::{OptimizedDataFrame, Column, Float64Column, Int64Column};
use pandrs::column::ColumnTrait; // Add ColumnTrait import
use pandrs::error::Error;
use pandrs::ml::preprocessing::{StandardScaler, MinMaxScaler};
use pandrs::ml::pipeline::{Pipeline, Transformer};

fn main() -> Result<(), Error> {
    println!("PandRS Machine Learning Module Basic Example (OptimizedDataFrame version)");
    println!("===================================================");

    // Create sample data
    let mut df = OptimizedDataFrame::new();

    // Feature 1: Float64 type
    let feature1 = Float64Column::with_name(vec![1.0, 2.0, 3.0, 4.0, 5.0], "feature1");
    df.add_column("feature1", Column::Float64(feature1))?;

    // Feature 2: Float64 type
    let feature2 = Float64Column::with_name(vec![10.0, 20.0, 30.0, 40.0, 50.0], "feature2");
    df.add_column("feature2", Column::Float64(feature2))?;

    // Feature 3: Int64 type
    let feature3 = Int64Column::with_name(vec![100, 200, 300, 400, 500], "feature3");
    df.add_column("feature3", Column::Int64(feature3))?;

    println!("Original DataFrame:");
    println!("{:?}", df);

    // Check basic information
    println!("\nBasic Information:");
    println!("Number of rows: {}", df.row_count());
    println!("Number of columns: {}", df.column_count());
    println!("Column names: {:?}", df.column_names());

    // Try using StandardScaler
    println!("\n=== Applying StandardScaler ===");
    let mut scaler = StandardScaler::new(vec!["feature1".to_string(), "feature2".to_string()]);
    let scaled_df = scaler.fit_transform(&df)?;
    println!("Standardized DataFrame:");
    println!("{:?}", scaled_df);

    // Check the values of the feature1 column
    if let Ok(col) = scaled_df.column("feature1") {
        if let Some(float_col) = col.as_float64() {
            println!("\nStandardized values of the feature1 column:");
            for i in 0..float_col.len() {
                if let Ok(val) = float_col.get(i) {
                    println!("Row {}: {:?}", i, val);
                }
            }
        }
    }

    // Try using MinMaxScaler
    println!("\n=== Applying MinMaxScaler ===");
    let mut minmax = MinMaxScaler::new(vec!["feature1".to_string(), "feature2".to_string()], (0.0, 1.0));
    let normalized_df = minmax.fit_transform(&df)?;
    println!("Normalized DataFrame:");
    println!("{:?}", normalized_df);

    // Check the values of the feature1 column
    if let Ok(col) = normalized_df.column("feature1") {
        if let Some(float_col) = col.as_float64() {
            println!("\nNormalized values of the feature1 column:");
            for i in 0..float_col.len() {
                if let Ok(val) = float_col.get(i) {
                    println!("Row {}: {:?}", i, val);
                }
            }
        }
    }

    // Try using a pipeline
    println!("\n=== Using a Pipeline ===");
    let mut pipeline = Pipeline::new();
    pipeline.add_transformer(StandardScaler::new(vec!["feature1".to_string()]))
           .add_transformer(MinMaxScaler::new(vec!["feature2".to_string()], (0.0, 1.0)));

    let result_df = pipeline.fit_transform(&df)?;
    println!("DataFrame after applying the pipeline:");
    println!("{:?}", result_df);

    println!("\n === Sample Execution Complete ===");
    Ok(())
}