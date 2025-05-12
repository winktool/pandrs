use pandrs::error::Result;
use pandrs::{DataFrame, Series};

fn main() -> Result<()> {
    println!("=== PandRS Basic Usage Example ===");

    // Creating Series
    let ages = Series::new(vec![30, 25, 40], Some("age".to_string()))?;
    let heights = Series::new(vec![180, 175, 182], Some("height".to_string()))?;
    let names = Series::new(
        vec![
            "Alice".to_string(),
            "Bob".to_string(),
            "Charlie".to_string(),
        ],
        Some("name".to_string()),
    )?;

    println!("Age Series: {:?}", ages);
    println!("Height Series: {:?}", heights);
    println!("Name Series: {:?}", names);

    // Statistics for numeric series
    println!("\n=== Statistics for Age Series ===");
    println!("Sum: {}", ages.sum());
    println!("Mean: {}", ages.mean()?);
    println!("Min: {}", ages.min()?);
    println!("Max: {}", ages.max()?);

    // Creating a DataFrame
    println!("\n=== Creating a DataFrame ===");
    let mut df = DataFrame::new();
    df.add_column("name".to_string(), names)?;
    df.add_column("age".to_string(), ages)?;
    df.add_column("height".to_string(), heights)?;

    println!("DataFrame: {:?}", df);
    println!("Number of Columns: {}", df.column_count());
    println!("Number of Rows: {}", df.row_count());
    println!("Column Names: {:?}", df.column_names());

    // Testing saving to and loading from CSV
    let file_path = "example_data.csv";
    df.to_csv(file_path)?;
    println!("\nSaved to CSV file: {}", file_path);

    // Testing loading from CSV (may not be fully implemented yet)
    match DataFrame::from_csv(file_path, true) {
        Ok(loaded_df) => {
            println!("DataFrame loaded from CSV: {:?}", loaded_df);
            println!("Number of Columns: {}", loaded_df.column_count());
            println!("Number of Rows: {}", loaded_df.row_count());
            println!("Column Names: {:?}", loaded_df.column_names());
        }
        Err(e) => {
            println!("Failed to load CSV: {:?}", e);
        }
    }

    println!("\n=== Sample Complete ===");
    Ok(())
}
