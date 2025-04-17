use pandrs::{DataFrame, Series, PandRSError};
use std::fs::File;
use std::io::Read;
use std::path::Path;

// Test for CSV file operations (using temporary files)
#[test]
fn test_csv_io() -> Result<(), PandRSError> {
    // Temporary file path for testing
    let temp_path = Path::new("temp_test.csv");

    // Create test DataFrame
    let mut df = DataFrame::new();
    let names = Series::new(
        vec![
            "Alice".to_string(),
            "Bob".to_string(),
            "Charlie".to_string(),
        ],
        Some("name".to_string()),
    )?;
    let ages = Series::new(vec![30, 25, 35], Some("age".to_string()))?;

    df.add_column("name".to_string(), names)?;
    df.add_column("age".to_string(), ages)?;

    // Write to CSV
    let write_result = df.to_csv(&temp_path);

    // For cleaning up temporary file after test
    struct CleanupGuard<'a>(&'a Path);
    impl<'a> Drop for CleanupGuard<'a> {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(self.0);
        }
    }
    let _guard = CleanupGuard(temp_path);

    // Confirm write success
    assert!(write_result.is_ok());

    // Confirm file exists
    assert!(temp_path.exists());

    // Check file contents
    let mut file = File::open(temp_path).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();

    // Check CSV header and content
    let lines: Vec<&str> = contents.lines().collect();
    assert!(lines.len() >= 1); // At least has a header row

    // Minimal check: verify header contains column names
    assert!(lines[0].contains("name"));
    assert!(lines[0].contains("age"));

    // Also test CSV reading
    // Specify that from_csv has a header
    let df_from_csv = DataFrame::from_csv(&temp_path, true)?;
    
    // Verify loaded DataFrame
    assert_eq!(df_from_csv.column_names().len(), 2, "Column count should match");
    assert!(df_from_csv.contains_column("name"), "name column should exist");
    assert!(df_from_csv.contains_column("age"), "age column should exist");
    
    let row_count = df_from_csv.row_count();
    assert_eq!(row_count, 3, "Row count should match original data");
    
    // Check name column values - using DataFrame::get_column_string_values
    let name_values = df_from_csv.get_column_string_values("name")?;
    assert!(name_values[0].contains("Alice"), "First row name column value should be correct");
    assert!(name_values[1].contains("Bob"), "Second row name column value should be correct");
    assert!(name_values[2].contains("Charlie"), "Third row name column value should be correct");
    
    // Check age column values - using DataFrame::get_column_string_values to check string content
    let age_str_values = df_from_csv.get_column_string_values("age")?;
    assert!(age_str_values[0].contains("30"), "First row age column value should be correct");
    assert!(age_str_values[1].contains("25"), "Second row age column value should be correct");
    assert!(age_str_values[2].contains("35"), "Third row age column value should be correct");
    
    Ok(())
}

// Test for JSON file operations (still in implementation)
#[test]
fn test_json_io() {
    // Since JSON I/O functionality is not fully implemented yet,
    // only perform simple structure checks

    use pandrs::io::json::JsonOrient;

    // Verify that record format and column format are defined
    let _record_orient = JsonOrient::Records;
    let _column_orient = JsonOrient::Columns;

    // JSON I/O tests will be added here in the future
}
