use pandrs::error::Error;
use pandrs::{BooleanColumn, Column, Float64Column, Int64Column, OptimizedDataFrame, StringColumn};

#[test]
fn test_optimized_dataframe_creation() {
    // Create empty DataFrame
    let df = OptimizedDataFrame::new();
    assert_eq!(df.column_count(), 0);
    assert_eq!(df.row_count(), 0);
    assert!(df.column_names().is_empty());
}

#[test]
fn test_optimized_dataframe_add_column() {
    // Add column to DataFrame
    let mut df = OptimizedDataFrame::new();
    let values = Int64Column::new(vec![10, 20, 30]);

    // Add column
    df.add_column("values", Column::Int64(values)).unwrap();

    // Validation
    assert_eq!(df.column_count(), 1);
    assert_eq!(df.row_count(), 3);
    assert_eq!(df.column_names(), &["values"]);
}

#[test]
fn test_optimized_dataframe_add_multiple_columns() {
    // Create DataFrame with multiple columns
    let mut df = OptimizedDataFrame::new();

    let ages = Int64Column::new(vec![25, 30, 35]);
    let heights = Int64Column::new(vec![170, 180, 175]);

    df.add_column("age", Column::Int64(ages)).unwrap();
    df.add_column("height", Column::Int64(heights)).unwrap();

    // Validation
    assert_eq!(df.column_count(), 2);
    assert_eq!(df.row_count(), 3);
    assert!(df.contains_column("age"));
    assert!(df.contains_column("height"));
    assert!(!df.contains_column("weight"));
}

#[test]
fn test_optimized_dataframe_column_length_mismatch() {
    // Test error when adding columns with different lengths
    let mut df = OptimizedDataFrame::new();

    let ages = Int64Column::new(vec![25, 30, 35]);
    df.add_column("age", Column::Int64(ages)).unwrap();

    // Add column with different length
    let heights = Int64Column::new(vec![170, 180]);
    let result = df.add_column("height", Column::Int64(heights));

    // Should error
    assert!(result.is_err());

    // Check specific error type
    match result {
        Err(Error::InconsistentRowCount { .. }) => (),
        _ => panic!("Expected an InconsistentRowCount error"),
    }
}

#[test]
fn test_optimized_dataframe_duplicate_column() {
    // Test error when adding duplicate column name
    let mut df = OptimizedDataFrame::new();

    let ages1 = Int64Column::new(vec![25, 30, 35]);
    df.add_column("age", Column::Int64(ages1)).unwrap();

    // Add column with same name
    let ages2 = Int64Column::new(vec![40, 45, 50]);
    let result = df.add_column("age", Column::Int64(ages2));

    // Should error
    assert!(result.is_err());

    // Check specific error type
    match result {
        Err(Error::DuplicateColumnName(_)) => (),
        _ => panic!("Expected a DuplicateColumnName error"),
    }
}

#[test]
fn test_optimized_dataframe_mixed_types() {
    let mut df = OptimizedDataFrame::new();

    // Add columns of different types
    let int_col = Int64Column::new(vec![1, 2, 3]);
    let float_col = Float64Column::new(vec![1.1, 2.2, 3.3]);
    let str_col = StringColumn::new(vec![
        "one".to_string(),
        "two".to_string(),
        "three".to_string(),
    ]);
    let bool_col = BooleanColumn::new(vec![true, false, true]);

    df.add_column("int", Column::Int64(int_col)).unwrap();
    df.add_column("float", Column::Float64(float_col)).unwrap();
    df.add_column("str", Column::String(str_col)).unwrap();
    df.add_column("bool", Column::Boolean(bool_col)).unwrap();

    // Validation
    assert_eq!(df.column_count(), 4);
    assert_eq!(df.row_count(), 3);

    // Check column types
    let int_view = df.column("int").unwrap();
    let float_view = df.column("float").unwrap();
    let str_view = df.column("str").unwrap();
    let bool_view = df.column("bool").unwrap();

    assert!(int_view.as_int64().is_some());
    assert!(float_view.as_float64().is_some());
    assert!(str_view.as_string().is_some());
    assert!(bool_view.as_boolean().is_some());
}

#[test]
fn test_optimized_dataframe_select_columns() {
    let mut df = OptimizedDataFrame::new();

    let id_col = Int64Column::new(vec![1, 2, 3, 4]);
    let name_col = StringColumn::new(vec![
        "Alice".to_string(),
        "Bob".to_string(),
        "Charlie".to_string(),
        "Dave".to_string(),
    ]);
    let age_col = Int64Column::new(vec![25, 30, 35, 40]);

    df.add_column("id", Column::Int64(id_col)).unwrap();
    df.add_column("name", Column::String(name_col)).unwrap();
    df.add_column("age", Column::Int64(age_col)).unwrap();

    // Select columns to create new DataFrame
    let selected = df.select(&["id", "name"]).unwrap();

    // Validation
    assert_eq!(selected.column_count(), 2);
    assert_eq!(selected.row_count(), 4);
    assert!(selected.contains_column("id"));
    assert!(selected.contains_column("name"));
    assert!(!selected.contains_column("age"));

    // When selecting a non-existent column
    let result = df.select(&["id", "nonexistent"]);
    assert!(result.is_err());
}

#[test]
fn test_optimized_dataframe_filter() {
    let mut df = OptimizedDataFrame::new();

    let id_col = Int64Column::new(vec![1, 2, 3, 4]);
    let filter_col = BooleanColumn::new(vec![true, false, true, false]);

    df.add_column("id", Column::Int64(id_col)).unwrap();
    df.add_column("filter", Column::Boolean(filter_col))
        .unwrap();

    // Filter using boolean column
    let filtered = df.filter("filter").unwrap();

    // Validation
    assert_eq!(filtered.row_count(), 2);

    // When filter column doesn't exist
    let result = df.filter("nonexistent");
    assert!(result.is_err());

    // When filter column is not boolean
    let mut df2 = OptimizedDataFrame::new();
    let id_col = Int64Column::new(vec![1, 2, 3, 4]);
    df2.add_column("id", Column::Int64(id_col)).unwrap();

    let result = df2.filter("id");
    assert!(result.is_err());
}
