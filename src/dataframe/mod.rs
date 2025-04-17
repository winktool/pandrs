mod join;
pub mod apply;
mod categorical;
mod transform;

pub use apply::Axis;
pub use transform::{MeltOptions, StackOptions, UnstackOptions};

use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::path::Path;

use crate::error::{PandRSError, Result};
use crate::index::{DataFrameIndex, Index, IndexTrait, RangeIndex};
use crate::io;
use crate::series::Series;

/// Trait representing a data value
/// Base trait for type erasure
pub trait DataValue: Debug + Any + Send + Sync {
    /// Create a clone of the value
    fn box_clone(&self) -> Box<dyn DataValue>;

    /// Downcast to Any trait
    fn as_any(&self) -> &dyn Any;
}

impl<T: 'static + Debug + Clone + Send + Sync> DataValue for T {
    fn box_clone(&self) -> Box<dyn DataValue> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Wrapper type for storing DataValue
#[derive(Debug)]
pub struct DataBox(pub Box<dyn DataValue>);

impl Clone for DataBox {
    fn clone(&self) -> Self {
        DataBox(self.0.box_clone())
    }
}

/// DataFrame struct: Column-oriented 2D data structure
#[derive(Debug, Clone)]
pub struct DataFrame {
    /// Mapping from column names to column data
    columns: HashMap<String, Series<DataBox>>,

    /// Column order preservation
    column_names: Vec<String>,

    /// Row index
    index: DataFrameIndex<String>,
}

impl DataFrame {
    /// Create a new empty DataFrame
    pub fn new() -> Self {
        // Create an empty index
        let range_idx = Index::<usize>::from_range(0..0).unwrap();
        // Convert numeric index to string index
        let string_values: Vec<String> = range_idx.values().iter().map(|i| i.to_string()).collect();
        let string_idx = Index::<String>::new(string_values).unwrap();
        let index = DataFrameIndex::<String>::from_simple(string_idx);

        DataFrame {
            columns: HashMap::new(),
            column_names: Vec::new(),
            index,
        }
    }
    
    /// Add row data to the DataFrame
    /// 
    /// This function adds one row of data in string representation to the DataFrame.
    /// The specified map contains column names and their string representation values.
    /// Empty strings are inserted for columns not included in the map.
    pub fn add_row_data(&mut self, row_data: HashMap<String, String>) -> Result<()> {
        if self.columns.is_empty() {
            return Err(PandRSError::InvalidOperation("Cannot add a row to a DataFrame with no columns".to_string()));
        }
        
        // Get all existing columns and values
        let mut all_column_data = HashMap::new();
        let current_row_count = self.row_count();
        
        // Collect current data
        for col_name in self.column_names.to_vec() {
            if let Some(series) = self.columns.get(&col_name) {
                let mut values = Vec::with_capacity(current_row_count + 1);
                
                // Collect existing data
                for i in 0..current_row_count {
                    values.push(series.get(i).cloned().unwrap_or_else(|| DataBox(Box::new("".to_string()))));
                }
                
                // Add new row data
                if let Some(value) = row_data.get(&col_name) {
                    values.push(DataBox(Box::new(value.clone())));
                } else {
                    // Add default value (e.g., empty string)
                    values.push(DataBox(Box::new("".to_string())));
                }
                
                all_column_data.insert(col_name, values);
            }
        }
        
        // Build new DataFrame
        let mut new_df = DataFrame::new();
        
        // Add to new DataFrame maintaining original column order
        for col_name in self.column_names.to_vec() {
            if let Some(values) = all_column_data.get(&col_name) {
                let new_series = Series::<DataBox>::new(values.clone(), Some(col_name.clone()))?;
                new_df.add_column(col_name.clone(), new_series)?;
            }
        }
        
        // Update result
        *self = new_df;
        
        Ok(())
    }

    /// Rename columns
    pub fn rename_columns(&mut self, column_map: &HashMap<String, String>) -> Result<()> {
        // Create conversion map from old column names to new ones
        let mut new_columns = HashMap::new();
        let mut new_column_names = Vec::with_capacity(self.column_names.len());
        
        // Get list of original column names
        for old_name in &self.column_names {
            // Check if exists in conversion map
            let new_name = if let Some(new_col_name) = column_map.get(old_name) {
                new_col_name.clone()
            } else {
                // Use original name if not in map
                old_name.clone()
            };
            
            // Check if column with same name already exists (duplicate column name after conversion)
            if new_column_names.contains(&new_name) && old_name != &new_name {
                return Err(PandRSError::Column(format!("Column name '{}' already exists", new_name)));
            }
            
            // Copy column data
            if let Some(col_data) = self.columns.get(old_name) {
                new_columns.insert(new_name.clone(), col_data.clone());
                new_column_names.push(new_name);
            }
        }
        
        // Update with new column data
        self.columns = new_columns;
        self.column_names = new_column_names;
        
        Ok(())
    }
    
    /// Create DataFrame with string index
    pub fn with_index(index: Index<String>) -> Self  
    {
        DataFrame {
            columns: HashMap::new(),
            column_names: Vec::new(),
            index: DataFrameIndex::<String>::from_simple(index),
        }
    }
    
    /// Create DataFrame with multi-index
    pub fn with_multi_index(index: crate::index::MultiIndex<String>) -> Self  
    {
        DataFrame {
            columns: HashMap::new(),
            column_names: Vec::new(),
            index: DataFrameIndex::<String>::from_multi(index),
        }
    }
    
    /// Create DataFrame with integer range index
    pub fn with_range_index(range: std::ops::Range<usize>) -> Result<Self> {
        let range_idx = Index::<usize>::from_range(range)?;
        // Convert numeric index to string index
        let string_values: Vec<String> = range_idx.values().iter().map(|i| i.to_string()).collect();
        let string_idx = Index::<String>::new(string_values)?;

        Ok(DataFrame {
            columns: HashMap::new(),
            column_names: Vec::new(),
            index: DataFrameIndex::<String>::from_simple(string_idx),
        })
    }

    /// Add a column to the DataFrame
    pub fn add_column<T>(&mut self, name: String, series: Series<T>) -> Result<()>
    where
        T: Debug + Clone + 'static + Send + Sync,
    {
        // Check if a column with this name already exists
        if self.columns.contains_key(&name) {
            return Err(PandRSError::Column(format!(
                "Column name '{}' already exists",
                name
            )));
        }

        // Check row count if the DataFrame is not empty
        if !self.column_names.is_empty() && series.len() != self.index.len() {
            return Err(PandRSError::Consistency(format!(
                "Column length ({}) does not match DataFrame row count ({})",
                series.len(),
                self.index.len()
            )));
        }

        // Type erasure and conversion to Box
        let boxed_series = self.box_series(series);

        // Set index if this is the first column
        if self.column_names.is_empty() {
            let range_idx = Index::<usize>::from_range(0..boxed_series.len())?;
            let string_values: Vec<String> = range_idx.values().iter().map(|i| i.to_string()).collect();
            let string_idx = Index::<String>::new(string_values)?;
            self.index = DataFrameIndex::<String>::from_simple(string_idx);
        }

        // Add the column
        self.columns.insert(name.clone(), boxed_series);
        self.column_names.push(name);

        Ok(())
    }
    
    /// Set a string index for the DataFrame
    pub fn set_index(&mut self, index: Index<String>) -> Result<()>
    {
        // Check index length
        if !self.column_names.is_empty() && index.len() != self.index.len() {
            return Err(PandRSError::Consistency(format!(
                "New index length ({}) does not match DataFrame row count ({})",
                index.len(),
                self.index.len()
            )));
        }
        
        // Set the index
        self.index = DataFrameIndex::<String>::from_simple(index);
        
        Ok(())
    }
    
    /// Set a multi-index for the DataFrame
    pub fn set_multi_index(&mut self, index: crate::index::MultiIndex<String>) -> Result<()>
    {
        // Check index length
        if !self.column_names.is_empty() && index.len() != self.index.len() {
            return Err(PandRSError::Consistency(format!(
                "New multi-index length ({}) does not match DataFrame row count ({})",
                index.len(),
                self.index.len()
            )));
        }
        
        // Set the index
        self.index = DataFrameIndex::<String>::from_multi(index);
        
        Ok(())
    }
    
    /// Get the index
    pub fn get_index(&self) -> &DataFrameIndex<String> {
        &self.index
    }

    // Internal helper to box a Series
    fn box_series<T>(&self, series: Series<T>) -> Series<DataBox>
    where
        T: Debug + Clone + 'static + Send + Sync,
    {
        // Box the values
        let boxed_values: Vec<DataBox> = series
            .values()
            .iter()
            .map(|v| DataBox(Box::new(v.clone())))
            .collect();

        // Create a new Series
        Series::new(boxed_values, series.name().cloned()).unwrap()
    }

    /// Get the number of columns
    pub fn column_count(&self) -> usize {
        self.column_names.len()
    }

    /// Get the number of rows
    pub fn row_count(&self) -> usize {
        self.index.len()
    }
    
    /// Set a column as the index
    pub fn set_column_as_index(&mut self, column_name: &str) -> Result<()> {
        // Check if column exists
        if !self.contains_column(column_name) {
            return Err(PandRSError::Column(format!(
                "Column '{}' does not exist",
                column_name
            )));
        }
        
        // Get column values
        let values = self.get_column_string_values(column_name)?;
        
        // Create string index
        let string_index = Index::new(values)?;
        
        // Set the index
        self.index = DataFrameIndex::<String>::from_simple(string_index);
        
        Ok(())
    }
    
    /// Get a column as Series
    pub fn get_column(&self, name: &str) -> Option<Series<String>> {
        if let Some(series) = self.columns.get(name) {
            // Convert from Series<DataBox> to Series<String>
            let string_values: Vec<String> = series
                .values()
                .iter()
                .map(|v| format!("{:?}", v))
                .collect();
                
            Series::new(string_values, Some(name.to_string())).ok()
        } else {
            None
        }
    }

    /// Get the list of column names
    pub fn column_names(&self) -> &[String] {
        &self.column_names
    }

    /// Check if a column exists
    pub fn contains_column(&self, name: &str) -> bool {
        self.columns.contains_key(name)
    }

    /// Save DataFrame to a CSV file
    pub fn to_csv<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        io::csv::write_csv(self, path)
    }

    /// Create a DataFrame from a CSV file
    pub fn from_csv<P: AsRef<Path>>(path: P, has_header: bool) -> Result<Self> {
        io::csv::read_csv(path, has_header)
    }
    
    /// Convert DataFrame to JSON string
    pub fn to_json(&self) -> Result<String> {
        let mut obj = serde_json::Map::new();
        
        // Convert each column to JSON
        for col in self.column_names() {
            let values = self.get_column(col).unwrap_or_default();
            obj.insert(col.clone(), serde_json::Value::Array(
                values.values().iter().map(|v| serde_json::Value::String(v.clone())).collect()
            ));
        }
        
        // Add index if available
        if let Some(index_values) = self.index.string_values() {
            obj.insert("index".to_string(), serde_json::Value::Array(
                index_values.iter().map(|v| serde_json::Value::String(v.clone())).collect()
            ));
        }
        
        Ok(serde_json::to_string(&obj)?)
    }
    
    /// Create DataFrame from JSON string
    pub fn from_json(json: &str) -> Result<Self> {
        let parsed: serde_json::Map<String, serde_json::Value> = serde_json::from_str(json)?;
        let mut data = HashMap::new();
        let mut index_values = None;
        
        // Process each field
        for (key, value) in parsed.iter() {
            if key == "index" {
                if let serde_json::Value::Array(arr) = value {
                    index_values = Some(
                        arr.iter()
                           .map(|v| match v {
                               serde_json::Value::String(s) => s.clone(),
                               _ => v.to_string(),
                           })
                           .collect()
                    );
                }
                continue;
            }
            
            if let serde_json::Value::Array(arr) = value {
                let values: Vec<String> = arr.iter()
                    .map(|v| match v {
                        serde_json::Value::String(s) => s.clone(),
                        _ => v.to_string(),
                    })
                    .collect();
                data.insert(key.clone(), values);
            }
        }
        
        Self::from_map(data, index_values)
    }
    
    /// Create DataFrame from a map (HashMap)
    pub fn from_map(data: HashMap<String, Vec<String>>, index: Option<Vec<String>>) -> Result<Self> {
        let mut df = Self::new();
        
        // Check row count (ensure all columns have the same length)
        let row_count = if !data.is_empty() {
            data.values().next().unwrap().len()
        } else {
            0
        };
        
        for (col_name, values) in data {
            if values.len() != row_count {
                return Err(PandRSError::Consistency(format!(
                    "Column '{}' length ({}) does not match other columns' length ({})",
                    col_name, values.len(), row_count
                )));
            }
            
            df.add_column(col_name, Series::new(values, None)?)?;
        }
        
        // Set index if provided
        if let Some(idx_values) = index {
            if idx_values.len() == row_count {
                let idx = Index::new(idx_values)?;
                df.set_index(idx)?;
            } else {
                return Err(PandRSError::Consistency(format!(
                    "Index length ({}) does not match data row count ({})",
                    idx_values.len(), row_count
                )));
            }
        }
        
        Ok(df)
    }

    /// Get string values from a column (for pivot operations)
    pub fn get_column_string_values(&self, column: &str) -> Result<Vec<String>> {
        if let Some(series) = self.columns.get(column) {
            let values = series
                .values()
                .iter()
                .map(|data_box| {
                    // Convert from DataBox to string
                    format!("{:?}", data_box)
                })
                .collect();
            Ok(values)
        } else {
            Err(PandRSError::Column(format!(
                "Column '{}' does not exist",
                column
            )))
        }
    }

    /// Get numeric values from a column (for pivot operations)
    pub fn get_column_numeric_values(&self, column: &str) -> Result<Vec<f64>> {
        if let Some(series) = self.columns.get(column) {
            let values = series
                .values()
                .iter()
                .map(|data_box| {
                    // Convert from DataBox to numeric value (string to number conversion)
                    // Note: In a real application, more precise type conversion would be needed
                    let str_val = format!("{:?}", data_box);
                    str_val.parse::<f64>().unwrap_or(0.0)
                })
                .collect();
            Ok(values)
        } else {
            Err(PandRSError::Column(format!(
                "Column '{}' does not exist",
                column
            )))
        }
    }
}

// Default trait implementation
impl Default for DataFrame {
    fn default() -> Self {
        Self::new()
    }
}
