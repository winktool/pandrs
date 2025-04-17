use crate::dataframe::DataBox;
use crate::error::{PandRSError, Result};
use crate::series::Series;
use crate::DataFrame;
use std::collections::{HashMap, HashSet};

/// Enum for join types
#[derive(Debug)]
pub enum JoinType {
    /// Inner join (only rows that match in both tables)
    Inner,
    /// Left join (all rows from the left table and matching rows from the right table)
    Left,
    /// Right join (all rows from the right table and matching rows from the left table)
    Right,
    /// Outer join (all rows from both tables)
    Outer,
}

impl DataFrame {
    /// Join two DataFrames
    ///
    /// # Arguments
    /// * `other` - The other DataFrame to join with
    /// * `on` - The column name to join on
    /// * `join_type` - The type of join to perform
    ///
    /// # Returns
    /// A new DataFrame resulting from the join operation
    pub fn join(&self, other: &DataFrame, on: &str, join_type: JoinType) -> Result<DataFrame> {
        // Check if the join column exists
        if !self.contains_column(on) {
            return Err(PandRSError::Column(format!(
                "Join column '{}' does not exist in the left DataFrame",
                on
            )));
        }

        if !other.contains_column(on) {
            return Err(PandRSError::Column(format!(
                "Join column '{}' does not exist in the right DataFrame",
                on
            )));
        }

        match join_type {
            JoinType::Inner => self.inner_join(other, on),
            JoinType::Left => self.left_join(other, on),
            JoinType::Right => self.right_join(other, on),
            JoinType::Outer => self.outer_join(other, on),
        }
    }

    // Perform inner join
    fn inner_join(&self, other: &DataFrame, on: &str) -> Result<DataFrame> {
        let mut result = DataFrame::new();
        let left_keys = self.get_column_string_values(on)?;
        let right_keys = other.get_column_string_values(on)?;

        // Inner join: use only keys that exist in both
        for (left_idx, left_key) in left_keys.iter().enumerate() {
            for (right_idx, right_key) in right_keys.iter().enumerate() {
                if left_key == right_key {
                    // Add the row that matches this key to the new DataFrame
                    self.add_join_row(&mut result, left_idx, other, right_idx, on)?;
                }
            }
        }

        Ok(result)
    }

    // Perform left join
    fn left_join(&self, other: &DataFrame, on: &str) -> Result<DataFrame> {
        let mut result = DataFrame::new();
        let left_keys = self.get_column_string_values(on)?;
        let right_keys = other.get_column_string_values(on)?;

        // Build a set of unique keys from the right side
        let right_keys_set: HashSet<&String> = right_keys.iter().collect();

        // Loop through all rows on the left side
        for (left_idx, left_key) in left_keys.iter().enumerate() {
            let mut has_match = false;

            // Check if there is a matching key on the right side
            if right_keys_set.contains(left_key) {
                for (right_idx, right_key) in right_keys.iter().enumerate() {
                    if left_key == right_key {
                        // Add the row that matches this key to the new DataFrame
                        self.add_join_row(&mut result, left_idx, other, right_idx, on)?;
                        has_match = true;
                    }
                }
            }

            // If there is no match on the right side, add only the left side data
            if !has_match {
                self.add_left_only_row(&mut result, left_idx, other, on)?;
            }
        }

        Ok(result)
    }

    // Perform right join
    fn right_join(&self, other: &DataFrame, on: &str) -> Result<DataFrame> {
        // Right join is implemented as a left join with swapped arguments
        other.left_join(self, on)
    }

    // Perform outer join
    fn outer_join(&self, other: &DataFrame, on: &str) -> Result<DataFrame> {
        let mut result = DataFrame::new();
        let left_keys = self.get_column_string_values(on)?;
        let right_keys = other.get_column_string_values(on)?;

        // Track processed left keys
        let mut processed_left_indices = HashSet::new();

        // Loop through all rows on the left side
        for (left_idx, left_key) in left_keys.iter().enumerate() {
            let mut has_match = false;

            // Find matching keys on the right side
            for (right_idx, right_key) in right_keys.iter().enumerate() {
                if left_key == right_key {
                    // Add the row that matches this key to the new DataFrame
                    self.add_join_row(&mut result, left_idx, other, right_idx, on)?;
                    has_match = true;
                }
            }

            // If there is no match on the right side, add only the left side data
            if !has_match {
                self.add_left_only_row(&mut result, left_idx, other, on)?;
            }

            processed_left_indices.insert(left_key.clone());
        }

        // Process keys on the right side that do not exist on the left side
        for (right_idx, right_key) in right_keys.iter().enumerate() {
            if !processed_left_indices.contains(right_key) {
                // If there is no match on the left side, add only the right side data
                self.add_right_only_row(&mut result, other, right_idx, on)?;
            }
        }

        Ok(result)
    }

    // Helper method to add a row to the DataFrame
    fn add_row_to_dataframe(
        &self,
        result: &mut DataFrame,
        row_data: HashMap<String, DataBox>,
    ) -> Result<()> {
        // Get all existing columns and values
        let mut all_column_data = HashMap::new();
        
        // Collect current data
        for col_name in result.column_names().to_vec() {
            if let Some(series) = result.get_column(&col_name) {
                let values: Vec<DataBox> = series.values()
                    .iter()
                    .map(|v| DataBox(Box::new(v.to_string())))
                    .collect();
                all_column_data.insert(col_name.clone(), values);
            }
        }
        
        // Add row data to each column
        for (col_name, values) in all_column_data.iter_mut() {
            if let Some(value) = row_data.get(col_name) {
                values.push(value.clone());
            } else {
                // Add default value (e.g., empty string)
                let default_value = DataBox(Box::new("".to_string()));
                values.push(default_value);
            }
        }
        
        // Start building the new DataFrame
        let mut new_df = DataFrame::new();
        
        // Maintain the original column order
        for col_name in result.column_names().to_vec() {
            if let Some(values) = all_column_data.get(&col_name) {
                let new_series = Series::<DataBox>::new(values.clone(), Some(col_name.clone()))?;
                new_df.add_column(col_name.clone(), new_series)?;
            }
        }
        
        // Update the result
        *result = new_df;
        
        Ok(())
    }

    // Helper method to add a joined row to the result DataFrame
    fn add_join_row(
        &self,
        result: &mut DataFrame,
        left_idx: usize,
        other: &DataFrame,
        right_idx: usize,
        on: &str,
    ) -> Result<()> {
        // Initialize columns if the result DataFrame is empty
        if result.column_count() == 0 {
            // Add all column names from the left side
            for col_name in self.column_names() {
                let empty_series = Series::<DataBox>::new(Vec::new(), Some(col_name.clone()))?;
                result.add_column(col_name.clone(), empty_series)?;
            }

            // Add column names from the right side (excluding the join column)
            for col_name in other.column_names() {
                if col_name != on {
                    let name = format!("{}_{}", col_name, "right");
                    let empty_series = Series::<DataBox>::new(Vec::new(), Some(name.clone()))?;
                    result.add_column(name, empty_series)?;
                }
            }
        }

        // Create new row data
        let mut row_data = HashMap::new();

        // Add values from the left side
        for col_name in self.column_names() {
            if let Some(series) = self.columns.get(col_name) {
                if let Some(value) = series.get(left_idx) {
                    row_data.insert(col_name.clone(), value.clone());
                }
            }
        }

        // Add values from the right side (excluding the join column)
        for col_name in other.column_names() {
            if col_name != on {
                let result_col_name = format!("{}_{}", col_name, "right");
                if let Some(series) = other.columns.get(col_name) {
                    if let Some(value) = series.get(right_idx) {
                        row_data.insert(result_col_name, value.clone());
                    }
                }
            }
        }

        // Add row data to the DataFrame
        self.add_row_to_dataframe(result, row_data)?;

        Ok(())
    }

    // Helper method to add a left-only row to the result DataFrame
    fn add_left_only_row(
        &self,
        result: &mut DataFrame,
        left_idx: usize,
        other: &DataFrame,
        on: &str,
    ) -> Result<()> {
        // Initialize columns if the result DataFrame is empty
        if result.column_count() == 0 {
            // Add all column names from the left side
            for col_name in self.column_names() {
                let empty_series = Series::<DataBox>::new(Vec::new(), Some(col_name.clone()))?;
                result.add_column(col_name.clone(), empty_series)?;
            }

            // Add column names from the right side (excluding the join column)
            for col_name in other.column_names() {
                if col_name != on {
                    let name = format!("{}_{}", col_name, "right");
                    let empty_series = Series::<DataBox>::new(Vec::new(), Some(name.clone()))?;
                    result.add_column(name, empty_series)?;
                }
            }
        }

        // Create new row data
        let mut row_data = HashMap::new();

        // Add values from the left side
        for col_name in self.column_names() {
            if let Some(series) = self.columns.get(col_name) {
                if let Some(value) = series.get(left_idx) {
                    row_data.insert(col_name.clone(), value.clone());
                }
            }
        }

        // Add empty values to the right side columns
        for col_name in other.column_names() {
            if col_name != on {
                let result_col_name = format!("{}_{}", col_name, "right");
                // Add empty/NA value
                row_data.insert(result_col_name, DataBox(Box::new(String::new())));
            }
        }

        // Add row data to the DataFrame
        self.add_row_to_dataframe(result, row_data)?;

        Ok(())
    }

    // Helper method to add a right-only row to the result DataFrame
    fn add_right_only_row(
        &self,
        result: &mut DataFrame,
        other: &DataFrame,
        right_idx: usize,
        on: &str,
    ) -> Result<()> {
        // Initialize columns if the result DataFrame is empty
        if result.column_count() == 0 {
            // Add all column names from the left side
            for col_name in self.column_names() {
                let empty_series = Series::<DataBox>::new(Vec::new(), Some(col_name.clone()))?;
                result.add_column(col_name.clone(), empty_series)?;
            }

            // Add column names from the right side (excluding the join column)
            for col_name in other.column_names() {
                if col_name != on {
                    let name = format!("{}_{}", col_name, "right");
                    let empty_series = Series::<DataBox>::new(Vec::new(), Some(name.clone()))?;
                    result.add_column(name, empty_series)?;
                }
            }
        }

        // Create new row data
        let mut row_data = HashMap::new();

        // Add empty values to the left side columns
        for col_name in self.column_names() {
            if col_name == on {
                // Get the join key from the right side
                if let Some(series) = other.columns.get(col_name) {
                    if let Some(value) = series.get(right_idx) {
                        row_data.insert(col_name.clone(), value.clone());
                    }
                }
            } else {
                // Add empty/NA value to other columns
                row_data.insert(col_name.clone(), DataBox(Box::new(String::new())));
            }
        }

        // Add values from the right side
        for col_name in other.column_names() {
            if col_name != on {
                let result_col_name = format!("{}_{}", col_name, "right");
                if let Some(series) = other.columns.get(col_name) {
                    if let Some(value) = series.get(right_idx) {
                        row_data.insert(result_col_name, value.clone());
                    }
                }
            }
        }

        // Add row data to the DataFrame
        self.add_row_to_dataframe(result, row_data)?;

        Ok(())
    }
}
