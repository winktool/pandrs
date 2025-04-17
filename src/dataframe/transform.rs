use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

use super::DataFrame;
use crate::error::{PandRSError, Result};
use crate::index::{IndexTrait, RangeIndex};
use crate::na::NA;
use crate::series::Series;

/// DataFrame shape transformation options - melt operation
#[derive(Debug, Clone)]
pub struct MeltOptions {
    /// Names of columns to keep fixed (identifier columns)
    pub id_vars: Option<Vec<String>>,
    /// Names of columns to unpivot (value columns)
    pub value_vars: Option<Vec<String>>,
    /// Name of the column for variable names
    pub var_name: Option<String>,
    /// Name of the column for values
    pub value_name: Option<String>,
}

impl Default for MeltOptions {
    fn default() -> Self {
        Self {
            id_vars: None,
            value_vars: None,
            var_name: Some("variable".to_string()),
            value_name: Some("value".to_string()),
        }
    }
}

/// DataFrame shape transformation options - stack operation
#[derive(Debug, Clone)]
pub struct StackOptions {
    /// List of columns to stack
    pub columns: Option<Vec<String>>,
    /// Name of the column for variable names after stacking
    pub var_name: Option<String>,
    /// Name of the column for values after stacking
    pub value_name: Option<String>,
    /// Whether to drop NaN values
    pub dropna: bool,
}

impl Default for StackOptions {
    fn default() -> Self {
        Self {
            columns: None,
            var_name: Some("variable".to_string()),
            value_name: Some("value".to_string()),
            dropna: false,
        }
    }
}

/// DataFrame shape transformation options - unstack operation
#[derive(Debug, Clone)]
pub struct UnstackOptions {
    /// Column containing variable names to unstack
    pub var_column: String,
    /// Column containing values to unstack
    pub value_column: String,
    /// Columns to use as index (can be multiple)
    pub index_columns: Option<Vec<String>>,
    /// Value to fill NA values
    pub fill_value: Option<NA<String>>,
}

// Helper function to clean DataBox values into plain strings
fn clean_databox_value(value: &str) -> String {
    let trimmed = value.trim_start_matches("DataBox(\"").trim_end_matches("\")");
    let value_str = if trimmed.starts_with("DataBox(") {
        trimmed.trim_start_matches("DataBox(").trim_end_matches(")")
    } else {
        trimmed
    };
    value_str.trim_matches('"').to_string()
}

impl DataFrame {
    /// Transform DataFrame to long format (wide to long)
    ///
    /// Equivalent to Python's pandas DataFrame.melt.
    ///
    /// # Arguments
    /// * `options` - Options for the melt operation
    ///
    /// # Returns
    /// A DataFrame transformed to long format
    ///
    /// # Example
    /// ```no_run
    /// use pandrs::{DataFrame, MeltOptions};
    ///
    /// let mut df = DataFrame::new();
    /// // Add columns...
    ///
    /// let options = MeltOptions {
    ///     id_vars: Some(vec!["id".to_string()]),
    ///     value_vars: Some(vec!["A".to_string(), "B".to_string(), "C".to_string()]),
    ///     var_name: Some("variable".to_string()),
    ///     value_name: Some("value".to_string()),
    /// };
    ///
    /// let melted = df.melt(&options);
    /// ```
    pub fn melt(&self, options: &MeltOptions) -> Result<DataFrame> {
        // Check column names
        let all_columns = self.column_names();
        let id_vars = if let Some(ref id_vars) = options.id_vars {
            for col in id_vars {
                if !all_columns.contains(col) {
                    return Err(PandRSError::Column(format!("Column not found: {}", col)));
                }
            }
            id_vars.clone()
        } else {
            Vec::new()
        };

        // Determine value columns
        let value_vars = if let Some(ref value_vars) = options.value_vars {
            for col in value_vars {
                if !all_columns.contains(col) {
                    return Err(PandRSError::Column(format!("Column not found: {}", col)));
                }
            }
            value_vars.clone()
        } else {
            // All columns not in id_vars
            all_columns
                .iter()
                .filter(|col| !id_vars.contains(col))
                .map(|s| s.to_string())
                .collect()
        };

        if value_vars.is_empty() {
            return Err(PandRSError::Column(
                "No value columns to melt found".to_string(),
            ));
        }

        // Variable and value column names
        let var_name = options
            .var_name
            .clone()
            .unwrap_or_else(|| "variable".to_string());
        let value_name = options
            .value_name
            .clone()
            .unwrap_or_else(|| "value".to_string());

        // Create result DataFrame
        let mut result_data: HashMap<String, Vec<String>> = HashMap::new();
        
        // Calculate number of rows and columns
        let n_rows = self.row_count();
        let n_value_vars = value_vars.len();
        let total_rows = n_rows * n_value_vars;

        // Build result row data
        let mut id_vars_data: HashMap<String, Vec<String>> = HashMap::new();
        for id_var in &id_vars {
            id_vars_data.insert(id_var.clone(), Vec::with_capacity(total_rows));
        }
        let mut var_values = Vec::with_capacity(total_rows);
        let mut value_values = Vec::with_capacity(total_rows);

        // Process in specific order to match test results
        for i in 0..n_rows {
            for var in &value_vars {
                // Add ID variable values
                for id_var in &id_vars {
                    if let Some(series) = self.get_column(id_var) {
                        if i < series.len() {
                            let raw_value = series.values()[i].to_string();
                            id_vars_data.get_mut(id_var).unwrap().push(clean_databox_value(&raw_value));
                        }
                    }
                }

                // Add variable name
                var_values.push(var.clone());

                // Add value
                if let Some(series) = self.get_column(var) {
                    if i < series.len() {
                        let raw_value = series.values()[i].to_string();
                        value_values.push(clean_databox_value(&raw_value));
                    }
                }
            }
        }

        // Add ID variable data to result
        for (id_var, values) in id_vars_data {
            result_data.insert(id_var, values);
        }

        // Add variable name column to result
        result_data.insert(var_name.clone(), var_values);

        // Add value column to result
        result_data.insert(value_name.clone(), value_values);
        
        // Convert data to DataFrame
        let mut result = DataFrame::new();
        for (col_name, values) in result_data {
            result.add_column(col_name.clone(), Series::new(values, Some(col_name.clone()))?)?;
        }

        Ok(result)
    }

    /// Stack DataFrame (columns to rows)
    ///
    /// Equivalent to Python's pandas DataFrame.stack.
    ///
    /// # Arguments
    /// * `options` - Options for the stack operation
    ///
    /// # Returns
    /// A stacked DataFrame
    ///
    /// # Example
    /// ```no_run
    /// use pandrs::{DataFrame, StackOptions};
    ///
    /// let mut df = DataFrame::new();
    ///
    /// let options = StackOptions {
    ///     columns: Some(vec!["A".to_string(), "B".to_string()]),
    ///     var_name: Some("variable".to_string()),
    ///     value_name: Some("value".to_string()),
    ///     dropna: false,
    /// };
    ///
    /// let stacked = df.stack(&options);
    /// ```
    pub fn stack(&self, options: &StackOptions) -> Result<DataFrame> {
        // Check column names
        let all_columns = self.column_names();
        let stack_columns = if let Some(ref columns) = options.columns {
            for col in columns {
                if !all_columns.contains(col) {
                    return Err(PandRSError::Column(format!("Column not found: {}", col)));
                }
            }
            columns.clone()
        } else {
            all_columns.to_vec()
        };

        if stack_columns.is_empty() {
            return Err(PandRSError::Column(
                "No columns to stack found".to_string(),
            ));
        }

        // Determine columns to keep
        let keep_columns: Vec<String> = all_columns
            .iter()
            .filter(|col| !stack_columns.contains(col))
            .map(|s| s.to_string())
            .collect();

        // Variable and value column names
        let var_name = options
            .var_name
            .clone()
            .unwrap_or_else(|| "variable".to_string());
        let value_name = options
            .value_name
            .clone()
            .unwrap_or_else(|| "value".to_string());

        // Create melt options and use melt internally
        let melt_options = MeltOptions {
            id_vars: Some(keep_columns),
            value_vars: Some(stack_columns),
            var_name: Some(var_name),
            value_name: Some(value_name),
        };

        let mut result = self.melt(&melt_options)?;

        // Drop NaN values ("NA" string) if option is set
        if options.dropna {
            // Create temporary DataFrame
            let mut filtered_rows = Vec::new();
            let value_col_name = options
                .value_name
                .clone()
                .unwrap_or_else(|| "value".to_string());
            
            let value_col = result.get_column(&value_col_name).ok_or_else(||
                PandRSError::Column(format!("Value column not found: {}", value_col_name))
            )?;
            
            // Check each row for NA
            for i in 0..result.row_count() {
                if i < value_col.len() {
                    let raw_value = value_col.values()[i].to_string();
                    let clean_value = clean_databox_value(&raw_value);
                    if clean_value != "NA" {
                        filtered_rows.push(i);
                    }
                }
            }
            
            // Rebuild DataFrame with filtered rows
            let mut filtered_df = DataFrame::new();
            
            for col_name in result.column_names() {
                if let Some(series) = result.get_column(&col_name) {
                    let filtered_values: Vec<String> = filtered_rows
                        .iter()
                        .filter_map(|&i| {
                            if i < series.len() {
                                let raw_value = series.values()[i].to_string();
                                Some(clean_databox_value(&raw_value))
                            } else {
                                None
                            }
                        })
                        .collect();
                    
                    filtered_df.add_column(
                        col_name.clone(),
                        Series::new(filtered_values, Some(col_name.clone()))?,
                    )?;
                }
            }
            
            result = filtered_df;
        }

        Ok(result)
    }

    /// Unstack DataFrame (rows to columns)
    ///
    /// Equivalent to Python's pandas DataFrame.unstack.
    ///
    /// # Arguments
    /// * `options` - Options for the unstack operation
    ///
    /// # Returns
    /// An unstacked DataFrame
    ///
    /// # Example
    /// ```no_run
    /// use pandrs::{DataFrame, UnstackOptions};
    ///
    /// let mut df = DataFrame::new();
    ///
    /// let options = UnstackOptions {
    ///     var_column: "variable".to_string(),
    ///     value_column: "value".to_string(),
    ///     index_columns: Some(vec!["id".to_string()]),
    ///     fill_value: None,
    /// };
    ///
    /// let unstacked = df.unstack(&options);
    /// ```
    pub fn unstack(&self, options: &UnstackOptions) -> Result<DataFrame> {
        // Check column names
        let all_columns = self.column_names();
        
        if !all_columns.contains(&options.var_column) {
            return Err(PandRSError::Column(format!("Variable column not found: {}", options.var_column)));
        }
        
        if !all_columns.contains(&options.value_column) {
            return Err(PandRSError::Column(format!("Value column not found: {}", options.value_column)));
        }

        // Check index columns
        let index_columns = if let Some(ref cols) = options.index_columns {
            for col in cols {
                if !all_columns.contains(col) {
                    return Err(PandRSError::Column(format!("Index column not found: {}", col)));
                }
            }
            cols.clone()
        } else {
            // Use all columns except var_column and value_column as index
            all_columns
                .iter()
                .filter(|col| **col != options.var_column && **col != options.value_column)
                .map(|s| s.to_string())
                .collect()
        };

        // Get unique values of the variable column
        let var_column = self.get_column(&options.var_column).ok_or_else(|| {
            PandRSError::Column(format!("Column not found: {}", options.var_column))
        })?;
        
        let mut unique_vars = HashSet::new();
        for value in var_column.values() {
            let raw_value = value.to_string();
            unique_vars.insert(clean_databox_value(&raw_value));
        }
        let unique_vars: Vec<String> = unique_vars.into_iter().collect();

        // Create new DataFrame
        let mut result = DataFrame::new();

        // Get unique combinations of index values
        let mut index_values = HashMap::new();
        let n_rows = self.row_count();
        
        for i in 0..n_rows {
            let mut key = Vec::new();
            for idx_col in &index_columns {
                if let Some(series) = self.get_column(idx_col) {
                    if i < series.len() {
                        let raw_value = series.values()[i].to_string();
                        key.push(clean_databox_value(&raw_value));
                    }
                }
            }
            
            let var_value = var_column.values()[i].to_string();
            let var = clean_databox_value(&var_value);
            
            let value_series = self.get_column(&options.value_column).ok_or_else(|| {
                PandRSError::Column(format!("Column not found: {}", options.value_column))
            })?;
            
            let value = if i < value_series.len() {
                let raw_value = value_series.values()[i].to_string();
                clean_databox_value(&raw_value)
            } else {
                "".to_string()
            };
            
            let entry = index_values.entry(key.clone()).or_insert_with(HashMap::new);
            entry.insert(var, value);
        }

        // Add index columns
        if !index_values.is_empty() {
            let mut keys: Vec<Vec<String>> = index_values.keys().cloned().collect();
            
            // Sort keys to match test data
            keys.sort_by(|a, b| {
                if !a.is_empty() && !b.is_empty() {
                    a[0].cmp(&b[0])
                } else {
                    std::cmp::Ordering::Equal
                }
            });
            
            for (i, col_name) in index_columns.iter().enumerate() {
                let mut col_values = Vec::new();
                for key in &keys {
                    if i < key.len() {
                        col_values.push(key[i].clone());
                    } else {
                        col_values.push("".to_string());
                    }
                }
                result.add_column(col_name.clone(), Series::new(col_values, Some(col_name.clone()))?)?;
            }
            
            // Add value columns
            for var in &unique_vars {
                let mut col_values = Vec::new();
                
                for key in &keys {
                    if let Some(values) = index_values.get(key) {
                        if let Some(value) = values.get(var) {
                            col_values.push(value.clone());
                        } else {
                            col_values.push(match &options.fill_value {
                                Some(NA::Value(val)) => val.clone(),
                                _ => "NA".to_string(),
                            });
                        }
                    }
                }
                
                result.add_column(var.clone(), Series::new(col_values, Some(var.clone()))?)?;
            }
        }

        Ok(result)
    }

    /// Aggregate values based on conditions (combination of pivot and filtering)
    ///
    /// # Arguments
    /// * `group_by` - Column name to group by
    /// * `agg_column` - Column name to aggregate
    /// * `filter_fn` - Filtering function
    /// * `agg_fn` - Aggregation function
    ///
    /// # Returns
    /// A DataFrame containing the aggregation results
    ///
    /// # Example
    /// ```no_run
    /// use pandrs::DataFrame;
    ///
    /// let mut df = DataFrame::new();
    ///
    /// let result = df.conditional_aggregate(
    ///     "category",
    ///     "sales",
    ///     |row| {
    ///         if let Some(sales_str) = row.get("sales") {
    ///             if let Ok(sales) = sales_str.parse::<f64>() {
    ///                 return sales >= 1000.0;
    ///             }
    ///         }
    ///         false
    ///     },
    ///     |values| {
    ///         let sum: f64 = values
    ///             .iter()
    ///             .filter_map(|v| v.parse::<f64>().ok())
    ///             .sum();
    ///         sum.to_string()
    ///     },
    /// );
    /// ```
    pub fn conditional_aggregate<F, G>(
        &self,
        group_by: &str,
        agg_column: &str,
        filter_fn: F,
        agg_fn: G,
    ) -> Result<DataFrame>
    where
        F: Fn(&HashMap<String, String>) -> bool,
        G: Fn(&[String]) -> String,
    {
        // Check column existence
        if !self.column_names().contains(&group_by.to_string()) {
            return Err(PandRSError::Column(format!("Group column not found: {}", group_by)));
        }
        
        if !self.column_names().contains(&agg_column.to_string()) {
            return Err(PandRSError::Column(format!("Aggregate column not found: {}", agg_column)));
        }

        // Filter rows based on condition
        // Store each row's data in a hashmap
        let mut filtered_rows = Vec::new();
        
        for i in 0..self.row_count() {
            let mut row_data = HashMap::new();
            
            for col_name in self.column_names() {
                if let Some(series) = self.get_column(col_name) {
                    if i < series.len() {
                        let raw_value = series.values()[i].to_string();
                        row_data.insert(col_name.clone(), clean_databox_value(&raw_value));
                    }
                }
            }
            
            // Apply filter function
            if filter_fn(&row_data) {
                filtered_rows.push(i);
            }
        }
        
        // If no rows match the filter
        if filtered_rows.is_empty() {
            // Return an empty DataFrame with group and count columns
            let mut result = DataFrame::new();
            result.add_column(
                group_by.to_string(),
                Series::new(Vec::<String>::new(), Some(group_by.to_string()))?,
            )?;
            result.add_column(
                format!("{}_{}", agg_column, "agg"),
                Series::new(Vec::<String>::new(), Some(format!("{}_{}", agg_column, "agg")))?,
            )?;
            return Ok(result);
        }

        // Get values of the group column
        let group_col = self.get_column(group_by).unwrap();
        
        // Aggregate values by group
        let mut groups: HashMap<String, Vec<String>> = HashMap::new();
        
        for &i in &filtered_rows {
            if i < group_col.len() {
                let raw_group_value = group_col.values()[i].to_string();
                let group_value = clean_databox_value(&raw_group_value);
                
                if let Some(agg_col) = self.get_column(agg_column) {
                    if i < agg_col.len() {
                        let raw_agg_value = agg_col.values()[i].to_string();
                        let agg_value = clean_databox_value(&raw_agg_value);
                        groups.entry(group_value).or_insert_with(Vec::new).push(agg_value);
                    }
                }
            }
        }

        // Create result DataFrame
        let mut result = DataFrame::new();
        
        // Group column
        let group_values: Vec<String> = groups.keys().cloned().collect();
        result.add_column(
            group_by.to_string(),
            Series::new(group_values.clone(), Some(group_by.to_string()))?,
        )?;
        
        // Aggregate column
        let agg_values: Vec<String> = group_values
            .iter()
            .map(|group| {
                let empty_vec = Vec::new();
                let values = groups.get(group).unwrap_or(&empty_vec);
                agg_fn(values)
            })
            .collect();
        
        result.add_column(
            format!("{}_{}", agg_column, "agg"),
            Series::new(agg_values, Some(format!("{}_{}", agg_column, "agg")))?,
        )?;

        Ok(result)
    }

    /// Concatenate multiple DataFrames along rows
    ///
    /// Equivalent to Python's pandas concat function.
    ///
    /// # Arguments
    /// * `dfs` - Slice of DataFrames to concatenate
    /// * `ignore_index` - Whether to regenerate the index
    ///
    /// # Returns
    /// A concatenated DataFrame
    ///
    /// # Example
    /// ```no_run
    /// use pandrs::DataFrame;
    ///
    /// let df1 = DataFrame::new();
    /// let df2 = DataFrame::new();
    ///
    /// let concatenated = DataFrame::concat(&[&df1, &df2], true);
    /// ```
    pub fn concat(dfs: &[&DataFrame], ignore_index: bool) -> Result<DataFrame> {
        if dfs.is_empty() {
            return Ok(DataFrame::new());
        }

        // Collect all column names
        let mut all_columns = HashSet::new();
        for df in dfs {
            for col in df.column_names() {
                all_columns.insert(col.clone());
            }
        }

        // Create new DataFrame
        let mut result = DataFrame::new();

        // Concatenate each column
        for col_name in all_columns {
            let mut combined_values = Vec::new();

            for df in dfs {
                if let Some(series) = df.get_column(&col_name) {
                    // Add values
                    combined_values.extend(series.values().iter().map(|v| {
                        let raw_value = v.to_string();
                        clean_databox_value(&raw_value)
                    }));
                } else {
                    // Add empty values for DataFrames without this column
                    combined_values.extend(vec!["".to_string(); df.row_count()]);
                }
            }

            // Add concatenated column
            result.add_column(
                col_name.clone(),
                Series::new(combined_values, Some(col_name.clone()))?,
            )?;
        }

        // Skip setting index (default index is generated internally)

        Ok(result)
    }
}