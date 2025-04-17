use std::collections::HashMap;
use std::fmt::Debug;

use super::DataFrame;
use crate::error::{PandRSError, Result};
use crate::na::NA;
use crate::series::Series;
use crate::temporal::TimeSeries;
use crate::temporal::WindowType;

/// Axis for function application
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// Apply function to each column
    Column = 0,
    /// Apply function to each row
    Row = 1,
}

impl DataFrame {
    /// Apply a function to each column or row
    ///
    /// Equivalent to Python's pandas DataFrame.apply.
    ///
    /// # Arguments
    /// * `f` - Function to apply
    /// * `axis` - Axis for function application (column or row)
    /// * `result_name` - Name of the resulting Series
    ///
    /// # Returns
    /// A new Series containing the results of the function application
    ///
    /// # Example
    /// ```no_run
    /// use pandrs::{DataFrame};
    /// use pandrs::dataframe::apply::Axis;
    ///
    /// // Create a DataFrame
    /// let mut df = DataFrame::new();
    /// // Add columns...
    ///
    /// // Get the first element of each column (example)
    /// let first_elems = df.apply(|series| series.get(0).unwrap().clone(), Axis::Column, Some("first".to_string()));
    /// ```
    pub fn apply<F, R>(&self, f: F, axis: Axis, result_name: Option<String>) -> Result<Series<R>>
    where
        F: Fn(&Series<String>) -> R,
        R: Debug + Clone,
    {
        match axis {
            Axis::Column => {
                // Apply f to each column
                let mut results = Vec::with_capacity(self.columns.len());
                
                for col_name in self.column_names() {
                    let col = self.get_column(col_name).unwrap();
                    results.push(f(&col));
                }
                
                // Construct Series from results
                Series::new(results, result_name)
            }
            Axis::Row => {
                // Row-wise application is not supported at this time
                Err(PandRSError::NotImplemented(
                    "Row-wise function application is not currently supported".to_string(),
                ))
            }
        }
    }
    
    /// Apply a function to each element
    ///
    /// Equivalent to Python's pandas DataFrame.applymap.
    ///
    /// # Arguments
    /// * `f` - Function to apply to each element
    ///
    /// # Returns
    /// A new DataFrame containing the results of the function application
    ///
    /// # Example
    /// ```no_run
    /// use pandrs::DataFrame;
    ///
    /// // Create a DataFrame
    /// let mut df = DataFrame::new();
    /// // Add columns...
    ///
    /// // Double each element (example)
    /// let doubled = df.applymap(|x| x.parse::<i32>().unwrap_or(0) * 2);
    /// ```
    pub fn applymap<F, R>(&self, f: F) -> Result<DataFrame>
    where
        F: Fn(&str) -> R,
        R: Debug + Clone + ToString,
    {
        let mut result_df = DataFrame::new();
        
        // Apply function to each element of each column
        for col_name in self.column_names() {
            let col = self.get_column(col_name).unwrap();
            let mut transformed = Vec::with_capacity(col.len());
            
            for val in col.values() {
                transformed.push(f(val).to_string());
            }
            
            // Add transformed column to DataFrame
            let series = Series::new(transformed, Some(col_name.clone()))?;
            result_df.add_column(col_name.clone(), series)?;
        }
        
        Ok(result_df)
    }
    
    /// Replace values based on a condition
    ///
    /// Equivalent to Python's pandas DataFrame.mask.
    ///
    /// # Arguments
    /// * `condition` - Function to evaluate the condition (replace if true)
    /// * `other` - Value to replace with
    ///
    /// # Returns
    /// A new DataFrame with values replaced based on the condition
    pub fn mask<F>(&self, condition: F, other: &str) -> Result<DataFrame>
    where
        F: Fn(&str) -> bool,
    {
        let mut result_df = DataFrame::new();
        
        // Replace elements that match the condition in each column
        for col_name in self.column_names() {
            let col = self.get_column(col_name).unwrap();
            let mut transformed = Vec::with_capacity(col.len());
            
            for val in col.values() {
                if condition(val) {
                    transformed.push(other.to_string());
                } else {
                    transformed.push(val.clone().to_string());
                }
            }
            
            // Add transformed column to DataFrame
            let series = Series::new(transformed, Some(col_name.clone()))?;
            result_df.add_column(col_name.clone(), series)?;
        }
        
        Ok(result_df)
    }
    
    /// Replace values based on a condition (inverse of mask)
    ///
    /// Equivalent to Python's pandas DataFrame.where.
    ///
    /// # Arguments
    /// * `condition` - Function to evaluate the condition (replace if false)
    /// * `other` - Value to replace with
    ///
    /// # Returns
    /// A new DataFrame with values replaced based on the condition
    pub fn where_func<F>(&self, condition: F, other: &str) -> Result<DataFrame>
    where
        F: Fn(&str) -> bool,
    {
        let mut result_df = DataFrame::new();
        
        // Replace elements that do not match the condition in each column
        for col_name in self.column_names() {
            let col = self.get_column(col_name).unwrap();
            let mut transformed = Vec::with_capacity(col.len());
            
            for val in col.values() {
                if !condition(val) {
                    transformed.push(other.to_string());
                } else {
                    transformed.push(val.clone().to_string());
                }
            }
            
            // Add transformed column to DataFrame
            let series = Series::new(transformed, Some(col_name.clone()))?;
            result_df.add_column(col_name.clone(), series)?;
        }
        
        Ok(result_df)
    }
    
    /// Replace values with corresponding values
    ///
    /// Equivalent to Python's pandas DataFrame.replace.
    ///
    /// # Arguments
    /// * `replace_map` - Map of values to replace
    ///
    /// # Returns
    /// A new DataFrame with values replaced
    pub fn replace(&self, replace_map: &HashMap<String, String>) -> Result<DataFrame> {
        let mut result_df = DataFrame::new();
        
        // Replace elements in each column
        for col_name in self.column_names() {
            let col = self.get_column(col_name).unwrap();
            let mut transformed = Vec::with_capacity(col.len());
            
            for val in col.values() {
                match replace_map.get(val) {
                    Some(replacement) => transformed.push(replacement.clone()),
                    None => transformed.push(val.clone().to_string()),
                }
            }
            
            // Add transformed column to DataFrame
            let series = Series::new(transformed, Some(col_name.clone()))?;
            result_df.add_column(col_name.clone(), series)?;
        }
        
        Ok(result_df)
    }
    
    /// Detect duplicate rows
    ///
    /// Equivalent to Python's pandas DataFrame.duplicated.
    ///
    /// # Arguments
    /// * `subset` - Subset of columns to check for duplicates (all columns if None)
    /// * `keep` - Which duplicate rows to keep ("first"=keep first, "last"=keep last, None=mark all duplicates)
    ///
    /// # Returns
    /// A Series of booleans indicating whether each row is a duplicate
    pub fn duplicated(
        &self,
        subset: Option<&[String]>,
        keep: Option<&str>,
    ) -> Result<Series<bool>> {
        let columns_to_check = match subset {
            Some(cols) => {
                // Check if specified columns exist
                for col in cols {
                    if !self.contains_column(col) {
                        return Err(PandRSError::Column(format!(
                            "Column '{}' does not exist",
                            col
                        )));
                    }
                }
                cols.to_vec()
            }
            None => self.column_names().to_vec(),
        };
        
        let row_count = self.row_count();
        let mut duplicated = vec![false; row_count];
        let mut seen = HashMap::new();
        
        // Convert each row to a unique key string
        for i in 0..row_count {
            let mut row_key = String::new();
            
            for col in &columns_to_check {
                let value = self.get_column(col).unwrap().values()[i].clone();
                row_key.push_str(&value);
                row_key.push('\0'); // Separator
            }
            
            match keep {
                Some("first") => {
                    // Mark all but the first occurrence
                    if seen.contains_key(&row_key) {
                        duplicated[i] = true;
                    } else {
                        seen.insert(row_key, i);
                    }
                }
                Some("last") => {
                    // Mark current row and unmark the last occurrence later
                    if seen.contains_key(&row_key) {
                        duplicated[*seen.get(&row_key).unwrap()] = true;
                    }
                    seen.insert(row_key, i);
                }
                _ => {
                    // Mark all occurrences after the first
                    if seen.contains_key(&row_key) {
                        duplicated[i] = true;
                        duplicated[*seen.get(&row_key).unwrap()] = true;
                    } else {
                        seen.insert(row_key, i);
                    }
                }
            }
        }
        
        // Return result as a Series
        Series::new(duplicated, Some("duplicated".to_string()))
    }
    
    /// Drop duplicate rows
    ///
    /// Equivalent to Python's pandas DataFrame.drop_duplicates.
    ///
    /// # Arguments
    /// * `subset` - Subset of columns to check for duplicates (all columns if None)
    /// * `keep` - Which duplicate rows to keep ("first"=keep first, "last"=keep last, None=drop all duplicates)
    ///
    /// # Returns
    /// A new DataFrame with duplicate rows removed
    pub fn drop_duplicates(
        &self,
        subset: Option<&[String]>,
        keep: Option<&str>,
    ) -> Result<DataFrame> {
        // Identify duplicate rows
        let is_duplicated = self.duplicated(subset, keep)?;
        
        let mut result_df = DataFrame::new();
        
        // Extract non-duplicate rows for each column
        for col_name in self.column_names() {
            let col = self.get_column(col_name).unwrap();
            let mut filtered = Vec::new();
            
            for i in 0..col.len() {
                if !is_duplicated.values()[i] {
                    filtered.push(col.values()[i].clone());
                }
            }
            
            // Add filtered column to DataFrame
            let series = Series::new(filtered, Some(col_name.clone()))?;
            result_df.add_column(col_name.clone(), series)?;
        }
        
        Ok(result_df)
    }
    
    /// Apply a fixed-length window (rolling window) operation
    /// 
    /// # Arguments
    /// * `window_size` - Window size
    /// * `column_name` - Target column name
    /// * `operation` - Type of window operation ("mean", "sum", "std", "min", "max")
    /// * `result_column` - Name of the result column (default is "{column_name}_{operation}_{window_size}")
    /// 
    /// # Returns
    /// DataFrame with the operation applied
    pub fn rolling(
        &self,
        window_size: usize,
        column_name: &str,
        operation: &str,
        result_column: Option<&str>,
    ) -> Result<DataFrame> {
        // Check if target column exists
        let series = match self.get_column(column_name) {
            Some(s) => s,
            None => return Err(PandRSError::KeyNotFound(column_name.to_string())),
        };
        
        // Result column name
        let result_col = result_column.unwrap_or(&format!(
            "{}_{}_{}", column_name, operation, window_size
        )).to_string();
        
        // Extract numeric data from series
        let values: Vec<NA<f64>> = series.values().iter()
            .map(|s| {
                match s.parse::<f64>() {
                    Ok(num) => NA::Value(num),
                    Err(_) => NA::NA,
                }
            })
            .collect();
        
        // Timestamps are obtained from the index
        // (use dummy dates if actual timestamps are not available)
        let timestamps = self.get_index().to_datetime_vec()?;
        
        // Create TimeSeries
        let time_series = TimeSeries::new(
            values,
            timestamps,
            Some(column_name.to_string()),
        )?;
        
        // Perform window operation
        let window_result = match operation.to_lowercase().as_str() {
            "mean" => time_series.rolling(window_size)?.mean()?,
            "sum" => time_series.rolling(window_size)?.sum()?,
            "std" => time_series.rolling(window_size)?.std(1)?,
            "min" => time_series.rolling(window_size)?.min()?,
            "max" => time_series.rolling(window_size)?.max()?,
            _ => return Err(PandRSError::Operation(format!(
                "Unsupported window operation: {}", operation
            ))),
        };
        
        // Convert result to String Series
        let result_strings: Vec<String> = window_result.values().iter()
            .map(|v| match v {
                NA::Value(val) => val.to_string(),
                NA::NA => "NA".to_string(),
            })
            .collect();
        
        // Create result Series
        let result_series = Series::new(
            result_strings,
            Some(result_col.clone()),
        )?;
        
        // Add result column to original DataFrame
        let mut result_df = self.clone();
        result_df.add_column(result_col, result_series)?;
        
        Ok(result_df)
    }
    
    /// Apply an expanding window operation
    /// 
    /// # Arguments
    /// * `min_periods` - Minimum period size
    /// * `column_name` - Target column name
    /// * `operation` - Type of window operation ("mean", "sum", "std", "min", "max")
    /// * `result_column` - Name of the result column (default is "{column_name}_expanding_{operation}")
    /// 
    /// # Returns
    /// DataFrame with the operation applied
    pub fn expanding(
        &self,
        min_periods: usize,
        column_name: &str,
        operation: &str,
        result_column: Option<&str>,
    ) -> Result<DataFrame> {
        // Check if target column exists
        let series = match self.get_column(column_name) {
            Some(s) => s,
            None => return Err(PandRSError::KeyNotFound(column_name.to_string())),
        };
        
        // Result column name
        let result_col = result_column.unwrap_or(&format!(
            "{}_expanding_{}", column_name, operation
        )).to_string();
        
        // Extract numeric data from series
        let values: Vec<NA<f64>> = series.values().iter()
            .map(|s| {
                match s.parse::<f64>() {
                    Ok(num) => NA::Value(num),
                    Err(_) => NA::NA,
                }
            })
            .collect();
        
        // Timestamps are obtained from the index
        let timestamps = self.get_index().to_datetime_vec()?;
        
        // Create TimeSeries
        let time_series = TimeSeries::new(
            values,
            timestamps,
            Some(column_name.to_string()),
        )?;
        
        // Perform window operation
        let window_result = match operation.to_lowercase().as_str() {
            "mean" => time_series.expanding(min_periods)?.mean()?,
            "sum" => time_series.expanding(min_periods)?.sum()?,
            "std" => time_series.expanding(min_periods)?.std(1)?,
            "min" => time_series.expanding(min_periods)?.min()?,
            "max" => time_series.expanding(min_periods)?.max()?,
            _ => return Err(PandRSError::Operation(format!(
                "Unsupported window operation: {}", operation
            ))),
        };
        
        // Convert result to String Series
        let result_strings: Vec<String> = window_result.values().iter()
            .map(|v| match v {
                NA::Value(val) => val.to_string(),
                NA::NA => "NA".to_string(),
            })
            .collect();
        
        // Create result Series
        let result_series = Series::new(
            result_strings,
            Some(result_col.clone()),
        )?;
        
        // Add result column to original DataFrame
        let mut result_df = self.clone();
        result_df.add_column(result_col, result_series)?;
        
        Ok(result_df)
    }
    
    /// Apply an exponentially weighted window operation
    /// 
    /// # Arguments
    /// * `column_name` - Target column name
    /// * `operation` - Type of window operation ("mean", "std")
    /// * `span` - Half-life (cannot be specified with alpha)
    /// * `alpha` - Decay factor 0.0 < alpha <= 1.0 (cannot be specified with span)
    /// * `result_column` - Name of the result column (default is "{column_name}_ewm_{operation}")
    /// 
    /// # Returns
    /// DataFrame with the operation applied
    pub fn ewm(
        &self,
        column_name: &str,
        operation: &str,
        span: Option<usize>,
        alpha: Option<f64>,
        result_column: Option<&str>,
    ) -> Result<DataFrame> {
        // Error if both span and alpha are specified
        if span.is_some() && alpha.is_some() {
            return Err(PandRSError::Consistency(
                "span and alpha cannot be specified simultaneously. Please specify one or the other.".to_string(),
            ));
        }
        
        // Check if target column exists
        let series = match self.get_column(column_name) {
            Some(s) => s,
            None => return Err(PandRSError::KeyNotFound(column_name.to_string())),
        };
        
        // Result column name
        let result_col = result_column.unwrap_or(&format!(
            "{}_ewm_{}", column_name, operation
        )).to_string();
        
        // Extract numeric data from series
        let values: Vec<NA<f64>> = series.values().iter()
            .map(|s| {
                match s.parse::<f64>() {
                    Ok(num) => NA::Value(num),
                    Err(_) => NA::NA,
                }
            })
            .collect();
        
        // Timestamps are obtained from the index
        let timestamps = self.get_index().to_datetime_vec()?;
        
        // Create TimeSeries
        let time_series = TimeSeries::new(
            values,
            timestamps,
            Some(column_name.to_string()),
        )?;
        
        // Perform window operation
        let window_result = match operation.to_lowercase().as_str() {
            "mean" => time_series.ewm(span, alpha, false)?.mean()?,
            "std" => time_series.ewm(span, alpha, false)?.std(1)?,
            _ => return Err(PandRSError::Operation(format!(
                "Unsupported EWM operation: {}", operation
            ))),
        };
        
        // Convert result to String Series
        let result_strings: Vec<String> = window_result.values().iter()
            .map(|v| match v {
                NA::Value(val) => val.to_string(),
                NA::NA => "NA".to_string(),
            })
            .collect();
        
        // Create result Series
        let result_series = Series::new(
            result_strings,
            Some(result_col.clone()),
        )?;
        
        // Add result column to original DataFrame
        let mut result_df = self.clone();
        result_df.add_column(result_col, result_series)?;
        
        Ok(result_df)
    }
}