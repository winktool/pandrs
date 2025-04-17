//! Grouping and aggregation functionality for OptimizedDataFrame

use std::collections::HashMap;

use super::core::OptimizedDataFrame;
use crate::column::{Column, ColumnType, Float64Column, Int64Column, StringColumn, BooleanColumn};
use crate::error::{Error, Result};

/// Enumeration representing aggregation operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateOp {
    /// Sum
    Sum,
    /// Mean
    Mean,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
    /// Count
    Count,
}

/// Structure representing grouping results
pub struct GroupBy<'a> {
    /// Original DataFrame
    df: &'a OptimizedDataFrame,
    /// Grouping key columns
    group_by_columns: Vec<String>,
    /// Row indices for each group
    groups: HashMap<Vec<String>, Vec<usize>>,
}

impl OptimizedDataFrame {
    /// Group DataFrame
    ///
    /// # Arguments
    /// * `columns` - Column names for grouping
    ///
    /// # Returns
    /// * `Result<GroupBy>` - Grouping results
    pub fn group_by<I, S>(&self, columns: I) -> Result<GroupBy<'_>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let group_by_columns: Vec<String> = columns
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect();
        
        // Verify existence of each column
        for column in &group_by_columns {
            if !self.column_indices.contains_key(column) {
                return Err(Error::ColumnNotFound(column.clone()));
            }
        }
        
        // Create grouping keys
        let mut groups: HashMap<Vec<String>, Vec<usize>> = HashMap::new();
        
        for row_idx in 0..self.row_count {
            let mut key = Vec::with_capacity(group_by_columns.len());
            
            for col_name in &group_by_columns {
                let col_idx = self.column_indices[col_name];
                let col = &self.columns[col_idx];
                
                let key_part = match col {
                    Column::Int64(int_col) => {
                        if let Ok(Some(val)) = int_col.get(row_idx) {
                            val.to_string()
                        } else {
                            "NULL".to_string()
                        }
                    },
                    Column::Float64(float_col) => {
                        if let Ok(Some(val)) = float_col.get(row_idx) {
                            val.to_string()
                        } else {
                            "NULL".to_string()
                        }
                    },
                    Column::String(str_col) => {
                        if let Ok(Some(val)) = str_col.get(row_idx) {
                            val.to_string()
                        } else {
                            "NULL".to_string()
                        }
                    },
                    Column::Boolean(bool_col) => {
                        if let Ok(Some(val)) = bool_col.get(row_idx) {
                            val.to_string()
                        } else {
                            "NULL".to_string()
                        }
                    },
                };
                
                key.push(key_part);
            }
            
            groups.entry(key).or_default().push(row_idx);
        }
        
        Ok(GroupBy {
            df: self,
            group_by_columns,
            groups,
        })
    }
    
    /// Group DataFrame using parallel processing
    ///
    /// # Arguments
    /// * `group_by_columns` - Column names for grouping
    ///
    /// # Returns
    /// * `Result<HashMap<String, Self>>` - Grouping results (map of keys and DataFrames)
    pub fn par_groupby(&self, group_by_columns: &[&str]) -> Result<HashMap<String, Self>> {
        use rayon::prelude::*;
        use std::collections::hash_map::Entry;
        use std::sync::{Arc, Mutex};
        
        // Optimization threshold based on data size
        const PARALLEL_THRESHOLD: usize = 50_000;
        
        // Get column indices for grouping keys
        let mut group_col_indices = Vec::with_capacity(group_by_columns.len());
        for &col_name in group_by_columns {
            let col_idx = self.column_indices.get(col_name)
                .ok_or_else(|| Error::ColumnNotFound(col_name.to_string()))?;
            group_col_indices.push(*col_idx);
        }
        
        // Generate group keys and group each row's index
        let groups: HashMap<String, Vec<usize>> = if self.row_count < PARALLEL_THRESHOLD {
            // Serial processing is more efficient for small data
            let mut groups = HashMap::new();
            
            for row_idx in 0..self.row_count {
                // Generate group key for this row
                let mut key_parts = Vec::with_capacity(group_col_indices.len());
                
                for &col_idx in &group_col_indices {
                    let column = &self.columns[col_idx];
                    let part = match column {
                        Column::Int64(col) => {
                            if let Ok(Some(val)) = col.get(row_idx) {
                                val.to_string()
                            } else {
                                "NA".to_string()
                            }
                        },
                        Column::Float64(col) => {
                            if let Ok(Some(val)) = col.get(row_idx) {
                                val.to_string()
                            } else {
                                "NA".to_string()
                            }
                        },
                        Column::String(col) => {
                            if let Ok(Some(val)) = col.get(row_idx) {
                                val.to_string()
                            } else {
                                "NA".to_string()
                            }
                        },
                        Column::Boolean(col) => {
                            if let Ok(Some(val)) = col.get(row_idx) {
                                val.to_string()
                            } else {
                                "NA".to_string()
                            }
                        },
                    };
                    key_parts.push(part);
                }
                
                let group_key = key_parts.join("_");
                
                match groups.entry(group_key) {
                    Entry::Vacant(e) => { e.insert(vec![row_idx]); },
                    Entry::Occupied(mut e) => { e.get_mut().push(row_idx); }
                }
            }
            
            groups
        } else {
            // For large data, use parallel processing + lock-free approach
            // 1. Create local group maps in parallel
            // 2. Merge them
            let chunk_size = (self.row_count / rayon::current_num_threads()).max(1000);
            
            // Step 1: Create local intermediate group maps in parallel
            let local_maps: Vec<HashMap<String, Vec<usize>>> = 
                (0..self.row_count).collect::<Vec<_>>()
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut local_groups = HashMap::new();
                    
                    for &row_idx in chunk {
                        // Generate group key for this row
                        let mut key_parts = Vec::with_capacity(group_col_indices.len());
                        
                        for &col_idx in &group_col_indices {
                            let column = &self.columns[col_idx];
                            let part = match column {
                                Column::Int64(col) => {
                                    if let Ok(Some(val)) = col.get(row_idx) {
                                        val.to_string()
                                    } else {
                                        "NA".to_string()
                                    }
                                },
                                Column::Float64(col) => {
                                    if let Ok(Some(val)) = col.get(row_idx) {
                                        val.to_string()
                                    } else {
                                        "NA".to_string()
                                    }
                                },
                                Column::String(col) => {
                                    if let Ok(Some(val)) = col.get(row_idx) {
                                        val.to_string()
                                    } else {
                                        "NA".to_string()
                                    }
                                },
                                Column::Boolean(col) => {
                                    if let Ok(Some(val)) = col.get(row_idx) {
                                        val.to_string()
                                    } else {
                                        "NA".to_string()
                                    }
                                },
                            };
                            key_parts.push(part);
                        }
                        
                        let group_key = key_parts.join("_");
                        
                        match local_groups.entry(group_key) {
                            Entry::Vacant(e) => { e.insert(vec![row_idx]); },
                            Entry::Occupied(mut e) => { e.get_mut().push(row_idx); }
                        }
                    }
                    
                    local_groups
                })
                .collect();
            
            // Step 2: Merge intermediate maps
            let mut merged_groups = HashMap::new();
            for local_map in local_maps {
                for (key, indices) in local_map {
                    match merged_groups.entry(key) {
                        Entry::Vacant(e) => { e.insert(indices); },
                        Entry::Occupied(mut e) => { e.get_mut().extend(indices); }
                    }
                }
            }
            
            merged_groups
        };
        
        // Efficiently create DataFrames for each group
        let result = if groups.len() < 100 || self.row_count < PARALLEL_THRESHOLD {
            // Use serial processing for small data or when group count is small
            let mut result = HashMap::with_capacity(groups.len());
            for (key, indices) in groups {
                let group_df = self.filter_by_indices(&indices)?;
                result.insert(key, group_df);
            }
            result
        } else {
            // Parallelize group processing for large data
            // Process each group in parallel and safely aggregate results
            let result_mutex = Arc::new(Mutex::new(HashMap::with_capacity(groups.len())));
            
            // Adjust chunk size to minimize overhead
            let chunk_size = (groups.len() / rayon::current_num_threads()).max(10);
            
            // Create list of groups and split into chunks for parallel processing
            let group_items: Vec<(String, Vec<usize>)> = groups.into_iter().collect();
            
            group_items.par_chunks(chunk_size)
                .for_each(|chunk| {
                    // Temporarily store processing results for each chunk
                    let mut local_results = HashMap::new();
                    
                    for (key, indices) in chunk {
                        if let Ok(group_df) = self.filter_by_indices(indices) {
                            local_results.insert(key.clone(), group_df);
                        }
                    }
                    
                    // Merge results into the main HashMap
                    if let Ok(mut result_map) = result_mutex.lock() {
                        for (key, df) in local_results {
                            result_map.insert(key, df);
                        }
                    }
                });
            
            // Get final results
            match Arc::try_unwrap(result_mutex) {
                Ok(mutex) => mutex.into_inner().unwrap_or_default(),
                Err(_) => HashMap::new(), // If failed to unwrap arc
            }
        };
        
        Ok(result)
    }
}

impl<'a> GroupBy<'a> {
    /// Execute aggregation operations for each group
    ///
    /// # Arguments
    /// * `aggregations` - List of aggregation operations (column name, operation, result column name)
    ///
    /// # Returns
    /// * `Result<OptimizedDataFrame>` - DataFrame containing aggregation results
    pub fn aggregate<I>(&self, aggregations: I) -> Result<OptimizedDataFrame>
    where
        I: IntoIterator<Item = (String, AggregateOp, String)>,
    {
        let aggregations: Vec<(String, AggregateOp, String)> = aggregations.into_iter().collect();
        
        // Verify existence of each column to be aggregated
        for (col_name, _, _) in &aggregations {
            if !self.df.column_indices.contains_key(col_name) {
                return Err(Error::ColumnNotFound(col_name.clone()));
            }
        }
        
        // Storage for aggregation results
        let mut result = OptimizedDataFrame::new();
        
        // Data for grouping key columns
        let mut group_key_data: HashMap<String, Vec<String>> = HashMap::new();
        for key in self.group_by_columns.iter() {
            group_key_data.insert(key.clone(), Vec::new());
        }
        
        // Data for aggregation result columns
        let mut agg_result_data: HashMap<String, Vec<f64>> = HashMap::new();
        for (_, _, alias) in &aggregations {
            agg_result_data.insert(alias.clone(), Vec::new());
        }
        
        // Execute aggregation for each group
        for (key, row_indices) in &self.groups {
            // Add values for grouping keys
            for (i, col_name) in self.group_by_columns.iter().enumerate() {
                group_key_data.get_mut(col_name).unwrap().push(key[i].clone());
            }
            
            // Execute aggregation operations
            for (col_name, op, alias) in &aggregations {
                let col_idx = self.df.column_indices[col_name];
                let col = &self.df.columns[col_idx];
                
                let result_value = match (col, op) {
                    (Column::Int64(int_col), AggregateOp::Sum) => {
                        let mut sum = 0;
                        for &idx in row_indices {
                            if let Ok(Some(val)) = int_col.get(idx) {
                                sum += val;
                            }
                        }
                        sum as f64
                    },
                    (Column::Int64(int_col), AggregateOp::Mean) => {
                        let mut sum = 0;
                        let mut count = 0;
                        for &idx in row_indices {
                            if let Ok(Some(val)) = int_col.get(idx) {
                                sum += val;
                                count += 1;
                            }
                        }
                        if count > 0 {
                            sum as f64 / count as f64
                        } else {
                            0.0
                        }
                    },
                    (Column::Int64(int_col), AggregateOp::Min) => {
                        let mut min = i64::MAX;
                        for &idx in row_indices {
                            if let Ok(Some(val)) = int_col.get(idx) {
                                min = min.min(val);
                            }
                        }
                        if min == i64::MAX {
                            0.0
                        } else {
                            min as f64
                        }
                    },
                    (Column::Int64(int_col), AggregateOp::Max) => {
                        let mut max = i64::MIN;
                        for &idx in row_indices {
                            if let Ok(Some(val)) = int_col.get(idx) {
                                max = max.max(val);
                            }
                        }
                        if max == i64::MIN {
                            0.0
                        } else {
                            max as f64
                        }
                    },
                    (Column::Float64(float_col), AggregateOp::Sum) => {
                        let mut sum = 0.0;
                        for &idx in row_indices {
                            if let Ok(Some(val)) = float_col.get(idx) {
                                sum += val;
                            }
                        }
                        sum
                    },
                    (Column::Float64(float_col), AggregateOp::Mean) => {
                        let mut sum = 0.0;
                        let mut count = 0;
                        for &idx in row_indices {
                            if let Ok(Some(val)) = float_col.get(idx) {
                                sum += val;
                                count += 1;
                            }
                        }
                        if count > 0 {
                            sum / count as f64
                        } else {
                            0.0
                        }
                    },
                    (Column::Float64(float_col), AggregateOp::Min) => {
                        let mut min = f64::INFINITY;
                        for &idx in row_indices {
                            if let Ok(Some(val)) = float_col.get(idx) {
                                min = min.min(val);
                            }
                        }
                        if min == f64::INFINITY {
                            0.0
                        } else {
                            min
                        }
                    },
                    (Column::Float64(float_col), AggregateOp::Max) => {
                        let mut max = f64::NEG_INFINITY;
                        for &idx in row_indices {
                            if let Ok(Some(val)) = float_col.get(idx) {
                                max = max.max(val);
                            }
                        }
                        if max == f64::NEG_INFINITY {
                            0.0
                        } else {
                            max
                        }
                    },
                    (_, AggregateOp::Count) => {
                        row_indices.len() as f64
                    },
                    _ => {
                        return Err(Error::OperationFailed(format!(
                            "Aggregation operation {:?} is not supported for column type {:?}",
                            op, col.column_type()
                        )));
                    }
                };
                
                agg_result_data.get_mut(alias).unwrap().push(result_value);
            }
        }
        
        // Add grouping key columns
        for (col_name, values) in group_key_data {
            // Add as string column
            let col = StringColumn::new(values);
            result.add_column(col_name, Column::String(col))?;
        }
        
        // Add aggregation result columns
        for (_, _, alias) in &aggregations {
            let values = agg_result_data.get(alias).unwrap();
            let col = Float64Column::new(values.clone());
            result.add_column(alias.clone(), Column::Float64(col))?;
        }
        
        Ok(result)
    }
    
    /// Aggregation shortcut method: Sum
    pub fn sum(&self, column: &str) -> Result<OptimizedDataFrame> {
        let agg_name = format!("{}_sum", column);
        self.aggregate(vec![(column.to_string(), AggregateOp::Sum, agg_name)])
    }
    
    /// Aggregation shortcut method: Mean
    pub fn mean(&self, column: &str) -> Result<OptimizedDataFrame> {
        let agg_name = format!("{}_mean", column);
        self.aggregate(vec![(column.to_string(), AggregateOp::Mean, agg_name)])
    }
    
    /// Aggregation shortcut method: Minimum value
    pub fn min(&self, column: &str) -> Result<OptimizedDataFrame> {
        let agg_name = format!("{}_min", column);
        self.aggregate(vec![(column.to_string(), AggregateOp::Min, agg_name)])
    }
    
    /// Aggregation shortcut method: Maximum value
    pub fn max(&self, column: &str) -> Result<OptimizedDataFrame> {
        let agg_name = format!("{}_max", column);
        self.aggregate(vec![(column.to_string(), AggregateOp::Max, agg_name)])
    }
    
    /// Aggregation shortcut method: Count
    pub fn count(&self, column: &str) -> Result<OptimizedDataFrame> {
        let agg_name = format!("{}_count", column);
        self.aggregate(vec![(column.to_string(), AggregateOp::Count, agg_name)])
    }
}
