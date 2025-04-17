//! Core structure definition and basic functionality for OptimizedDataFrame

use std::collections::HashMap;
use std::fmt::{self, Debug, Display};
use crate::column::{Column, ColumnTrait, ColumnType, Int64Column, Float64Column, StringColumn, BooleanColumn};
use crate::error::{Error, Result};
use crate::index::{DataFrameIndex, IndexTrait, Index};

/// Optimized DataFrame implementation
/// Uses column-oriented storage for fast data processing
#[derive(Clone)]
pub struct OptimizedDataFrame {
    // Column data
    pub(crate) columns: Vec<Column>,
    // Column name → index mapping
    pub(crate) column_indices: HashMap<String, usize>,
    // Column order
    pub(crate) column_names: Vec<String>,
    // Row count
    pub(crate) row_count: usize,
    // Index (optional)
    pub(crate) index: Option<DataFrameIndex<String>>,
}

/// Structure representing a view (reference) to a column
#[derive(Clone)]
pub struct ColumnView {
    pub(crate) column: Column,
}

impl Debug for OptimizedDataFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Maximum display rows
        const MAX_ROWS: usize = 10;
        
        if self.columns.is_empty() {
            return write!(f, "OptimizedDataFrame (0 rows x 0 columns)");
        }
        
        writeln!(f, "OptimizedDataFrame ({} rows x {} columns):", self.row_count, self.columns.len())?;
        
        // Display column headers
        write!(f, "{:<5} |", "idx")?;
        for name in &self.column_names {
            write!(f, " {:<15} |", name)?;
        }
        writeln!(f)?;
        
        // Separator line
        write!(f, "{:-<5}-+", "")?;
        for _ in &self.column_names {
            write!(f, "-{:-<15}-+", "")?;
        }
        writeln!(f)?;
        
        // Display up to MAX_ROWS rows
        let display_rows = std::cmp::min(self.row_count, MAX_ROWS);
        for i in 0..display_rows {
            if let Some(ref idx) = self.index {
                let idx_value = match idx {
                    DataFrameIndex::Simple(ref simple_idx) => {
                        if i < simple_idx.len() {
                            simple_idx.get_value(i).map(|s| s.to_string()).unwrap_or_else(|| i.to_string())
                        } else {
                            i.to_string()
                        }
                    },
                    DataFrameIndex::Multi(_) => i.to_string()
                };
                write!(f, "{:<5} |", idx_value)?;
            } else {
                write!(f, "{:<5} |", i)?;
            }
            
            for col_idx in 0..self.columns.len() {
                let col = &self.columns[col_idx];
                let value = match col {
                    Column::Int64(col) => {
                        if let Ok(Some(val)) = col.get(i) {
                            format!("{}", val)
                        } else {
                            "NULL".to_string()
                        }
                    },
                    Column::Float64(col) => {
                        if let Ok(Some(val)) = col.get(i) {
                            format!("{:.3}", val)
                        } else {
                            "NULL".to_string()
                        }
                    },
                    Column::String(col) => {
                        if let Ok(Some(val)) = col.get(i) {
                            format!("\"{}\"", val)
                        } else {
                            "NULL".to_string()
                        }
                    },
                    Column::Boolean(col) => {
                        if let Ok(Some(val)) = col.get(i) {
                            format!("{}", val)
                        } else {
                            "NULL".to_string()
                        }
                    },
                };
                write!(f, " {:<15} |", value)?;
            }
            writeln!(f)?;
        }
        
        // Ellipsis for additional rows
        if self.row_count > MAX_ROWS {
            writeln!(f, "... ({} more rows)", self.row_count - MAX_ROWS)?;
        }
        
        Ok(())
    }
}

impl OptimizedDataFrame {
    /// Create a new empty DataFrame
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            column_indices: HashMap::new(),
            column_names: Vec::new(),
            row_count: 0,
            index: None,
        }
    }
    
    /// Create DataFrame with string index
    pub fn with_index(index: Index<String>) -> Self {
        Self {
            columns: Vec::new(),
            column_indices: HashMap::new(),
            column_names: Vec::new(),
            row_count: index.len(),
            index: Some(DataFrameIndex::<String>::from_simple(index)),
        }
    }
    
    /// Create DataFrame with multi-index
    pub fn with_multi_index(index: crate::index::MultiIndex<String>) -> Self {
        Self {
            columns: Vec::new(),
            column_indices: HashMap::new(),
            column_names: Vec::new(),
            row_count: index.len(),
            index: Some(DataFrameIndex::<String>::from_multi(index)),
        }
    }
    
    /// Create DataFrame with range index
    pub fn with_range_index(range: std::ops::Range<usize>) -> Result<Self> {
        let range_idx = Index::<usize>::from_range(range)?;
        // Convert numeric index to string index
        let string_values: Vec<String> = range_idx.values().iter().map(|i| i.to_string()).collect();
        let string_idx = Index::<String>::new(string_values)?;
        Ok(Self::with_index(string_idx))
    }
    
    /// Get row count
    pub fn row_count(&self) -> usize {
        self.row_count
    }
    
    /// Get column count
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }
    
    /// Get list of column names
    pub fn column_names(&self) -> &[String] {
        &self.column_names
    }
    
    /// Check if specified column exists
    pub fn contains_column(&self, name: &str) -> bool {
        self.column_indices.contains_key(name)
    }
}
