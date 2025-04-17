//! Series module - Core data structure for columns in PandRS
//!
//! This module provides the Series type, which is a one-dimensional array
//! of homogeneous data with labels along one axis.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::{Add, Div, Mul, Sub};
use std::iter::Sum;
use num_traits::NumCast;

use crate::error::{PandRSError, Result};
use crate::index::{Index, RangeIndex};
use crate::na::NA;

// Series type variants
pub mod categorical;
pub mod na_series;

// Re-exports
pub use categorical::{Categorical, CategoricalOrder, StringCategorical};
pub use na_series::NASeries;

/// Series data structure
///
/// A Series is similar to a one-dimensional array with axis labels.
/// It provides functionality for storing and manipulating data.
#[derive(Debug, Clone)]
pub struct Series<T: Debug + Clone> {
    /// Series data values
    values: Vec<T>,
    /// Index for labeling the data
    index: RangeIndex,
    /// Optional name for the series
    name: Option<String>,
}

impl<T: Debug + Clone> Series<T> {
    /// Create a new Series from a vector
    ///
    /// # Arguments
    /// * `values` - Data values
    /// * `name` - Optional name for the series
    ///
    /// # Returns
    /// * `Result<Self>` - New Series instance
    ///
    /// # Example
    /// ```
    /// use pandrs::Series;
    ///
    /// let series = Series::new(vec![1, 2, 3], Some("data".to_string())).unwrap();
    /// assert_eq!(series.len(), 3);
    /// ```
    pub fn new(values: Vec<T>, name: Option<String>) -> Result<Self> {
        let len = values.len();
        let index = RangeIndex::from_range(0..len)?;

        Ok(Series {
            values,
            index,
            name,
        })
    }

    /// Create a Series with custom index
    pub fn with_index<I>(values: Vec<T>, index: Index<I>, name: Option<String>) -> Result<Self>
    where
        I: Debug + Clone + Eq + std::hash::Hash + std::fmt::Display,
    {
        if values.len() != index.len() {
            return Err(PandRSError::Consistency(format!(
                "Length of values ({}) does not match length of index ({})",
                values.len(),
                index.len()
            )));
        }

        // Currently only supporting integer indices
        let range_index = RangeIndex::from_range(0..values.len())?;

        Ok(Series {
            values,
            index: range_index,
            name,
        })
    }

    /// Get the length of the Series
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if the Series is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get value by position
    pub fn get(&self, pos: usize) -> Option<&T> {
        self.values.get(pos)
    }

    /// Get the array of values
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Get the name
    pub fn name(&self) -> Option<&String> {
        self.name.as_ref()
    }
    
    /// Set the name (mutable reference version)
    pub fn set_name(&mut self, name: String) {
        self.name = Some(name);
    }

    /// Get the index
    pub fn index(&self) -> &RangeIndex {
        &self.index
    }

    /// Set the name and return self (builder pattern)
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }
    
    /// Convert Series<T> to Series<String>
    pub fn to_string_series(&self) -> Result<Series<String>> 
    where 
        T: std::fmt::Display,
    {
        let string_values: Vec<String> = self.values.iter().map(|v| v.to_string()).collect();
        Series::new(string_values, self.name.clone())
    }
}

// Specialized implementation for numeric Series
impl<T> Series<T>
where
    T: Debug
        + Clone
        + Copy
        + Sum<T>
        + PartialOrd
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + NumCast
        + Default,
{
    /// Calculate the sum
    pub fn sum(&self) -> T {
        if self.values.is_empty() {
            T::default()
        } else {
            self.values.iter().copied().sum()
        }
    }

    /// Calculate the mean
    pub fn mean(&self) -> Result<T> {
        if self.values.is_empty() {
            return Err(PandRSError::Consistency(
                "Cannot calculate mean of an empty Series".to_string(),
            ));
        }

        let sum = self.sum();
        let count = match num_traits::cast(self.len()) {
            Some(n) => n,
            None => {
                return Err(PandRSError::Cast(
                    "Cannot cast length to numeric type".to_string(),
                ))
            }
        };

        Ok(sum / count)
    }

    /// Calculate the minimum value
    pub fn min(&self) -> Result<T> {
        if self.values.is_empty() {
            return Err(PandRSError::Consistency(
                "Cannot calculate minimum of an empty Series".to_string(),
            ));
        }

        let min = self
            .values
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
            .unwrap();

        Ok(min)
    }

    /// Calculate the maximum value
    pub fn max(&self) -> Result<T> {
        if self.values.is_empty() {
            return Err(PandRSError::Consistency(
                "Cannot calculate maximum of an empty Series".to_string(),
            ));
        }

        let max = self
            .values
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
            .unwrap();

        Ok(max)
    }
}

// Default trait implementation for Series
impl<T> Default for Series<T>
where
    T: Debug + Clone,
    Vec<T>: Default,
{
    fn default() -> Self {
        Series {
            values: Vec::default(),
            index: RangeIndex::from_range(0..0).unwrap(),
            name: None,
        }
    }
}

// Add statistical functions to numeric Series (f64 specialized version)
impl Series<f64> {
    /// Calculate variance (sample variance)
    /// 
    /// Calculates the unbiased variance (divided by n-1)
    pub fn var(&self) -> Result<f64> {
        if self.values.is_empty() {
            return Err(PandRSError::Consistency(
                "Cannot calculate variance of an empty Series".to_string(),
            ));
        }
        
        if self.values.len() == 1 {
            return Err(PandRSError::Consistency(
                "Variance is not defined for a Series with only one element".to_string(),
            ));
        }
        
        let mean = self.mean()?;
        let sum_squared_diff: f64 = self.values.iter()
            .map(|&v| (v - mean).powi(2))
            .sum();
        
        // Calculate the unbiased variance (sample variance)
        Ok(sum_squared_diff / (self.values.len() - 1) as f64)
    }
    
    /// Calculate population variance
    /// 
    /// Calculates the population variance (divided by n)
    pub fn var_pop(&self) -> Result<f64> {
        if self.values.is_empty() {
            return Err(PandRSError::Consistency(
                "Cannot calculate variance of an empty Series".to_string(),
            ));
        }
        
        let mean = self.mean()?;
        let sum_squared_diff: f64 = self.values.iter()
            .map(|&v| (v - mean).powi(2))
            .sum();
        
        // Calculate the population variance
        Ok(sum_squared_diff / self.values.len() as f64)
    }
    
    /// Calculate standard deviation (sample standard deviation)
    pub fn std(&self) -> Result<f64> {
        Ok(self.var()?.sqrt())
    }
    
    /// Calculate population standard deviation
    pub fn std_pop(&self) -> Result<f64> {
        Ok(self.var_pop()?.sqrt())
    }
    
    /// Calculate quantile
    /// 
    /// q: Quantile between 0.0 and 1.0 (0.5 is the median)
    pub fn quantile(&self, q: f64) -> Result<f64> {
        if self.values.is_empty() {
            return Err(PandRSError::Consistency(
                "Cannot calculate quantile of an empty Series".to_string(),
            ));
        }
        
        if q < 0.0 || q > 1.0 {
            return Err(PandRSError::InvalidInput(
                "Quantile must be between 0.0 and 1.0".to_string(),
            ));
        }
        
        let mut sorted_values = self.values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        if q == 0.0 {
            return Ok(sorted_values[0]);
        }
        
        if q == 1.0 {
            return Ok(sorted_values[sorted_values.len() - 1]);
        }
        
        let pos = q * (sorted_values.len() - 1) as f64;
        let idx_lower = pos.floor() as usize;
        let idx_upper = pos.ceil() as usize;
        
        if idx_lower == idx_upper {
            Ok(sorted_values[idx_lower])
        } else {
            let weight_upper = pos - idx_lower as f64;
            let weight_lower = 1.0 - weight_upper;
            Ok(weight_lower * sorted_values[idx_lower] + weight_upper * sorted_values[idx_upper])
        }
    }
    
    /// Calculate median
    pub fn median(&self) -> Result<f64> {
        self.quantile(0.5)
    }
    
    /// Calculate kurtosis
    pub fn kurtosis(&self) -> Result<f64> {
        if self.values.len() < 4 {
            return Err(PandRSError::Consistency(
                "At least 4 data points are required to calculate kurtosis".to_string(),
            ));
        }
        
        let mean = self.mean()?;
        let n = self.values.len() as f64;
        
        let m4: f64 = self.values.iter()
            .map(|&x| (x - mean).powi(4))
            .sum::<f64>() / n;
            
        let m2: f64 = self.values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / n;
            
        // Excess kurtosis (kurtosis of normal distribution minus 3)
        let kurtosis = m4 / m2.powi(2) - 3.0;
        
        Ok(kurtosis)
    }
    
    /// Calculate skewness
    pub fn skewness(&self) -> Result<f64> {
        if self.values.len() < 3 {
            return Err(PandRSError::Consistency(
                "At least 3 data points are required to calculate skewness".to_string(),
            ));
        }
        
        let mean = self.mean()?;
        let n = self.values.len() as f64;
        
        let m3: f64 = self.values.iter()
            .map(|&x| (x - mean).powi(3))
            .sum::<f64>() / n;
            
        let m2: f64 = self.values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / n;
            
        let skewness = m3 / m2.powf(1.5);
        
        Ok(skewness)
    }
    
    /// Calculate covariance with another Series
    pub fn cov(&self, other: &Series<f64>) -> Result<f64> {
        if self.values.is_empty() || other.values.is_empty() {
            return Err(PandRSError::Consistency(
                "Cannot calculate covariance with an empty Series".to_string(),
            ));
        }
        
        if self.values.len() != other.values.len() {
            return Err(PandRSError::Consistency(
                "Both Series must have the same length to calculate covariance".to_string(),
            ));
        }
        
        let mean_x = self.mean()?;
        let mean_y = other.mean()?;
        let n = self.values.len() as f64;
        
        let sum_xy = self.values.iter().zip(other.values.iter())
            .map(|(&x, &y)| (x - mean_x) * (y - mean_y))
            .sum::<f64>();
            
        // Use unbiased covariance estimator
        Ok(sum_xy / (n - 1.0))
    }
    
    /// Calculate correlation coefficient with another Series
    pub fn corr(&self, other: &Series<f64>) -> Result<f64> {
        let cov = self.cov(other)?;
        let std_x = self.std()?;
        let std_y = other.std()?;
        
        if std_x == 0.0 || std_y == 0.0 {
            return Err(PandRSError::Consistency(
                "Cannot calculate correlation coefficient when standard deviation is 0".to_string(),
            ));
        }
        
        Ok(cov / (std_x * std_y))
    }
    
    /// Get descriptive statistics in one call
    pub fn describe(&self) -> Result<crate::stats::DescriptiveStats> {
        crate::stats::describe(self.values())
    }
}

impl Series<String> {
    /// Convert string Series to vector of numeric `NA<f64>`
    /// 
    /// Converts each element to a number. Returns NA if conversion is not possible.
    pub fn to_numeric_vec(&self) -> Result<Vec<NA<f64>>> {
        let mut result = Vec::with_capacity(self.len());
        
        for value in &self.values {
            match value.parse::<f64>() {
                Ok(num) => result.push(NA::Value(num)),
                Err(_) => result.push(NA::NA),
            }
        }
        
        Ok(result)
    }
    
    /// Apply a function to each element
    pub fn apply_map<F>(&self, f: F) -> Series<String>
    where
        F: Fn(&String) -> String,
    {
        let transformed: Vec<String> = self.values.iter()
            .map(|v| f(v))
            .collect();
        
        Series {
            values: transformed,
            index: self.index.clone(),
            name: self.name.clone(),
        }
    }
}
