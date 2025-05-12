//! Backward compatibility module for statistics
//!
//! This module provides backward compatibility for code that uses the old
//! statistics module structure. It re-exports types and functions from
//! the new module structure with appropriate deprecation notices.

#[allow(deprecated)]
pub mod descriptive {
    //! Backward compatibility for descriptive statistics
    
    use crate::error::Result;
    
    /// Calculate basic statistics for data (backward compatibility)
    #[deprecated(
        since = "0.1.0-alpha.2",
        note = "Use `pandrs::stats::descriptive::describe` instead"
    )]
    pub fn describe<T: AsRef<[f64]>>(data: T) -> Result<crate::stats::DescriptiveStats> {
        // Forward to new implementation
        crate::stats::descriptive::describe_impl(data.as_ref())
    }

    /// Calculate correlation coefficient (backward compatibility)
    #[deprecated(
        since = "0.1.0-alpha.2",
        note = "Use `pandrs::stats::descriptive::correlation` instead"
    )]
    pub fn correlation<T: AsRef<[f64]>, U: AsRef<[f64]>>(x: T, y: U) -> Result<f64> {
        // Forward to new implementation
        crate::stats::descriptive::correlation_impl(x.as_ref(), y.as_ref())
    }
    
    /// Calculate variance (backward compatibility)
    #[deprecated(
        since = "0.1.0-alpha.2",
        note = "Use `pandrs::stats::descriptive::variance` instead"
    )]
    pub fn variance<T: AsRef<[f64]>>(data: T, ddof: usize) -> Result<f64> {
        // Forward to new implementation
        crate::stats::descriptive::variance(data, ddof)
    }
    
    /// Calculate standard deviation (backward compatibility)
    #[deprecated(
        since = "0.1.0-alpha.2",
        note = "Use `pandrs::stats::descriptive::std_dev` instead"
    )]
    pub fn std_dev<T: AsRef<[f64]>>(data: T, ddof: usize) -> Result<f64> {
        // Forward to new implementation
        crate::stats::descriptive::std_dev(data, ddof)
    }
}

#[allow(deprecated)]
pub mod inference {
    //! Backward compatibility for inferential statistics
    
    use crate::error::Result;
    
    /// Perform two-sample t-test (backward compatibility)
    #[deprecated(
        since = "0.1.0-alpha.2",
        note = "Use `pandrs::stats::inference::ttest` instead"
    )]
    pub fn ttest<T: AsRef<[f64]>, U: AsRef<[f64]>>(
        sample1: T,
        sample2: U,
        alpha: f64,
        equal_var: bool,
    ) -> Result<crate::stats::TTestResult> {
        // Forward to new implementation
        crate::stats::inference::ttest_impl(sample1.as_ref(), sample2.as_ref(), alpha, equal_var)
    }
}

#[allow(deprecated)]
pub mod regression {
    //! Backward compatibility for regression analysis
    
    use crate::dataframe::DataFrame;
    use crate::error::Result;
    
    /// Perform linear regression analysis (backward compatibility)
    #[deprecated(
        since = "0.1.0-alpha.2",
        note = "Use `pandrs::stats::regression::linear_regression` instead"
    )]
    pub fn linear_regression(
        df: &DataFrame,
        y_column: &str,
        x_columns: &[&str],
    ) -> Result<crate::stats::LinearRegressionResult> {
        // Forward to new implementation
        crate::stats::regression::linear_regression_impl(df, y_column, x_columns)
    }
}

#[allow(deprecated)]
pub mod sampling {
    //! Backward compatibility for sampling methods
    
    use crate::dataframe::DataFrame;
    use crate::error::Result;
    
    /// Perform random sampling (backward compatibility)
    #[deprecated(
        since = "0.1.0-alpha.2",
        note = "Use `pandrs::stats::sampling::sample` instead"
    )]
    pub fn sample(
        df: &DataFrame,
        fraction: f64,
        replace: bool,
    ) -> Result<DataFrame> {
        // Forward to new implementation
        crate::stats::sampling::sample_impl(df, fraction, replace)
    }
    
    /// Generate bootstrap samples (backward compatibility)
    #[deprecated(
        since = "0.1.0-alpha.2",
        note = "Use `pandrs::stats::sampling::bootstrap` instead"
    )]
    pub fn bootstrap<T: AsRef<[f64]>>(
        data: T,
        n_samples: usize,
    ) -> Result<Vec<Vec<f64>>> {
        // Forward to new implementation
        crate::stats::sampling::bootstrap_impl(data.as_ref(), n_samples)
    }
}

#[allow(deprecated)]
pub mod categorical {
    //! Backward compatibility for categorical data analysis
    
    use crate::dataframe::DataFrame;
    use crate::error::Result;
    
    /// Create a contingency table from two categorical columns (backward compatibility)
    #[deprecated(
        since = "0.1.0-alpha.2",
        note = "Use `pandrs::stats::categorical::contingency_table_from_df` instead"
    )]
    pub fn contingency_table_from_df(
        df: &DataFrame,
        col1: &str,
        col2: &str,
    ) -> Result<crate::stats::ContingencyTable> {
        // Forward to new implementation
        crate::stats::categorical::dataframe_contingency_table(df, col1, col2)
    }
}