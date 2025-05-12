//! Machine learning model evaluation metrics
//!
//! This module provides metrics for evaluating the performance of machine learning models,
//! including regression metrics and classification metrics.

pub mod regression;
pub mod classification;

// Add backward compatibility layer
#[deprecated(
    since = "0.1.0-alpha.2",
    note = "Use regression and classification modules directly instead"
)]
pub use regression::{
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    explained_variance_score,
};

#[deprecated(
    since = "0.1.0-alpha.2",
    note = "Use regression and classification modules directly instead"
)]
pub use classification::{
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
};