//! Module providing machine learning functionality
//! 
//! This module provides transformation pipelines and utilities
//! for using PandRS data structures with machine learning algorithms.
//!
//! Note: This module is implemented using the optimized OptimizedDataFrame.

pub mod pipeline;
pub mod preprocessing;
pub mod metrics;
pub mod dimension_reduction;
pub mod models;
pub mod clustering;
pub mod anomaly_detection;

// External crates may be re-exported in the future
// pub use linfa;