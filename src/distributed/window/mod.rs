//! # Window Function Support for Distributed Processing
//!
//! This module provides window function capabilities for distributed DataFrames,
//! enabling advanced analytics like rolling calculations, cumulative aggregations,
//! and rank-based statistics.

pub mod core;
pub mod operations;
pub mod functions;
pub mod backward_compat;

// Re-export core types for easier access
pub use self::core::{WindowFrameBoundary, WindowFrameType, WindowFrame, WindowFunction};

// Re-export operations
pub use self::operations::WindowFunctionExt;

// Re-export functions module
pub use self::functions;