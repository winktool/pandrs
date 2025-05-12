//! # Execution Engines for Distributed Processing
//!
//! This module provides implementations of execution engines for distributed processing.

// DataFusion engine
pub mod datafusion;

// Ballista engine
pub mod ballista;

// Re-export engine implementations
pub use datafusion::DataFusionEngine;
pub use ballista::BallistaEngine;