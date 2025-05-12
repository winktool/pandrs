//! # Query Plan Explanation
//!
//! This module provides functionality for explaining query plans, helping users
//! understand how their queries will be executed.

pub mod core;
pub mod format;
pub mod visualize;
pub mod conversion;
pub mod backward_compat;

// Re-export core types for easier access
pub use self::core::{ExplainFormat, ExplainOptions, PlanNode};

// Re-export conversion functions
pub use self::conversion::{explain_plan, execution_plan_to_plan_node};