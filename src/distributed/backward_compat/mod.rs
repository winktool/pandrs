//! # Backward Compatibility Layer for Distributed Module
//!
//! This module provides backward compatibility with code that was using the
//! previous organization of the distributed module.

// Re-export types that were previously in this module directly
#[allow(deprecated)]
pub use crate::distributed::expr::backward_compat::*;

// Re-export config types for backward compatibility
#[deprecated(since = "0.1.0", note = "Use distributed::core::DistributedConfig instead")]
pub use crate::distributed::core::{
    DistributedConfig,
    ExecutorType,
    MemoryManager,
};

// Re-export context types for backward compatibility
#[deprecated(since = "0.1.0", note = "Use distributed::core::DistributedContext instead")]
pub use crate::distributed::core::{
    DistributedContext,
    ToDistributed,
};

// Re-export dataframe types for backward compatibility
#[deprecated(since = "0.1.0", note = "Use distributed::core::DistributedDataFrame instead")]
pub use crate::distributed::core::{
    DistributedDataFrame,
    DistributedSeriesView,
    DataFrameOperations,
};

// Re-export statistics types for backward compatibility
#[deprecated(since = "0.1.0", note = "Use distributed::core::statistics instead")]
pub use crate::distributed::core::statistics::{
    DistributedStatistics,
    ColumnStatistics,
};

// Re-export execution types for backward compatibility
#[deprecated(since = "0.1.0", note = "Use distributed::execution instead")]
pub use crate::distributed::execution::{
    ExecutionEngine,
    ExecutionContext,
    ExecutionPlan,
    ExecutionResult,
    ExecutionMetrics,
    Operation,
    JoinType,
    AggregateExpr,
    SortExpr,
};

// Re-export schema validator for backward compatibility
#[cfg(feature = "distributed")]
#[deprecated(since = "0.1.0", note = "Use distributed::expr::schema_validator instead")]
pub use crate::distributed::schema_validator::SchemaValidator;

// Re-export window operations for backward compatibility
#[cfg(feature = "distributed")]
#[deprecated(since = "0.1.0", note = "Use distributed::window instead")]
pub use crate::distributed::window::{
    WindowFrameBoundary, 
    WindowFrameType, 
    WindowFrame, 
    WindowFunction, 
    WindowFunctionExt,
    functions as window_functions,
};

// Re-export explain for backward compatibility
#[cfg(feature = "distributed")]
#[deprecated(since = "0.1.0", note = "Use distributed::explain instead")]
pub use crate::distributed::explain::{
    ExplainOptions, 
    ExplainFormat, 
    PlanNode, 
    explain_plan
};

// Re-export fault tolerance for backward compatibility
#[cfg(feature = "distributed")]
#[deprecated(since = "0.1.0", note = "Use distributed::fault_tolerance instead")]
pub use crate::distributed::fault_tolerance::{
    FaultToleranceHandler, 
    FaultTolerantContext, 
    RetryPolicy, 
    RecoveryStrategy,
    FailureType, 
    FailureInfo, 
    RecoveryAction, 
    ExecutionCheckpoint, 
    CheckpointManager,
};