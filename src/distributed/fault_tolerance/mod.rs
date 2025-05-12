//! # Fault Tolerance for Distributed Processing
//!
//! This module provides fault tolerance mechanisms for distributed query execution,
//! ensuring that queries can complete even when parts of the distributed system fail.

pub mod core;
pub mod recovery;
pub mod checkpoint;
pub mod backward_compat;

// Re-export core types for easier access
pub use self::core::{RetryPolicy, FailureType, FailureInfo, RecoveryStrategy};

// Re-export recovery types
pub use self::recovery::{RecoveryAction, FaultToleranceHandler, FaultTolerantExecutor, FaultTolerantContext};

// Re-export checkpoint types
pub use self::checkpoint::{ExecutionCheckpoint, CheckpointManager};