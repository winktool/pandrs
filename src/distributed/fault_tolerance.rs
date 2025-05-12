//! # Fault Tolerance for Distributed Processing (Legacy)
//!
//! DEPRECATED: This file is maintained for backward compatibility only.
//! Please use the `distributed::fault_tolerance` module directory structure instead.
//!
//! This module provides fault tolerance mechanisms for distributed query execution,
//! ensuring that queries can complete even when parts of the distributed system fail.
//!
//! @deprecated

use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use crate::error::{Result, Error};
use super::execution::{ExecutionPlan, Operation, ExecutionResult};
use super::partition::PartitionSet;

/// Retry policy for failed operations
#[derive(Debug, Clone, Copy)]
pub enum RetryPolicy {
    /// No retry attempts
    None,
    /// Fixed interval between retry attempts
    Fixed {
        /// Maximum number of retry attempts
        max_retries: usize,
        /// Delay between retry attempts in milliseconds
        delay_ms: u64,
    },
    /// Exponential backoff between retry attempts
    Exponential {
        /// Maximum number of retry attempts
        max_retries: usize,
        /// Initial delay in milliseconds
        initial_delay_ms: u64,
        /// Maximum delay in milliseconds
        max_delay_ms: u64,
        /// Backoff factor
        backoff_factor: f64,
    },
}

impl RetryPolicy {
    /// Creates a default retry policy with 3 retries and 1 second delay
    pub fn default_fixed() -> Self {
        Self::Fixed {
            max_retries: 3,
            delay_ms: 1000,
        }
    }
    
    /// Creates a default exponential backoff retry policy
    pub fn default_exponential() -> Self {
        Self::Exponential {
            max_retries: 5,
            initial_delay_ms: 100,
            max_delay_ms: 10000,
            backoff_factor: 2.0,
        }
    }
    
    /// Gets the maximum number of retries
    pub fn max_retries(&self) -> usize {
        match self {
            Self::None => 0,
            Self::Fixed { max_retries, .. } => *max_retries,
            Self::Exponential { max_retries, .. } => *max_retries,
        }
    }
    
    /// Gets the delay for a specific retry attempt
    pub fn delay_for_attempt(&self, attempt: usize) -> Duration {
        match self {
            Self::None => Duration::from_millis(0),
            Self::Fixed { delay_ms, .. } => Duration::from_millis(*delay_ms),
            Self::Exponential { initial_delay_ms, max_delay_ms, backoff_factor, .. } => {
                let delay = (*initial_delay_ms as f64 * backoff_factor.powi(attempt as i32)) as u64;
                Duration::from_millis(delay.min(*max_delay_ms))
            },
        }
    }
}

/// Type of operation failures
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FailureType {
    /// Network failure (communication error)
    Network,
    /// Node failure (a compute node went down)
    Node,
    /// Memory error (out of memory)
    Memory,
    /// Timeout (operation took too long)
    Timeout,
    /// Data error (corrupted data, schema mismatch, etc.)
    Data,
    /// Unknown error
    Unknown,
}

impl FailureType {
    /// Determines if the failure is retriable
    pub fn is_retriable(&self) -> bool {
        match self {
            Self::Network | Self::Node | Self::Timeout => true,
            Self::Memory | Self::Data | Self::Unknown => false,
        }
    }
    
    /// Gets a failure type from an error
    pub fn from_error(error: &Error) -> Self {
        match error {
            Error::IoError(_) => Self::Network,
            Error::Timeout(_) => Self::Timeout,
            Error::OutOfMemory(_) => Self::Memory,
            Error::DataError(_) => Self::Data,
            _ => Self::Unknown,
        }
    }
}

/// Information about a query failure
#[derive(Debug, Clone)]
pub struct FailureInfo {
    /// Type of failure
    pub failure_type: FailureType,
    /// Time of failure
    pub failure_time: Instant,
    /// Node ID (if applicable)
    pub node_id: Option<String>,
    /// Specific error message
    pub error_message: String,
    /// Whether the failure has been recovered
    pub recovered: bool,
    /// Number of retry attempts
    pub retry_attempts: usize,
}

impl FailureInfo {
    /// Creates a new failure info
    pub fn new(failure_type: FailureType, error_message: impl Into<String>) -> Self {
        Self {
            failure_type,
            failure_time: Instant::now(),
            node_id: None,
            error_message: error_message.into(),
            recovered: false,
            retry_attempts: 0,
        }
    }
    
    /// Sets the node ID
    pub fn with_node_id(mut self, node_id: impl Into<String>) -> Self {
        self.node_id = Some(node_id.into());
        self
    }
    
    /// Marks the failure as recovered
    pub fn mark_recovered(&mut self) {
        self.recovered = true;
    }
    
    /// Increments the retry attempts
    pub fn increment_retry(&mut self) {
        self.retry_attempts += 1;
    }
}

/// Recovery strategy for failed operations
#[derive(Debug, Clone, Copy)]
pub enum RecoveryStrategy {
    /// Retry the entire query
    RetryQuery,
    /// Retry only the failed partitions
    RetryFailedPartitions,
    /// Reroute the query to different nodes
    Reroute,
    /// Fallback to a local execution engine
    LocalFallback,
}

/// Handler for fault detection and recovery
pub struct FaultToleranceHandler {
    /// Retry policy
    retry_policy: RetryPolicy,
    /// Recovery strategy
    recovery_strategy: RecoveryStrategy,
    /// Recent failures
    recent_failures: Arc<RwLock<Vec<FailureInfo>>>,
    /// Node health tracking
    node_health: Arc<RwLock<HashMap<String, bool>>>,
    /// Partition recovery tracking
    partition_recovery: Arc<RwLock<HashMap<usize, Vec<FailureInfo>>>>,
}

impl FaultToleranceHandler {
    /// Creates a new fault tolerance handler
    pub fn new(retry_policy: RetryPolicy, recovery_strategy: RecoveryStrategy) -> Self {
        Self {
            retry_policy,
            recovery_strategy,
            recent_failures: Arc::new(RwLock::new(Vec::new())),
            node_health: Arc::new(RwLock::new(HashMap::new())),
            partition_recovery: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Creates a new fault tolerance handler with default settings
    pub fn default() -> Self {
        Self::new(RetryPolicy::default_exponential(), RecoveryStrategy::RetryFailedPartitions)
    }
    
    /// Executes an operation with fault tolerance
    pub fn execute_with_retry<F, T>(&self, operation: F) -> Result<T>
    where
        F: Fn() -> Result<T>,
    {
        let mut attempt = 0;
        
        loop {
            match operation() {
                Ok(result) => return Ok(result),
                Err(error) => {
                    let failure_type = FailureType::from_error(&error);
                    
                    // Record the failure
                    let mut failure = FailureInfo::new(failure_type.clone(), error.to_string());
                    
                    if let Ok(mut failures) = self.recent_failures.write() {
                        failures.push(failure.clone());
                    }
                    
                    // Check if we should retry
                    if !failure_type.is_retriable() || attempt >= self.retry_policy.max_retries() {
                        return Err(error);
                    }
                    
                    // Increment attempt and wait before retrying
                    attempt += 1;
                    failure.increment_retry();
                    
                    // Sleep for the appropriate delay
                    std::thread::sleep(self.retry_policy.delay_for_attempt(attempt));
                }
            }
        }
    }
    
    /// Handles a failure during execution
    pub fn handle_failure(
        &self,
        plan: &ExecutionPlan,
        error: &Error,
    ) -> Result<Option<RecoveryAction>> {
        let failure_type = FailureType::from_error(error);
        
        if !failure_type.is_retriable() {
            return Err(Error::DistributedProcessing(
                format!("Non-retriable error: {}", error)
            ));
        }
        
        // Record the failure
        let mut failure = FailureInfo::new(failure_type, error.to_string());
        
        if let Ok(mut failures) = self.recent_failures.write() {
            failures.push(failure.clone());
        }
        
        // Determine recovery action based on strategy
        let action = match self.recovery_strategy {
            RecoveryStrategy::RetryQuery => {
                Some(RecoveryAction::RetryQuery {
                    plan: plan.clone(),
                    delay: self.retry_policy.delay_for_attempt(0),
                })
            },
            RecoveryStrategy::RetryFailedPartitions => {
                Some(RecoveryAction::RetryFailedPartitions {
                    plan: plan.clone(),
                    partition_ids: vec![], // Would come from actual failure details
                    delay: self.retry_policy.delay_for_attempt(0),
                })
            },
            RecoveryStrategy::Reroute => {
                Some(RecoveryAction::Reroute {
                    plan: plan.clone(),
                    excluded_nodes: vec![], // Would come from node health tracking
                })
            },
            RecoveryStrategy::LocalFallback => {
                Some(RecoveryAction::LocalFallback {
                    plan: plan.clone(),
                })
            },
        };
        
        Ok(action)
    }
    
    /// Gets recent failures
    pub fn recent_failures(&self) -> Result<Vec<FailureInfo>> {
        match self.recent_failures.read() {
            Ok(failures) => Ok(failures.clone()),
            Err(_) => Err(Error::DistributedProcessing(
                "Failed to read recent failures".to_string()
            )),
        }
    }
    
    /// Gets node health
    pub fn node_health(&self) -> Result<HashMap<String, bool>> {
        match self.node_health.read() {
            Ok(health) => Ok(health.clone()),
            Err(_) => Err(Error::DistributedProcessing(
                "Failed to read node health".to_string()
            )),
        }
    }
    
    /// Updates the health status of a node
    pub fn update_node_health(&self, node_id: impl Into<String>, healthy: bool) -> Result<()> {
        match self.node_health.write() {
            Ok(mut health) => {
                health.insert(node_id.into(), healthy);
                Ok(())
            },
            Err(_) => Err(Error::DistributedProcessing(
                "Failed to update node health".to_string()
            )),
        }
    }
    
    /// Clears all recorded failures
    pub fn clear_failures(&self) -> Result<()> {
        match self.recent_failures.write() {
            Ok(mut failures) => {
                failures.clear();
                Ok(())
            },
            Err(_) => Err(Error::DistributedProcessing(
                "Failed to clear failures".to_string()
            )),
        }
    }
}

/// Action to take for recovery
#[derive(Debug, Clone)]
pub enum RecoveryAction {
    /// Retry the entire query
    RetryQuery {
        /// Execution plan to retry
        plan: ExecutionPlan,
        /// Delay before retry
        delay: Duration,
    },
    /// Retry only the failed partitions
    RetryFailedPartitions {
        /// Execution plan to retry
        plan: ExecutionPlan,
        /// Partition IDs that failed
        partition_ids: Vec<usize>,
        /// Delay before retry
        delay: Duration,
    },
    /// Reroute the query to different nodes
    Reroute {
        /// Execution plan to reroute
        plan: ExecutionPlan,
        /// Nodes to exclude from execution
        excluded_nodes: Vec<String>,
    },
    /// Fallback to a local execution engine
    LocalFallback {
        /// Execution plan to execute locally
        plan: ExecutionPlan,
    },
}

/// Executor that provides fault tolerance
pub struct FaultTolerantExecutor<E> {
    /// Inner executor
    inner: E,
    /// Fault tolerance handler
    fault_handler: Arc<FaultToleranceHandler>,
}

impl<E> FaultTolerantExecutor<E> {
    /// Creates a new fault tolerant executor
    pub fn new(inner: E, fault_handler: FaultToleranceHandler) -> Self {
        Self {
            inner,
            fault_handler: Arc::new(fault_handler),
        }
    }
    
    /// Gets the fault tolerance handler
    pub fn fault_handler(&self) -> Arc<FaultToleranceHandler> {
        self.fault_handler.clone()
    }
}

/// Extension trait for execution contexts to add fault tolerance
pub trait FaultTolerantContext {
    /// Executes a plan with fault tolerance
    fn execute_with_fault_tolerance(
        &self,
        plan: &ExecutionPlan,
        fault_handler: &FaultToleranceHandler,
    ) -> Result<ExecutionResult>;
    
    /// Recovers from a failure
    fn recover_from_failure(
        &self,
        action: RecoveryAction,
    ) -> Result<ExecutionResult>;
}

/// Checkpoint for fault recovery
#[derive(Debug, Clone)]
pub struct ExecutionCheckpoint {
    /// Execution plan
    pub plan: ExecutionPlan,
    /// Partial result (if available)
    pub partial_result: Option<PartitionSet>,
    /// Failed partitions
    pub failed_partitions: Vec<usize>,
    /// Checkpoint time
    pub checkpoint_time: Instant,
}

impl ExecutionCheckpoint {
    /// Creates a new execution checkpoint
    pub fn new(plan: ExecutionPlan) -> Self {
        Self {
            plan,
            partial_result: None,
            failed_partitions: Vec::new(),
            checkpoint_time: Instant::now(),
        }
    }
    
    /// Sets the partial result
    pub fn with_partial_result(mut self, partial_result: PartitionSet) -> Self {
        self.partial_result = Some(partial_result);
        self
    }
    
    /// Adds a failed partition
    pub fn add_failed_partition(&mut self, partition_id: usize) {
        self.failed_partitions.push(partition_id);
    }
}

/// Manager for execution checkpoints
pub struct CheckpointManager {
    /// Checkpoints indexed by execution ID
    checkpoints: Arc<Mutex<HashMap<String, ExecutionCheckpoint>>>,
}

impl CheckpointManager {
    /// Creates a new checkpoint manager
    pub fn new() -> Self {
        Self {
            checkpoints: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    /// Creates a checkpoint for an execution
    pub fn create_checkpoint(&self, execution_id: impl Into<String>, plan: ExecutionPlan) -> Result<()> {
        let checkpoint = ExecutionCheckpoint::new(plan);
        
        match self.checkpoints.lock() {
            Ok(mut checkpoints) => {
                checkpoints.insert(execution_id.into(), checkpoint);
                Ok(())
            },
            Err(_) => Err(Error::DistributedProcessing(
                "Failed to create checkpoint".to_string()
            )),
        }
    }
    
    /// Gets a checkpoint for an execution
    pub fn get_checkpoint(&self, execution_id: &str) -> Result<Option<ExecutionCheckpoint>> {
        match self.checkpoints.lock() {
            Ok(checkpoints) => Ok(checkpoints.get(execution_id).cloned()),
            Err(_) => Err(Error::DistributedProcessing(
                "Failed to get checkpoint".to_string()
            )),
        }
    }
    
    /// Updates a checkpoint for an execution
    pub fn update_checkpoint(
        &self,
        execution_id: &str,
        partial_result: Option<PartitionSet>,
        failed_partitions: Option<Vec<usize>>,
    ) -> Result<()> {
        match self.checkpoints.lock() {
            Ok(mut checkpoints) => {
                if let Some(checkpoint) = checkpoints.get_mut(execution_id) {
                    if let Some(result) = partial_result {
                        checkpoint.partial_result = Some(result);
                    }
                    
                    if let Some(partitions) = failed_partitions {
                        checkpoint.failed_partitions = partitions;
                    }
                    
                    Ok(())
                } else {
                    Err(Error::DistributedProcessing(
                        format!("Checkpoint not found for execution ID: {}", execution_id)
                    ))
                }
            },
            Err(_) => Err(Error::DistributedProcessing(
                "Failed to update checkpoint".to_string()
            )),
        }
    }
    
    /// Removes a checkpoint for an execution
    pub fn remove_checkpoint(&self, execution_id: &str) -> Result<()> {
        match self.checkpoints.lock() {
            Ok(mut checkpoints) => {
                checkpoints.remove(execution_id);
                Ok(())
            },
            Err(_) => Err(Error::DistributedProcessing(
                "Failed to remove checkpoint".to_string()
            )),
        }
    }
}