//! GPU acceleration module for PandRS
//!
//! This module provides GPU-accelerated operations using CUDA/cuBLAS for large-scale
//! data processing. It allows PandRS to leverage the massive parallelism offered by 
//! GPU hardware for significant performance improvements on compatible operations.

use std::sync::{Arc, Mutex};
use std::fmt;
use std::error::Error as StdError;

use crate::error::{Result, Error, PandRSError};
use crate::DataFrame;
use crate::Series;
use crate::optimized::dataframe::OptimizedDataFrame;

/// Configuration options for GPU operations
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Whether to enable GPU acceleration when available
    pub enabled: bool,
    /// Maximum memory usage for GPU operations (in bytes)
    pub memory_limit: usize,
    /// Device ID to use (for multi-GPU systems)
    pub device_id: i32,
    /// Whether to fallback to CPU if GPU operation fails
    pub fallback_to_cpu: bool,
    /// Whether to use pinned memory for faster transfers
    pub use_pinned_memory: bool,
    /// Minimum size threshold for offloading to GPU (in elements)
    pub min_size_threshold: usize,
}

impl Default for GpuConfig {
    fn default() -> Self {
        GpuConfig {
            enabled: true,
            memory_limit: 1024 * 1024 * 1024, // 1GB default
            device_id: 0,
            fallback_to_cpu: true,
            use_pinned_memory: true,
            min_size_threshold: 10_000, // Only use GPU for operations on at least 10K elements
        }
    }
}

/// GPU operation context
pub struct GpuContext {
    /// GPU device status
    device_status: Arc<Mutex<GpuDeviceStatus>>,
    /// Configuration
    config: GpuConfig,
}

/// GPU device status
#[derive(Debug, Clone)]
pub struct GpuDeviceStatus {
    /// Whether a CUDA-compatible GPU is available
    pub available: bool,
    /// CUDA version
    pub cuda_version: Option<String>,
    /// Device name
    pub device_name: Option<String>,
    /// Total device memory (in bytes)
    pub total_memory: Option<usize>,
    /// Free device memory (in bytes)
    pub free_memory: Option<usize>,
    /// Number of CUDA cores
    pub core_count: Option<usize>,
}

impl Default for GpuDeviceStatus {
    fn default() -> Self {
        GpuDeviceStatus {
            available: false,
            cuda_version: None,
            device_name: None,
            total_memory: None,
            free_memory: None,
            core_count: None,
        }
    }
}

impl GpuContext {
    /// Create a new GPU context with the given configuration
    pub fn new(config: GpuConfig) -> Self {
        let device_status = Arc::new(Mutex::new(Self::detect_device(&config)));
        
        GpuContext {
            device_status,
            config,
        }
    }
    
    /// Detect available GPU devices and their capabilities
    fn detect_device(config: &GpuConfig) -> GpuDeviceStatus {
        #[cfg(feature = "cuda")]
        {
            // Actual implementation when CUDA is enabled
            use cuda_builder::CudaBuilder;
            
            // Check if CUDA is available
            match CudaBuilder::new() {
                Ok(builder) => {
                    let devices = builder.get_device_count();
                    
                    if devices > 0 && config.device_id < devices as i32 {
                        let device = builder.get_device(config.device_id as usize).unwrap();
                        let props = device.get_properties().unwrap();
                        
                        let memory_info = device.get_memory_info().unwrap();
                        
                        return GpuDeviceStatus {
                            available: true,
                            cuda_version: Some(format!("{}.{}", 
                                props.major, props.minor)),
                            device_name: Some(props.name.clone()),
                            total_memory: Some(memory_info.total as usize),
                            free_memory: Some(memory_info.free as usize),
                            core_count: Some(props.multiprocessor_count as usize 
                                * props.max_threads_per_multiprocessor as usize),
                        };
                    }
                }
                Err(_) => {
                    // CUDA not available, return default status
                }
            }
        }
        
        // Default to unavailable when CUDA is not enabled
        GpuDeviceStatus::default()
    }
    
    /// Check if GPU acceleration is available and enabled
    pub fn is_available(&self) -> bool {
        if !self.config.enabled {
            return false;
        }
        
        self.device_status.lock().unwrap().available
    }
    
    /// Get device status
    pub fn get_device_status(&self) -> GpuDeviceStatus {
        self.device_status.lock().unwrap().clone()
    }
    
    /// Update device status
    pub fn update_device_status(&self) {
        let new_status = Self::detect_device(&self.config);
        *self.device_status.lock().unwrap() = new_status;
    }
    
    /// Check if operation should be offloaded to GPU based on size threshold
    pub fn should_use_gpu(&self, size: usize) -> bool {
        self.is_available() && size >= self.config.min_size_threshold
    }
}

/// Types of GPU operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuOperationType {
    /// Matrix multiplication
    MatrixMultiply,
    /// Element-wise addition
    Add,
    /// Element-wise subtraction
    Subtract,
    /// Element-wise multiplication
    Multiply,
    /// Element-wise division
    Divide,
    /// Aggregation (sum, mean, etc.)
    Aggregate,
    /// Sorting
    Sort,
    /// Filtering
    Filter,
}

/// Custom error type for GPU operations
#[derive(Debug)]
pub enum GpuError {
    /// CUDA driver not available
    DriverNotAvailable,
    /// Insufficient GPU memory
    InsufficientMemory,
    /// Incompatible data type
    IncompatibleDataType,
    /// Device error
    DeviceError(String),
    /// Kernel execution error
    KernelExecutionError(String),
    /// Transfer error
    TransferError(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GpuError::DriverNotAvailable => write!(f, "CUDA driver not available"),
            GpuError::InsufficientMemory => write!(f, "Insufficient GPU memory"),
            GpuError::IncompatibleDataType => write!(f, "Incompatible data type for GPU operation"),
            GpuError::DeviceError(msg) => write!(f, "GPU device error: {}", msg),
            GpuError::KernelExecutionError(msg) => write!(f, "GPU kernel execution error: {}", msg),
            GpuError::TransferError(msg) => write!(f, "GPU data transfer error: {}", msg),
        }
    }
}

impl StdError for GpuError {}

impl From<GpuError> for Error {
    fn from(error: GpuError) -> Self {
        Error::Other(Box::new(error))
    }
}

/// Struct for managing GPU operations across the application
#[derive(Clone)]
pub struct GpuManager {
    /// The global GPU context
    context: Arc<GpuContext>,
}

impl GpuManager {
    /// Create a new GPU manager with default configuration
    pub fn new() -> Self {
        Self::with_config(GpuConfig::default())
    }
    
    /// Create a new GPU manager with the given configuration
    pub fn with_config(config: GpuConfig) -> Self {
        GpuManager {
            context: Arc::new(GpuContext::new(config)),
        }
    }
    
    /// Get GPU context
    pub fn context(&self) -> &GpuContext {
        &self.context
    }
    
    /// Check if GPU acceleration is available
    pub fn is_available(&self) -> bool {
        self.context.is_available()
    }
    
    /// Get detailed GPU device information
    pub fn device_info(&self) -> GpuDeviceStatus {
        self.context.get_device_status()
    }
}

// Global GPU manager instance initialized lazily
lazy_static::lazy_static! {
    static ref GPU_MANAGER: Mutex<Option<GpuManager>> = Mutex::new(None);
}

/// Initialize global GPU manager with default configuration
pub fn init_gpu() -> Result<GpuDeviceStatus> {
    init_gpu_with_config(GpuConfig::default())
}

/// Initialize global GPU manager with custom configuration
pub fn init_gpu_with_config(config: GpuConfig) -> Result<GpuDeviceStatus> {
    let manager = GpuManager::with_config(config);
    let status = manager.device_info();
    
    let mut global = GPU_MANAGER.lock().unwrap();
    *global = Some(manager);
    
    Ok(status)
}

/// Get the global GPU manager instance
pub fn get_gpu_manager() -> Result<GpuManager> {
    let global = GPU_MANAGER.lock().unwrap();
    
    match &*global {
        Some(manager) => Ok(manager.clone()),
        None => {
            // Initialize with default config if not already initialized
            drop(global); // Release lock before recursive call
            init_gpu()?;
            get_gpu_manager()
        }
    }
}

// Include appropriate implementation modules based on features
#[cfg(feature = "cuda")]
pub mod cuda;

// Include this module regardless of CUDA availability to provide fallback
pub mod operations;

// Benchmarking module
pub mod benchmark;