// Computation functionality module
pub mod parallel;
pub mod lazy;

#[cfg(feature = "cuda")]
pub mod gpu;

// Re-exports
pub use parallel::ParallelUtils;
pub use lazy::LazyFrame;

#[cfg(feature = "cuda")]
pub use gpu::{GpuConfig, init_gpu, GpuDeviceStatus, GpuBenchmark};