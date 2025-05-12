// Series implementations module
pub mod base;
pub mod na;
pub mod categorical;
pub mod functions;

#[cfg(feature = "cuda")]
pub mod gpu;

// Re-exports for convenience
pub use base::Series;
pub use na::NASeries;
pub use categorical::{Categorical, CategoricalOrder, StringCategorical};

// Optional feature re-exports
#[cfg(feature = "cuda")]
pub use gpu::SeriesGpuExt;

// Legacy exports for backward compatibility
pub use categorical::{Categorical as LegacyCategorical, CategoricalOrder as LegacyCategoricalOrder,
                     StringCategorical as LegacyStringCategorical};
// For backward compatibility (old NASeries)
#[allow(deprecated)]
pub use na::NASeries as LegacyNASeries;