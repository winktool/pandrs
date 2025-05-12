// Core data structures and traits for PandRS
pub mod error;
pub mod column;
pub mod data_value;
pub mod index;
pub mod multi_index;

// Re-exports for convenience
pub use error::{Error, PandRSError, Result};
pub use column::{Column, ColumnType, ColumnTrait, ColumnCast, BitMask};
pub use data_value::DataValue;
pub use index::{Index, IndexTrait};
pub use multi_index::MultiIndex;