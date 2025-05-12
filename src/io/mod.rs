pub mod csv;
pub mod json;
#[cfg(feature = "parquet")]
pub mod parquet;
#[cfg(feature = "excel")]
pub mod excel;
#[cfg(feature = "sql")]
pub mod sql;

// Re-export commonly used functions
pub use csv::{read_csv, write_csv};
pub use json::{read_json, write_json};
#[cfg(feature = "parquet")]
pub use parquet::{read_parquet, write_parquet, ParquetCompression};
#[cfg(feature = "excel")]
pub use excel::{read_excel, write_excel};
#[cfg(feature = "sql")]
pub use sql::{read_sql, execute_sql, write_to_sql};