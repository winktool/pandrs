//! Module providing data visualization functionality
//!
//! This module includes both text-based (textplots) and high-quality visualization (plotters)
//! capabilities.

use std::fs::File;
use std::io::Write;
use std::path::Path;
use textplots::{Chart, Plot, Shape};

use crate::error::{PandRSError, Result};
use crate::temporal::TimeSeries;
use crate::DataFrame;
use crate::Series;

// Export high-quality visualization module
pub mod plotters_ext;

/// Plot types
#[derive(Debug, Clone, Copy)]
pub enum PlotType {
    /// Line graph
    Line,
    /// Scatter plot
    Scatter,
    /// Point plot
    Points,
}

/// Plot output formats
#[derive(Debug, Clone, Copy)]
pub enum OutputFormat {
    /// Terminal output
    Terminal,
    /// File output (text format)
    TextFile,
}

/// Plot configuration
#[derive(Debug, Clone)]
pub struct PlotConfig {
    /// Title
    pub title: String,
    /// X-axis label
    pub x_label: String,
    /// Y-axis label
    pub y_label: String,
    /// Width (characters)
    pub width: u32,
    /// Height (lines)
    pub height: u32,
    /// Plot type
    pub plot_type: PlotType,
    /// Output format
    pub format: OutputFormat,
}

impl Default for PlotConfig {
    fn default() -> Self {
        PlotConfig {
            title: "Plot".to_string(),
            x_label: "X".to_string(),
            y_label: "Y".to_string(),
            width: 80,
            height: 25,
            plot_type: PlotType::Line,
            format: OutputFormat::Terminal,
        }
    }
}

/// Visualization extension: Visualization of Series
impl<T> Series<T>
where
    T: Clone + Copy + Into<f32> + std::fmt::Debug,
{
    /// Plot Series and save to file or display in terminal
    pub fn plot<P: AsRef<Path>>(&self, path: P, config: PlotConfig) -> Result<()> {
        let values: Vec<f32> = self.values().iter().map(|v| (*v).into()).collect();
        let indices: Vec<f32> = (0..values.len()).map(|i| i as f32).collect();

        plot_xy(&indices, &values, path, config)
    }
}

/// Visualization extension: Visualization of DataFrame
impl DataFrame {
    /// Plot two columns as XY coordinates
    pub fn plot_xy<P: AsRef<Path>>(
        &self,
        x_col: &str,
        y_col: &str,
        path: P,
        config: PlotConfig,
    ) -> Result<()> {
        // Check for column existence
        if !self.contains_column(x_col) {
            return Err(PandRSError::Column(format!(
                "Column '{}' does not exist",
                x_col
            )));
        }
        if !self.contains_column(y_col) {
            return Err(PandRSError::Column(format!(
                "Column '{}' does not exist",
                y_col
            )));
        }

        // Convert column data to numeric
        let x_values = self.get_column_numeric_values(x_col)?;
        let y_values = self.get_column_numeric_values(y_col)?;

        // Convert to f32
        let x_f32: Vec<f32> = x_values.iter().map(|&v| v as f32).collect();
        let y_f32: Vec<f32> = y_values.iter().map(|&v| v as f32).collect();

        plot_xy(&x_f32, &y_f32, path, config)
    }

    /// Draw line graphs for multiple columns
    pub fn plot_lines<P: AsRef<Path>>(
        &self,
        columns: &[&str],
        path: P,
        config: PlotConfig,
    ) -> Result<()> {
        // Check for existence of each column
        for col in columns {
            if !self.contains_column(col) {
                return Err(PandRSError::Column(format!("Column '{}' does not exist", col)));
            }
        }

        // Get indices
        let indices: Vec<f32> = (0..self.row_count()).map(|i| i as f32).collect();

        // Use only the first column (textplots has limitations for displaying multiple series)
        if let Some(&first_col) = columns.first() {
            let values = self.get_column_numeric_values(first_col)?;
            let values_f32: Vec<f32> = values.iter().map(|&v| v as f32).collect();

            let mut custom_config = config;
            custom_config.title = format!("{} ({})", custom_config.title, first_col);

            return plot_xy(&indices, &values_f32, path, custom_config);
        }

        Err(PandRSError::Empty("No columns to plot".to_string()))
    }
}

/// Visualization extension: Visualization of time series data
impl<T> TimeSeries<T>
where
    T: crate::temporal::Temporal,
{
    /// Plot time series data as a line graph
    pub fn plot<P: AsRef<Path>>(&self, path: P, config: PlotConfig) -> Result<()> {
        // Get value data
        let values: Vec<f32> = self
            .values()
            .iter()
            .map(|v| match v {
                crate::na::NA::Value(val) => (*val as i32) as f32, // Convert to numeric
                crate::na::NA::NA => 0.0, // Treat NA as 0 (more appropriate handling needed in practice)
            })
            .collect();

        // Use dates as indices
        let indices: Vec<f32> = (0..values.len()).map(|i| i as f32).collect();

        // Update plot configuration
        let mut custom_config = config;
        if custom_config.x_label == "X" {
            custom_config.x_label = "Date".to_string();
        }

        plot_xy(&indices, &values, path, custom_config)
    }
}

/// Basic plot function for XY coordinates
fn plot_xy<P: AsRef<Path>>(x: &[f32], y: &[f32], path: P, config: PlotConfig) -> Result<()> {
    if x.len() != y.len() {
        return Err(PandRSError::Consistency(
            "X and Y lengths do not match".to_string(),
        ));
    }

    // Do nothing if data is empty
    if x.is_empty() {
        return Err(PandRSError::Empty(
            "No data to plot".to_string(),
        ));
    }

    // Create points
    let points: Vec<(f32, f32)> = x.iter().zip(y.iter()).map(|(&x, &y)| (x, y)).collect();

    // Create chart
    let mut chart_string = String::new();
    chart_string.push_str(&format!("=== {} ===\n", config.title));
    chart_string.push_str(&format!(
        "X-axis: {}, Y-axis: {}\n\n",
        config.x_label, config.y_label
    ));

    // Draw plot
    let chart_result = match config.plot_type {
        PlotType::Line => Chart::new(
            config.width as u32,
            config.height as u32,
            x[0],
            x[x.len() - 1],
        )
        .lineplot(&Shape::Lines(&points))
        .to_string(),
        PlotType::Scatter | PlotType::Points => Chart::new(
            config.width as u32,
            config.height as u32,
            x[0],
            x[x.len() - 1],
        )
        .lineplot(&Shape::Points(&points))
        .to_string(),
    };

    chart_string.push_str(&chart_result);

    // Output
    match config.format {
        OutputFormat::Terminal => {
            println!("{}", chart_string);
            Ok(())
        }
        OutputFormat::TextFile => {
            let mut file = File::create(path).map_err(PandRSError::Io)?;
            file.write_all(chart_string.as_bytes())
                .map_err(PandRSError::Io)?;
            Ok(())
        }
    }
}
