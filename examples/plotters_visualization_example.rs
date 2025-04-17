//! Example of visualization features using Plotters
//!
//! This sample demonstrates the basic usage of high-quality visualization features with Plotters.

use pandrs::{DataFrame, Series};
use pandrs::vis::plotters_ext::{PlotSettings, PlotKind, OutputType};
use pandrs::vis::PlotConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Example of plotting a single Series
    println!("Sample 1: Creating a plot for a single Series");
    let values = vec![15.0, 23.5, 18.2, 29.8, 32.1, 28.5, 19.2, 22.3, 25.6, 21.9];
    
    // Line chart creation
    let series = Series::new(values.clone(), Some("Temperature Change".to_string()))?;
    let line_settings = PlotSettings {
        title: "Temperature Change Over Time".to_string(),
        x_label: "Time".to_string(),
        y_label: "Temperature (°C)".to_string(),
        plot_kind: PlotKind::Line,
        ..PlotSettings::default()
    };
    series.plotters_plot("examples/temp_line.png", line_settings)?;
    println!("  ✓ Line chart generated: examples/temp_line.png");
    
    // Bar chart creation
    let bar_settings = PlotSettings {
        title: "Temperature Change Over Time".to_string(),
        x_label: "Time".to_string(),
        y_label: "Temperature (°C)".to_string(),
        plot_kind: PlotKind::Bar,
        ..PlotSettings::default()
    };
    series.plotters_plot("examples/temp_bar.png", bar_settings)?;
    println!("  ✓ Bar chart generated: examples/temp_bar.png");
    
    // Scatter plot in SVG format
    let scatter_settings = PlotSettings {
        title: "Temperature Change Over Time".to_string(),
        x_label: "Time".to_string(),
        y_label: "Temperature (°C)".to_string(),
        plot_kind: PlotKind::Scatter,
        output_type: OutputType::SVG,
        ..PlotSettings::default()
    };
    series.plotters_plot("examples/temp_scatter.svg", scatter_settings)?;
    println!("  ✓ Scatter plot (SVG) generated: examples/temp_scatter.svg");
    
    // Histogram creation
    let hist_settings = PlotSettings {
        title: "Temperature Distribution Histogram".to_string(),
        x_label: "Temperature Range (°C)".to_string(),
        y_label: "Frequency".to_string(),
        ..PlotSettings::default()
    };
    series.plotters_histogram("examples/temp_histogram.png", 5, hist_settings)?;
    println!("  ✓ Histogram generated: examples/temp_histogram.png");
    
    // 2. Visualization using DataFrame
    println!("\nSample 2: Visualization using DataFrame");
    let mut df = DataFrame::new();
    
    // Prepare data
    let days = Series::new(vec![1, 2, 3, 4, 5, 6, 7], Some("Day".to_string()))?;
    let temp = Series::new(vec![22.5, 25.1, 23.8, 27.2, 26.5, 24.9, 29.1], Some("Temperature".to_string()))?;
    let humidity = Series::new(vec![67.0, 72.3, 69.5, 58.2, 62.1, 71.5, 55.8], Some("Humidity".to_string()))?;
    let pressure = Series::new(vec![1013.2, 1010.5, 1009.8, 1014.5, 1018.2, 1015.7, 1011.3], Some("Pressure".to_string()))?;
    
    // Add columns to DataFrame
    df.add_column("Day".to_string(), days)?;
    df.add_column("Temperature".to_string(), temp)?;
    df.add_column("Humidity".to_string(), humidity)?;
    df.add_column("Pressure".to_string(), pressure)?;
    
    // Create XY scatter plot (relationship between temperature and humidity)
    let xy_config = PlotConfig {
        title: "Relationship Between Temperature and Humidity".to_string(),
        x_label: "Temperature".to_string(),
        y_label: "Humidity".to_string(),
        ..PlotConfig::default()
    };
    df.plot_xy("Temperature", "Humidity", "examples/temp_humidity.png", xy_config)?;
    println!("  ✓ Scatter plot (Temperature vs Humidity) generated: examples/temp_humidity.png");
    
    // Comparison of multiple series
    let multi_settings = PlotSettings {
        title: "Weather Data Trends".to_string(),
        x_label: "Days".to_string(),
        plot_kind: PlotKind::Line,
        ..PlotSettings::default()
    };
    df.plotters_plot_columns(&["Temperature", "Humidity", "Pressure"], "examples/weather_multi.png", multi_settings)?;
    println!("  ✓ Multi-series plot generated: examples/weather_multi.png");
    
    println!("\nAll samples executed successfully. Please check the generated files.");
    Ok(())
}