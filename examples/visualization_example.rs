use pandrs::{DataFrame, OutputFormat, PlotConfig, PlotType, Series};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Visualization Example ===\n");

    // Create sample data
    let x = Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("x".to_string()))?;
    let y = Series::new(vec![2.0, 3.5, 4.2, 4.8, 7.0], Some("y".to_string()))?;
    let z = Series::new(vec![1.5, 2.2, 3.1, 5.3, 8.5], Some("z".to_string()))?;

    // Clone before adding to DataFrame
    let y_for_plot = y.clone();

    // Plot Series - Terminal Output
    let config = PlotConfig {
        title: "Sample Series Plot".to_string(),
        x_label: "Index".to_string(),
        y_label: "Value".to_string(),
        width: 80,
        height: 25,
        plot_type: PlotType::Line,
        format: OutputFormat::Terminal,
    };

    println!("Plotting y series:");
    y_for_plot.plot("", config.clone())?;

    // Plot Scatter
    let scatter_config = PlotConfig {
        title: "X vs Y Scatter Plot".to_string(),
        x_label: "X Value".to_string(),
        y_label: "Y Value".to_string(),
        plot_type: PlotType::Scatter,
        ..config.clone()
    };

    // Create DataFrame
    let mut df = DataFrame::new();
    df.add_column("x".to_string(), x)?;
    df.add_column("y".to_string(), y)?;
    df.add_column("z".to_string(), z)?;

    println!("\nPlotting XY Scatter Plot:");
    df.plot_xy("x", "y", "", scatter_config)?;

    // Single Series Line Plot (Text plot has limitations for multiple series)
    let line_config = PlotConfig {
        title: "Single Series Plot".to_string(),
        x_label: "X".to_string(),
        y_label: "Value".to_string(),
        plot_type: PlotType::Line,
        ..config.clone()
    };

    println!("\nPlotting Line Graph:");
    df.plot_lines(&["z"], "", line_config)?;

    // Save as Text File
    let file_config = PlotConfig {
        title: "File Output Plot".to_string(),
        x_label: "Index".to_string(),
        y_label: "Value".to_string(),
        plot_type: PlotType::Line,
        format: OutputFormat::TextFile,
        ..config.clone()
    };

    println!("\nSaving plot to file...");
    let y_from_df = df.get_column_numeric_values("y")?;
    // Convert f64 to f32
    let y_f32: Vec<f32> = y_from_df.iter().map(|&val| val as f32).collect();
    let y_series = Series::new(y_f32, Some("y".to_string()))?;
    y_series.plot("examples/plot.txt", file_config)?;
    println!("Plot saved: examples/plot.txt");

    println!("\n=== Visualization Example Complete ===");
    Ok(())
}
