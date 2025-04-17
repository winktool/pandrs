use pandrs::series::{CategoricalOrder, StringCategorical};
use pandrs::{DataFrame, Series};
use pandrs::error::Result;

fn main() -> Result<()> {
    println!("=== Example of Using Categorical Data Type ===\n");
    
    // ===========================================================
    // Creating Basic Categorical Data
    // ===========================================================
    
    println!("--- Creating Basic Categorical Data ---");
    let values = vec!["Tokyo", "Osaka", "Tokyo", "Nagoya", "Osaka", "Tokyo"];
    let values_str: Vec<String> = values.iter().map(|s| s.to_string()).collect();
    
    // Create categorical data (unique values are automatically extracted)
    let cat = StringCategorical::new(
        values_str, 
        None,  // Automatically detect categories
        Some(CategoricalOrder::Unordered),
    )?;
    
    println!("Original Data: {:?}", values);
    println!("Categories: {:?}", cat.categories());
    println!("Order Type: {:?}", cat.ordered());
    println!("Data Length: {}", cat.len());
    
    // Retrieve actual values from categorical data
    println!("\nFirst 3 values: {} {} {}", 
        cat.get(0).unwrap(),
        cat.get(1).unwrap(),
        cat.get(2).unwrap()
    );
    
    // ===========================================================
    // Creating with Explicit Category List
    // ===========================================================
    
    println!("\n--- Creating with Explicit Category List ---");
    let values2 = vec!["Red", "Blue", "Red"];
    let values2_str: Vec<String> = values2.iter().map(|s| s.to_string()).collect();
    
    // Define all categories beforehand
    let categories = vec!["Red", "Blue", "Green", "Yellow"];
    let categories_str: Vec<String> = categories.iter().map(|s| s.to_string()).collect();
    
    // Create ordered categorical data
    let cat2 = StringCategorical::new(
        values2_str,
        Some(categories_str),  // Explicit category list
        Some(CategoricalOrder::Ordered),
    )?;
    
    println!("Categories: {:?}", cat2.categories()); // Red, Blue, Green, Yellow
    println!("Data: {:?}", 
        (0..cat2.len()).map(|i| cat2.get(i).unwrap()).collect::<Vec<_>>()
    ); // Red, Blue, Red
    
    // ===========================================================
    // Operations on Categorical Data
    // ===========================================================
    
    println!("\n--- Example of Categorical Operations ---");
    
    // Base categorical data
    let fruits = vec!["Apple", "Banana", "Apple", "Orange"];
    let fruits_str: Vec<String> = fruits.iter().map(|s| s.to_string()).collect();
    let mut fruit_cat = StringCategorical::new(fruits_str, None, None)?;
    
    println!("Original Categories: {:?}", fruit_cat.categories());
    
    // Add categories
    let new_cats = vec!["Grape", "Strawberry"];
    let new_cats_str: Vec<String> = new_cats.iter().map(|s| s.to_string()).collect();
    fruit_cat.add_categories(new_cats_str)?;
    
    println!("Categories after addition: {:?}", fruit_cat.categories());
    
    // Change category order
    let reordered = vec!["Banana", "Strawberry", "Orange", "Apple", "Grape"];
    let reordered_str: Vec<String> = reordered.iter().map(|s| s.to_string()).collect();
    fruit_cat.reorder_categories(reordered_str)?;
    
    println!("Categories after reordering: {:?}", fruit_cat.categories());
    println!("Data: {:?}", 
        (0..fruit_cat.len()).map(|i| fruit_cat.get(i).unwrap()).collect::<Vec<_>>()
    );
    
    // ===========================================================
    // Integration with DataFrame
    // ===========================================================
    
    println!("\n--- Integration of Categorical Data with DataFrame ---");
    
    // Create a basic DataFrame
    let mut df = DataFrame::new();
    
    // Add regular columns
    let regions = vec!["Hokkaido", "Kanto", "Kansai", "Kyushu", "Kanto", "Kansai"];
    let regions_str: Vec<String> = regions.iter().map(|s| s.to_string()).collect();
    let pop = vec!["Low", "High", "High", "Medium", "High", "High"];
    let pop_str: Vec<String> = pop.iter().map(|s| s.to_string()).collect();
    
    df.add_column("Region".to_string(), Series::new(regions_str, Some("Region".to_string()))?)?;
    df.add_column("Population".to_string(), Series::new(pop_str, Some("Population".to_string()))?)?;
    
    println!("Original DataFrame:\n{:?}", df);
    
    // ===========================================================
    // Creating Simplified Categorical DataFrame
    // ===========================================================
    
    // Create a DataFrame directly from categorical data
    println!("\n--- Creating DataFrame with Categorical Data ---");
    
    // Create categorical data
    let populations = vec!["Low", "Medium", "High"];
    let populations_str: Vec<String> = populations.iter().map(|s| s.to_string()).collect();
    let pop_cat = StringCategorical::new(
        populations_str,
        None,  // Automatically detect
        Some(CategoricalOrder::Ordered)
    )?;
    
    // Region data
    let regions = vec!["Hokkaido", "Kanto", "Kansai"];
    let regions_str: Vec<String> = regions.iter().map(|s| s.to_string()).collect();
    
    // Create DataFrame from both categorical data
    let categoricals = vec![
        ("Population".to_string(), pop_cat),
    ];
    
    let mut df_cat = DataFrame::from_categoricals(categoricals)?;
    
    // Add region column
    df_cat.add_column("Region".to_string(), Series::new(regions_str, Some("Region".to_string()))?)?;
    
    println!("\nDataFrame after adding categorical data:\n{:?}", df_cat);
    
    // Check if columns are categorical
    println!("\nIs 'Population' column categorical: {}", df_cat.is_categorical("Population"));
    println!("Is 'Region' column categorical: {}", df_cat.is_categorical("Region"));
    
    // ===========================================================
    // Example of Multi-Categorical DataFrame
    // ===========================================================
    
    println!("\n--- Example of Multi-Categorical DataFrame ---");
    
    // Create product and color data as separate categories
    let products = vec!["A", "B", "C"];
    let products_str: Vec<String> = products.iter().map(|s| s.to_string()).collect();
    let product_cat = StringCategorical::new(products_str, None, None)?;
    
    let colors = vec!["Red", "Blue", "Green"];
    let colors_str: Vec<String> = colors.iter().map(|s| s.to_string()).collect();
    let color_cat = StringCategorical::new(colors_str, None, None)?;
    
    // Create a DataFrame containing both categories
    let multi_categoricals = vec![
        ("Product".to_string(), product_cat),
        ("Color".to_string(), color_cat),
    ];
    
    let multi_df = DataFrame::from_categoricals(multi_categoricals)?;
    
    println!("Multi-Categorical DataFrame:\n{:?}", multi_df);
    println!("\nIs 'Product' column categorical: {}", multi_df.is_categorical("Product"));
    println!("Is 'Color' column categorical: {}", multi_df.is_categorical("Color"));
    
    // ===========================================================
    // Aggregation and Analysis of Categorical Data
    // ===========================================================
    
    println!("\n--- Aggregation and Grouping of Categorical Data ---");
    
    // Start with a simple DataFrame
    let mut df_simple = DataFrame::new();
    
    // Add product data
    let products = vec!["A", "B", "C", "A", "B"];
    let products_str: Vec<String> = products.iter().map(|s| s.to_string()).collect();
    let sales = vec!["100", "150", "200", "120", "180"];
    let sales_str: Vec<String> = sales.iter().map(|s| s.to_string()).collect();
    
    df_simple.add_column("Product".to_string(), Series::new(products_str.clone(), Some("Product".to_string()))?)?;
    df_simple.add_column("Sales".to_string(), Series::new(sales_str, Some("Sales".to_string()))?)?;
    
    println!("Original DataFrame:\n{:?}", df_simple);
    
    // Aggregate by product
    let product_counts = df_simple.value_counts("Product")?;
    println!("\nProduct Counts:\n{:?}", product_counts);
    
    // Transformation and interaction between categorical and series
    println!("\n--- Interaction between Categorical and Series ---");
    
    // Create a simple categorical series
    let letter_cat = StringCategorical::new(
        vec!["A".to_string(), "B".to_string(), "C".to_string()],
        None,
        None
    )?;
    
    // Convert to series
    let letter_series = letter_cat.to_series(Some("Letter".to_string()))?;
    println!("Converted from categorical to series: {:?}", letter_series);
    
    // Additional information about categorical data
    println!("\n--- Characteristics of Categorical Data ---");
    println!("Categorical data is stored in memory only once, regardless of repeated string values.");
    println!("This makes it particularly efficient for datasets with many duplicate string values.");
    println!("Additionally, ordered categorical data allows meaningful sorting of data.");
    
    println!("\n=== Sample Complete ===");
    Ok(())
}