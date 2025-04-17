use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::hash::Hash;

use crate::error::{PandRSError, Result};
use crate::index::{Index, StringIndex};
use crate::na::NA;
use crate::series::{Series, NASeries};

/// Categorical order type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CategoricalOrder {
    /// Without order
    Unordered,
    /// With defined order for categories
    Ordered,
}

/// Structure representing categorical data
///
/// Provides a memory-efficient representation of categorical data.
/// Has similar functionality to pandas' `Categorical` in Python.
#[derive(Debug, Clone)]
pub struct Categorical<T>
where
    T: Debug + Clone + Eq + Hash + Display + Ord,
{
    /// Codes (integer indices) to actual category values
    codes: Vec<i32>,
    
    /// List of unique category values
    categories: Vec<T>,
    
    /// Map from category to integer code
    category_map: HashMap<T, i32>,
    
    /// Category ordering
    ordered: CategoricalOrder,
}

impl<T> Categorical<T>
where
    T: Debug + Clone + Eq + Hash + Display + Ord,
{
    /// Create a new categorical data structure
    ///
    /// # Arguments
    /// * `values` - Original data values
    /// * `categories` - List of categories to use (if None, automatically generated from unique values)
    /// * `ordered` - Whether to order the categories
    pub fn new(
        values: Vec<T>,
        categories: Option<Vec<T>>,
        ordered: Option<CategoricalOrder>,
    ) -> Result<Self> {
        // Setup categories
        let (categories, category_map) = match categories {
            Some(cats) => {
                // Create a map from specified categories
                let mut map = HashMap::with_capacity(cats.len());
                for (i, cat) in cats.iter().enumerate() {
                    if map.insert(cat.clone(), i as i32).is_some() {
                        return Err(PandRSError::Consistency(format!(
                            "Category '{:?}' is duplicated",
                            cat
                        )));
                    }
                }
                (cats, map)
            }
            None => {
                // Collect unique categories from data
                let mut unique_cats = Vec::new();
                let mut map = HashMap::new();
                
                for value in &values {
                    if !map.contains_key(value) {
                        map.insert(value.clone(), unique_cats.len() as i32);
                        unique_cats.push(value.clone());
                    }
                }
                
                (unique_cats, map)
            }
        };
        
        // Convert values to codes
        let mut codes = Vec::with_capacity(values.len());
        for value in values {
            match category_map.get(&value) {
                Some(&code) => codes.push(code),
                None => {
                    return Err(PandRSError::Consistency(format!(
                        "Value '{:?}' is not in categories",
                        value
                    )));
                }
            }
        }
        
        Ok(Categorical {
            codes,
            categories,
            category_map,
            ordered: ordered.unwrap_or(CategoricalOrder::Unordered),
        })
    }
    
    /// Create Categorical directly from codes (internal use)
    fn from_codes(
        codes: Vec<i32>,
        categories: Vec<T>,
        ordered: CategoricalOrder,
    ) -> Result<Self> {
        // Build category map
        let mut category_map = HashMap::with_capacity(categories.len());
        for (i, cat) in categories.iter().enumerate() {
            category_map.insert(cat.clone(), i as i32);
        }
        
        // Validate codes
        let max_code = categories.len() as i32 - 1;
        for &code in &codes {
            if code != -1 && (code < 0 || code > max_code) {
                return Err(PandRSError::Consistency(format!(
                    "Code {} is out of valid range",
                    code
                )));
            }
        }
        
        Ok(Categorical {
            codes,
            categories,
            category_map,
            ordered,
        })
    }
    
    /// Get the length of the data
    pub fn len(&self) -> usize {
        self.codes.len()
    }
    
    /// Check if data is empty
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }
    
    /// Get the list of categories
    pub fn categories(&self) -> &[T] {
        &self.categories
    }
    
    /// Get the codes
    pub fn codes(&self) -> &[i32] {
        &self.codes
    }
    
    /// Get the order information
    pub fn ordered(&self) -> &CategoricalOrder {
        &self.ordered
    }
    
    /// Get the value at a specific index
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.codes.len() {
            return None;
        }
        
        let code = self.codes[index];
        if code == -1 {
            // -1 represents a missing value
            None
        } else {
            Some(&self.categories[code as usize])
        }
    }
    
    /// Set the ordering
    pub fn set_ordered(&mut self, ordered: CategoricalOrder) {
        self.ordered = ordered;
    }
    
    /// Change the category order
    pub fn reorder_categories(&mut self, new_categories: Vec<T>) -> Result<()> {
        if new_categories.len() != self.categories.len() {
            return Err(PandRSError::Consistency(format!(
                "Number of new categories {} does not match current number of categories {}",
                new_categories.len(),
                self.categories.len()
            )));
        }
        
        // Verify that all existing values are included in new categories
        let mut new_cat_set = HashMap::with_capacity(new_categories.len());
        for cat in &new_categories {
            new_cat_set.insert(cat, ());
        }
        
        for cat in &self.categories {
            if !new_cat_set.contains_key(cat) {
                return Err(PandRSError::Consistency(format!(
                    "Category '{:?}' is not in the new category list",
                    cat
                )));
            }
        }
        
        // Build new categories and map
        let mut new_map = HashMap::with_capacity(new_categories.len());
        let mut new_codes = vec![-1; self.codes.len()];
        
        for (i, cat) in new_categories.iter().enumerate() {
            new_map.insert(cat.clone(), i as i32);
        }
        
        // Remap codes
        for (i, &old_code) in self.codes.iter().enumerate() {
            if old_code != -1 {
                let old_cat = &self.categories[old_code as usize];
                if let Some(&new_code) = new_map.get(old_cat) {
                    new_codes[i] = new_code;
                }
            }
        }
        
        self.categories = new_categories;
        self.category_map = new_map;
        self.codes = new_codes;
        
        Ok(())
    }
    
    /// Add categories
    pub fn add_categories(&mut self, new_categories: Vec<T>) -> Result<()> {
        let mut categories = self.categories.clone();
        let mut category_map = self.category_map.clone();
        
        // Add new categories
        for cat in new_categories {
            if category_map.contains_key(&cat) {
                return Err(PandRSError::Consistency(format!(
                    "Category '{:?}' already exists",
                    cat
                )));
            }
            
            let new_code = categories.len() as i32;
            category_map.insert(cat.clone(), new_code);
            categories.push(cat);
        }
        
        self.categories = categories;
        self.category_map = category_map;
        
        Ok(())
    }
    
    /// Remove categories
    pub fn remove_categories(&mut self, categories_to_remove: &[T]) -> Result<()> {
        // Set of codes to remove
        let mut codes_to_remove = Vec::new();
        
        for cat in categories_to_remove {
            if let Some(&code) = self.category_map.get(cat) {
                codes_to_remove.push(code);
            } else {
                return Err(PandRSError::Consistency(format!(
                    "Category '{:?}' does not exist",
                    cat
                )));
            }
        }
        
        // Change data using removed categories to missing values
        for code in &mut self.codes {
            if codes_to_remove.contains(code) {
                *code = -1;
            }
        }
        
        // Build new category list and map
        let mut new_categories = Vec::new();
        let mut new_map = HashMap::new();
        
        for (i, cat) in self.categories.iter().enumerate() {
            if !categories_to_remove.contains(cat) {
                let new_code = new_categories.len() as i32;
                new_map.insert(cat.clone(), new_code);
                new_categories.push(cat.clone());
            }
        }
        
        // Remap codes
        for code in &mut self.codes {
            if *code != -1 {
                let old_cat = &self.categories[*code as usize];
                *code = *new_map.get(old_cat).unwrap();
            }
        }
        
        self.categories = new_categories;
        self.category_map = new_map;
        
        Ok(())
    }
    
    /// Convert to category values
    pub fn as_values(&self) -> Vec<Option<T>> {
        self.codes
            .iter()
            .map(|&code| {
                if code == -1 {
                    None
                } else {
                    Some(self.categories[code as usize].clone())
                }
            })
            .collect()
    }
    
    /// Count occurrences of unique categories
    pub fn value_counts(&self) -> Result<Series<usize>> {
        let mut counts = vec![0; self.categories.len()];
        
        for &code in &self.codes {
            if code != -1 {
                counts[code as usize] += 1;
            }
        }
        
        // Build Series from count results
        let mut values = Vec::with_capacity(self.categories.len());
        let mut count_values = Vec::with_capacity(self.categories.len());
        
        for (i, &count) in counts.iter().enumerate() {
            if count > 0 {
                values.push(self.categories[i].clone());
                count_values.push(count);
            }
        }
        
        // Build Series from index and count values
        let index = Index::new(values)?;
        let result = Series::with_index(count_values, index, Some("count".to_string()))?;
        
        Ok(result)
    }
    
    /// Convert to Series
    pub fn to_series(&self, name: Option<String>) -> Result<Series<T>> {
        // Convert codes to values
        let values: Vec<T> = self
            .codes
            .iter()
            .filter_map(|&code| {
                if code == -1 {
                    None
                } else {
                    Some(self.categories[code as usize].clone())
                }
            })
            .collect();
        
        Series::new(values, name)
    }
}

/// Alias for string categorical
pub type StringCategorical = Categorical<String>;

impl<T> Categorical<T>
where
    T: Debug + Clone + Eq + Hash + Display + Ord,
{
    /// Create categorical data from a vector with NA values
    ///
    /// # Arguments
    /// * `values` - Data values that may contain NA
    /// * `categories` - List of categories to use (if None, automatically generated from unique values)
    /// * `ordered` - Whether to order the categories
    pub fn from_na_vec(
        values: Vec<NA<T>>,
        categories: Option<Vec<T>>,
        ordered: Option<CategoricalOrder>,
    ) -> Result<Self> {
        // Setup categories
        let (categories, category_map) = match categories {
            Some(cats) => {
                // Create a map from specified categories
                let mut map = HashMap::with_capacity(cats.len());
                for (i, cat) in cats.iter().enumerate() {
                    if map.insert(cat.clone(), i as i32).is_some() {
                        return Err(PandRSError::Consistency(format!(
                            "Category '{:?}' is duplicated",
                            cat
                        )));
                    }
                }
                (cats, map)
            }
            None => {
                // Collect unique categories from data
                let mut unique_cats = Vec::new();
                let mut map = HashMap::new();
                
                for value in &values {
                    if let NA::Value(val) = value {
                        if !map.contains_key(val) {
                            map.insert(val.clone(), unique_cats.len() as i32);
                            unique_cats.push(val.clone());
                        }
                    }
                }
                
                (unique_cats, map)
            }
        };
        
        // Convert values to codes
        let mut codes = Vec::with_capacity(values.len());
        for value in values {
            match value {
                NA::Value(val) => {
                    match category_map.get(&val) {
                        Some(&code) => codes.push(code),
                        None => {
                            return Err(PandRSError::Consistency(format!(
                                "Value '{:?}' is not in categories",
                                val
                            )));
                        }
                    }
                }
                NA::NA => codes.push(-1), // NA values are represented as -1 code
            }
        }
        
        Ok(Categorical {
            codes,
            categories,
            category_map,
            ordered: ordered.unwrap_or(CategoricalOrder::Unordered),
        })
    }
    
    /// Convert categorical data to a vector of `NA<T>`
    pub fn to_na_vec(&self) -> Vec<NA<T>> {
        self.codes
            .iter()
            .map(|&code| {
                if code == -1 {
                    NA::NA
                } else {
                    NA::Value(self.categories[code as usize].clone())
                }
            })
            .collect()
    }
    
    /// Convert categorical data to NASeries
    pub fn to_na_series(&self, name: Option<String>) -> Result<NASeries<T>> {
        let na_values = self.to_na_vec();
        NASeries::new(na_values, name)
    }
    
    /// Create the union (all unique categories) of two categorical data structures
    pub fn union(&self, other: &Self) -> Result<Self> {
        // Copy own categories
        let mut categories = self.categories.clone();
        let mut category_map = self.category_map.clone();
        
        // Add categories from the other that are not in self
        for cat in &other.categories {
            if !category_map.contains_key(cat) {
                let new_code = categories.len() as i32;
                category_map.insert(cat.clone(), new_code);
                categories.push(cat.clone());
            }
        }
        
        // Create new code list (keep original data)
        let codes = self.codes.clone();
        
        // Return new categorical
        Ok(Categorical {
            codes,
            categories,
            category_map,
            ordered: self.ordered.clone(),
        })
    }
    
    /// Create the intersection (only common categories) of two categorical data structures
    pub fn intersection(&self, other: &Self) -> Result<Self> {
        // Collect common categories
        let mut common_categories = Vec::new();
        let mut category_map = HashMap::new();
        
        for cat in &self.categories {
            if other.category_map.contains_key(cat) {
                let new_code = common_categories.len() as i32;
                category_map.insert(cat.clone(), new_code);
                common_categories.push(cat.clone());
            }
        }
        
        // Create new code list (keep original data but set to NA if not in common categories)
        let mut codes = Vec::with_capacity(self.codes.len());
        for &old_code in &self.codes {
            if old_code == -1 {
                codes.push(-1); // Keep original NA
            } else {
                let old_cat = &self.categories[old_code as usize];
                match category_map.get(old_cat) {
                    Some(&new_code) => codes.push(new_code),
                    None => codes.push(-1), // Set to NA if not in common categories
                }
            }
        }
        
        // Return new categorical
        Ok(Categorical {
            codes,
            categories: common_categories,
            category_map,
            ordered: self.ordered.clone(),
        })
    }
    
    /// Create the difference (categories in self but not in other) of two categorical data structures
    pub fn difference(&self, other: &Self) -> Result<Self> {
        // Collect difference categories
        let mut diff_categories = Vec::new();
        let mut category_map = HashMap::new();
        
        for cat in &self.categories {
            if !other.category_map.contains_key(cat) {
                let new_code = diff_categories.len() as i32;
                category_map.insert(cat.clone(), new_code);
                diff_categories.push(cat.clone());
            }
        }
        
        // Create new code list (set to NA if not in difference categories)
        let mut codes = Vec::with_capacity(self.codes.len());
        for &old_code in &self.codes {
            if old_code == -1 {
                codes.push(-1); // Keep original NA
            } else {
                let old_cat = &self.categories[old_code as usize];
                match category_map.get(old_cat) {
                    Some(&new_code) => codes.push(new_code),
                    None => codes.push(-1), // Set to NA if not in difference categories
                }
            }
        }
        
        // Return new categorical
        Ok(Categorical {
            codes,
            categories: diff_categories,
            category_map,
            ordered: self.ordered.clone(),
        })
    }
}

impl<T> PartialEq for Categorical<T>
where
    T: Debug + Clone + Eq + Hash + Display + Ord,
{
    fn eq(&self, other: &Self) -> bool {
        // Same length and content of codes
        if self.codes.len() != other.codes.len() {
            return false;
        }
        
        for (a, b) in self.codes.iter().zip(other.codes.iter()) {
            if a != b {
                return false;
            }
        }
        
        // Same length and content of categories
        if self.categories.len() != other.categories.len() {
            return false;
        }
        
        for (a, b) in self.categories.iter().zip(other.categories.iter()) {
            if a != b {
                return false;
            }
        }
        
        // Same order information
        self.ordered == other.ordered
    }
}

impl<T> PartialOrd for Categorical<T>
where
    T: Debug + Clone + Eq + Hash + Display + Ord,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Cannot compare if not ordered
        if self.ordered != CategoricalOrder::Ordered || other.ordered != CategoricalOrder::Ordered {
            return None;
        }
        
        // Cannot compare if categories are different
        if self.categories != other.categories {
            return None;
        }
        
        // Compare codes
        let len = self.codes.len().min(other.codes.len());
        for i in 0..len {
            let a = self.codes[i];
            let b = other.codes[i];
            
            // -1 is considered the smallest as it represents a missing value
            match (a, b) {
                (-1, -1) => continue,
                (-1, _) => return Some(Ordering::Less),
                (_, -1) => return Some(Ordering::Greater),
                (_, _) => {
                    if a < b {
                        return Some(Ordering::Less);
                    } else if a > b {
                        return Some(Ordering::Greater);
                    }
                }
            }
        }
        
        // Compare by length if all codes are the same
        self.codes.len().partial_cmp(&other.codes.len())
    }
}