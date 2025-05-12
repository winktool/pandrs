//! # Query Plan Explanation (Legacy)
//!
//! DEPRECATED: This file is maintained for backward compatibility only.
//! Please use the `distributed::explain` module directory structure instead.
//!
//! This module provides functionality for explaining query plans, helping users
//! understand how their queries will be executed.
//!
//! @deprecated

use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};

use crate::error::Result;
use super::execution::{ExecutionPlan, Operation};
use super::statistics::TableStatistics;

/// Formats for query plan explanation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplainFormat {
    /// Text format
    Text,
    /// JSON format
    Json,
    /// Dot format (for visualization)
    Dot,
}

/// Options for query plan explanation
#[derive(Debug, Clone)]
pub struct ExplainOptions {
    /// Format of the explanation
    pub format: ExplainFormat,
    /// Whether to include statistics
    pub with_statistics: bool,
    /// Whether to show the optimized plan
    pub optimized: bool,
    /// Whether to analyze the plan
    pub analyze: bool,
}

impl Default for ExplainOptions {
    fn default() -> Self {
        Self {
            format: ExplainFormat::Text,
            with_statistics: false,
            optimized: true,
            analyze: false,
        }
    }
}

/// Query plan node types
#[derive(Debug, Clone)]
pub enum PlanNode {
    /// Scan operation
    Scan {
        /// Table name
        table: String,
        /// Columns to scan
        columns: Vec<String>,
        /// Statistics for the table
        statistics: Option<TableStatistics>,
    },
    /// Project operation
    Project {
        /// Output columns
        columns: Vec<String>,
        /// Input node
        input: Box<PlanNode>,
    },
    /// Filter operation
    Filter {
        /// Filter predicate
        predicate: String,
        /// Input node
        input: Box<PlanNode>,
        /// Estimated selectivity (0.0-1.0)
        selectivity: Option<f64>,
    },
    /// Join operation
    Join {
        /// Join type
        join_type: String,
        /// Left input
        left: Box<PlanNode>,
        /// Right input
        right: Box<PlanNode>,
        /// Join keys
        keys: Vec<(String, String)>,
    },
    /// Aggregate operation
    Aggregate {
        /// Grouping keys
        keys: Vec<String>,
        /// Aggregation expressions
        aggregates: Vec<String>,
        /// Input node
        input: Box<PlanNode>,
    },
    /// Sort operation
    Sort {
        /// Sort expressions
        sort_exprs: Vec<String>,
        /// Input node
        input: Box<PlanNode>,
    },
    /// Limit operation
    Limit {
        /// Maximum number of rows
        limit: usize,
        /// Input node
        input: Box<PlanNode>,
    },
    /// Window operation
    Window {
        /// Window functions
        window_functions: Vec<String>,
        /// Input node
        input: Box<PlanNode>,
    },
    /// Custom operation
    Custom {
        /// Operation name
        name: String,
        /// Parameters
        params: HashMap<String, String>,
        /// Input node
        input: Box<PlanNode>,
    },
}

impl PlanNode {
    /// Explains the plan in the given format
    pub fn explain(&self, options: &ExplainOptions) -> String {
        match options.format {
            ExplainFormat::Text => self.explain_text(options, 0),
            ExplainFormat::Json => self.explain_json(options),
            ExplainFormat::Dot => self.explain_dot(options),
        }
    }
    
    /// Explains the plan in text format
    fn explain_text(&self, options: &ExplainOptions, indent: usize) -> String {
        let indent_str = " ".repeat(indent * 2);
        
        let node_str = match self {
            Self::Scan { table, columns, statistics } => {
                let mut result = format!("{}Scan: {}", indent_str, table);
                
                if !columns.is_empty() {
                    result.push_str(&format!(" [{}]", columns.join(", ")));
                }
                
                if options.with_statistics {
                    if let Some(stats) = statistics {
                        result.push_str(&format!(" (rows: {}, size: {} bytes)",
                            stats.row_count, stats.size_bytes));
                    }
                }
                
                result
            },
            Self::Project { columns, input } => {
                let mut result = format!("{}Project: [{}]", indent_str, columns.join(", "));
                result.push('\n');
                result.push_str(&input.explain_text(options, indent + 1));
                result
            },
            Self::Filter { predicate, input, selectivity } => {
                let mut result = format!("{}Filter: {}", indent_str, predicate);
                
                if options.with_statistics {
                    if let Some(sel) = selectivity {
                        result.push_str(&format!(" (selectivity: {:.2})", sel));
                    }
                }
                
                result.push('\n');
                result.push_str(&input.explain_text(options, indent + 1));
                result
            },
            Self::Join { join_type, left, right, keys } => {
                let keys_str = keys.iter()
                    .map(|(l, r)| format!("{} = {}", l, r))
                    .collect::<Vec<_>>()
                    .join(" AND ");
                
                let mut result = format!("{}Join: {} ON {}", indent_str, join_type, keys_str);
                result.push('\n');
                result.push_str(&left.explain_text(options, indent + 1));
                result.push('\n');
                result.push_str(&right.explain_text(options, indent + 1));
                result
            },
            Self::Aggregate { keys, aggregates, input } => {
                let mut result = format!("{}Aggregate: group by [{}], agg [{}]",
                    indent_str, keys.join(", "), aggregates.join(", "));
                result.push('\n');
                result.push_str(&input.explain_text(options, indent + 1));
                result
            },
            Self::Sort { sort_exprs, input } => {
                let mut result = format!("{}Sort: [{}]", indent_str, sort_exprs.join(", "));
                result.push('\n');
                result.push_str(&input.explain_text(options, indent + 1));
                result
            },
            Self::Limit { limit, input } => {
                let mut result = format!("{}Limit: {}", indent_str, limit);
                result.push('\n');
                result.push_str(&input.explain_text(options, indent + 1));
                result
            },
            Self::Window { window_functions, input } => {
                let mut result = format!("{}Window: [{}]", indent_str, window_functions.join(", "));
                result.push('\n');
                result.push_str(&input.explain_text(options, indent + 1));
                result
            },
            Self::Custom { name, params, input } => {
                let params_str = params.iter()
                    .map(|(k, v)| format!("{}: {}", k, v))
                    .collect::<Vec<_>>()
                    .join(", ");
                
                let mut result = format!("{}Custom: {} [{}]", indent_str, name, params_str);
                result.push('\n');
                result.push_str(&input.explain_text(options, indent + 1));
                result
            },
        };
        
        node_str
    }
    
    /// Explains the plan in JSON format
    fn explain_json(&self, options: &ExplainOptions) -> String {
        // Simplified JSON serialization
        match self {
            Self::Scan { table, columns, statistics } => {
                let stats_str = if options.with_statistics {
                    if let Some(stats) = statistics {
                        format!(", \"rows\": {}, \"size\": {}", stats.row_count, stats.size_bytes)
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                };
                
                format!(
                    "{{\"type\": \"Scan\", \"table\": \"{}\", \"columns\": [{}]{}}}",
                    table,
                    columns.iter().map(|c| format!("\"{}\"", c)).collect::<Vec<_>>().join(", "),
                    stats_str
                )
            },
            Self::Project { columns, input } => {
                format!(
                    "{{\"type\": \"Project\", \"columns\": [{}], \"input\": {}}}",
                    columns.iter().map(|c| format!("\"{}\"", c)).collect::<Vec<_>>().join(", "),
                    input.explain_json(options)
                )
            },
            Self::Filter { predicate, input, selectivity } => {
                let selectivity_str = if options.with_statistics {
                    if let Some(sel) = selectivity {
                        format!(", \"selectivity\": {:.2}", sel)
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                };
                
                format!(
                    "{{\"type\": \"Filter\", \"predicate\": \"{}\"{}, \"input\": {}}}",
                    escape_json(predicate),
                    selectivity_str,
                    input.explain_json(options)
                )
            },
            Self::Join { join_type, left, right, keys } => {
                let keys_json = keys.iter()
                    .map(|(l, r)| format!("{{\"left\": \"{}\", \"right\": \"{}\"}}", l, r))
                    .collect::<Vec<_>>()
                    .join(", ");
                
                format!(
                    "{{\"type\": \"Join\", \"join_type\": \"{}\", \"keys\": [{}], \"left\": {}, \"right\": {}}}",
                    join_type,
                    keys_json,
                    left.explain_json(options),
                    right.explain_json(options)
                )
            },
            Self::Aggregate { keys, aggregates, input } => {
                format!(
                    "{{\"type\": \"Aggregate\", \"keys\": [{}], \"aggregates\": [{}], \"input\": {}}}",
                    keys.iter().map(|k| format!("\"{}\"", k)).collect::<Vec<_>>().join(", "),
                    aggregates.iter().map(|a| format!("\"{}\"", a)).collect::<Vec<_>>().join(", "),
                    input.explain_json(options)
                )
            },
            Self::Sort { sort_exprs, input } => {
                format!(
                    "{{\"type\": \"Sort\", \"sort_exprs\": [{}], \"input\": {}}}",
                    sort_exprs.iter().map(|e| format!("\"{}\"", e)).collect::<Vec<_>>().join(", "),
                    input.explain_json(options)
                )
            },
            Self::Limit { limit, input } => {
                format!(
                    "{{\"type\": \"Limit\", \"limit\": {}, \"input\": {}}}",
                    limit,
                    input.explain_json(options)
                )
            },
            Self::Window { window_functions, input } => {
                format!(
                    "{{\"type\": \"Window\", \"window_functions\": [{}], \"input\": {}}}",
                    window_functions.iter().map(|w| format!("\"{}\"", w)).collect::<Vec<_>>().join(", "),
                    input.explain_json(options)
                )
            },
            Self::Custom { name, params, input } => {
                let params_json = params.iter()
                    .map(|(k, v)| format!("\"{}\"\"{}\"", k, v))
                    .collect::<Vec<_>>()
                    .join(", ");
                
                format!(
                    "{{\"type\": \"Custom\", \"name\": \"{}\", \"params\": {{{}}}, \"input\": {}}}",
                    name,
                    params_json,
                    input.explain_json(options)
                )
            },
        }
    }
    
    /// Explains the plan in DOT format (for visualization)
    fn explain_dot(&self, options: &ExplainOptions) -> String {
        let mut result = String::from("digraph plan {\n");
        result.push_str("  node [shape=box];\n");
        
        let mut node_counter = 0;
        self.explain_dot_recursive(&mut result, &mut node_counter, options);
        
        result.push_str("}\n");
        result
    }
    
    /// Helper for DOT format explanation
    fn explain_dot_recursive(&self, result: &mut String, counter: &mut usize, options: &ExplainOptions) -> usize {
        let current = *counter;
        *counter += 1;
        
        let node_label = match self {
            Self::Scan { table, columns, statistics } => {
                let cols_str = if columns.is_empty() {
                    String::new()
                } else {
                    format!("\\nColumns: [{}]", columns.join(", "))
                };
                
                let stats_str = if options.with_statistics {
                    if let Some(stats) = statistics {
                        format!("\\nRows: {}\\nSize: {} bytes", stats.row_count, stats.size_bytes)
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                };
                
                format!("Scan: {}{}{}", table, cols_str, stats_str)
            },
            Self::Project { columns, .. } => {
                format!("Project: [{}]", columns.join(", "))
            },
            Self::Filter { predicate, selectivity, .. } => {
                let selectivity_str = if options.with_statistics {
                    if let Some(sel) = selectivity {
                        format!("\\nSelectivity: {:.2}", sel)
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                };
                
                format!("Filter: {}{}", predicate, selectivity_str)
            },
            Self::Join { join_type, keys, .. } => {
                let keys_str = keys.iter()
                    .map(|(l, r)| format!("{} = {}", l, r))
                    .collect::<Vec<_>>()
                    .join(" AND ");
                
                format!("Join: {}\\nON {}", join_type, keys_str)
            },
            Self::Aggregate { keys, aggregates, .. } => {
                format!("Aggregate:\\nGroup by: [{}]\\nAgg: [{}]",
                    keys.join(", "), aggregates.join(", "))
            },
            Self::Sort { sort_exprs, .. } => {
                format!("Sort: [{}]", sort_exprs.join(", "))
            },
            Self::Limit { limit, .. } => {
                format!("Limit: {}", limit)
            },
            Self::Window { window_functions, .. } => {
                format!("Window: [{}]", window_functions.join(", "))
            },
            Self::Custom { name, params, .. } => {
                let params_str = params.iter()
                    .map(|(k, v)| format!("{}: {}", k, v))
                    .collect::<Vec<_>>()
                    .join(", ");
                
                format!("Custom: {}\\n[{}]", name, params_str)
            },
        };
        
        result.push_str(&format!("  node{} [label=\"{}\"];\n", current, escape_dot(&node_label)));
        
        match self {
            Self::Scan { .. } => {},
            Self::Project { input, .. } |
            Self::Filter { input, .. } |
            Self::Aggregate { input, .. } |
            Self::Sort { input, .. } |
            Self::Limit { input, .. } |
            Self::Window { input, .. } |
            Self::Custom { input, .. } => {
                let child = input.explain_dot_recursive(result, counter, options);
                result.push_str(&format!("  node{} -> node{};\n", current, child));
            },
            Self::Join { left, right, .. } => {
                let left_child = left.explain_dot_recursive(result, counter, options);
                let right_child = right.explain_dot_recursive(result, counter, options);
                result.push_str(&format!("  node{} -> node{};\n", current, left_child));
                result.push_str(&format!("  node{} -> node{};\n", current, right_child));
            },
        }
        
        current
    }
}

impl Display for PlanNode {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.explain(&ExplainOptions::default()))
    }
}

/// Explains an execution plan
pub fn explain_plan(plan: &ExecutionPlan, options: &ExplainOptions) -> Result<String> {
    let plan_node = execution_plan_to_plan_node(plan)?;
    Ok(plan_node.explain(options))
}

/// Converts an execution plan to a plan node
fn execution_plan_to_plan_node(plan: &ExecutionPlan) -> Result<PlanNode> {
    let node = match plan.operation() {
        Operation::Select { columns } => {
            let input = dummy_scan_node(plan);
            
            PlanNode::Project {
                columns: columns.clone(),
                input: Box::new(input),
            }
        },
        Operation::Filter { predicate } => {
            let input = dummy_scan_node(plan);
            
            PlanNode::Filter {
                predicate: predicate.clone(),
                input: Box::new(input),
                selectivity: None,
            }
        },
        Operation::Join { right, join_type, left_keys, right_keys } => {
            let left_input = dummy_scan_node(plan);
            let right_input = PlanNode::Scan {
                table: right.clone(),
                columns: vec![],
                statistics: None,
            };
            
            let join_type_str = match join_type {
                super::execution::JoinType::Inner => "INNER",
                super::execution::JoinType::Left => "LEFT",
                super::execution::JoinType::Right => "RIGHT",
                super::execution::JoinType::Full => "FULL OUTER",
                super::execution::JoinType::Cross => "CROSS",
            };
            
            let keys = left_keys.iter()
                .zip(right_keys.iter())
                .map(|(l, r)| (l.clone(), r.clone()))
                .collect();
            
            PlanNode::Join {
                join_type: join_type_str.to_string(),
                left: Box::new(left_input),
                right: Box::new(right_input),
                keys,
            }
        },
        Operation::GroupBy { keys, aggregates } => {
            let input = dummy_scan_node(plan);
            let agg_exprs = aggregates.iter()
                .map(|agg| format!("{}({}) as {}", agg.function, agg.input, agg.output))
                .collect();
            
            PlanNode::Aggregate {
                keys: keys.clone(),
                aggregates: agg_exprs,
                input: Box::new(input),
            }
        },
        Operation::OrderBy { sort_exprs } => {
            let input = dummy_scan_node(plan);
            let sort_exprs_str = sort_exprs.iter()
                .map(|expr| {
                    let direction = if expr.ascending { "ASC" } else { "DESC" };
                    let nulls = if expr.nulls_first { "NULLS FIRST" } else { "NULLS LAST" };
                    format!("{} {} {}", expr.column, direction, nulls)
                })
                .collect();
            
            PlanNode::Sort {
                sort_exprs: sort_exprs_str,
                input: Box::new(input),
            }
        },
        Operation::Limit { limit } => {
            let input = dummy_scan_node(plan);
            
            PlanNode::Limit {
                limit: *limit,
                input: Box::new(input),
            }
        },
        Operation::Window { window_functions } => {
            let input = dummy_scan_node(plan);
            let window_exprs = window_functions.iter()
                .map(|wf| wf.to_sql())
                .collect();
            
            PlanNode::Window {
                window_functions: window_exprs,
                input: Box::new(input),
            }
        },
        Operation::Custom { name, params } => {
            let input = dummy_scan_node(plan);
            
            PlanNode::Custom {
                name: name.clone(),
                params: params.clone(),
                input: Box::new(input),
            }
        },
    };
    
    Ok(node)
}

/// Creates a dummy scan node for an execution plan's input
fn dummy_scan_node(plan: &ExecutionPlan) -> PlanNode {
    if plan.inputs().is_empty() {
        // No inputs, use a dummy table
        PlanNode::Scan {
            table: "DUMMY".to_string(),
            columns: vec![],
            statistics: None,
        }
    } else {
        // Use the first input as the scan source
        PlanNode::Scan {
            table: plan.inputs()[0].clone(),
            columns: vec![],
            statistics: None,
        }
    }
}

/// Escapes a string for JSON
fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Escapes a string for DOT
fn escape_dot(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}