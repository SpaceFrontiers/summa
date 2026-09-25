//! Query Field Router - Routes queries to specific fields based on regex patterns
//!
//! This module provides functionality to detect if a query matches a regex pattern,
//! extract capture groups, substitute them into a template, and route the result
//! to a specific field instead of (or in addition to) default fields.
//!
//! # Template Language
//!
//! The substitution template supports:
//! - Simple group references: `{0}`, `{1}`, `{2}` etc.
//! - Expression syntax with functions: `{g(1).replace('-', '').lower()}`
//!
//! ## Available Functions
//!
//! - `g(n)` - Get capture group n (0 = entire match, 1+ = capture groups)
//! - `.replace(from, to)` - Replace all occurrences of `from` with `to`
//! - `.lower()` - Convert to lowercase
//! - `.upper()` - Convert to uppercase
//! - `.trim()` - Remove leading/trailing whitespace
//!
//! # Example
//!
//! ```text
//! # In SDL:
//! index documents {
//!     field title: text [indexed, stored]
//!     field uri: text [indexed, stored]
//!
//!     # Route DOI queries to uri field exclusively
//!     query_router {
//!         pattern: r"10\.\d{4,}/[^\s]+"
//!         substitution: "doi://{0}"
//!         target_field: uri
//!         mode: exclusive
//!     }
//!
//!     # Route ISBN with hyphen removal
//!     query_router {
//!         pattern: r"^isbn:([\d\-]+)$"
//!         substitution: "isbn://{g(1).replace('-', '')}"
//!         target_field: uri
//!         mode: exclusive
//!     }
//! }
//! ```

use regex::Regex;
use serde::{Deserialize, Serialize};

/// Routing mode for matched queries
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RoutingMode {
    /// Query only the target field (replace default fields)
    #[serde(rename = "exclusive")]
    Exclusive,
    /// Query both target field and default fields
    #[serde(rename = "additional")]
    #[default]
    Additional,
}

/// A single query routing rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRouterRule {
    /// Regex pattern to match against the query
    pub pattern: String,
    /// Substitution template using {0}, {1}, etc. for capture groups
    /// {0} is the entire match, {1} is first capture group, etc.
    pub substitution: String,
    /// Target field name to route the substituted query to
    pub target_field: String,
    /// Whether this is exclusive (replaces default) or additional
    #[serde(default)]
    pub mode: RoutingMode,
}

/// Result of applying a routing rule
#[derive(Debug, Clone)]
pub struct RoutedQuery {
    /// The transformed query string
    pub query: String,
    /// Target field name
    pub target_field: String,
    /// Routing mode
    pub mode: RoutingMode,
}

/// Template expression evaluator
///
/// Evaluates expressions like `{g(1).replace('-', '').lower()}`
mod template {
    use regex::Captures;

    /// Evaluate a substitution template with the given regex captures
    pub fn evaluate(template: &str, captures: &Captures) -> String {
        let mut result = String::new();
        let mut chars = template.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '{' {
                // Parse expression until closing brace
                let mut expr = String::new();
                let mut brace_depth = 1;

                for c in chars.by_ref() {
                    if c == '{' {
                        brace_depth += 1;
                        expr.push(c);
                    } else if c == '}' {
                        brace_depth -= 1;
                        if brace_depth == 0 {
                            break;
                        }
                        expr.push(c);
                    } else {
                        expr.push(c);
                    }
                }

                // Evaluate the expression
                let value = evaluate_expr(&expr, captures);
                result.push_str(&value);
            } else {
                result.push(c);
            }
        }

        result
    }

    /// Evaluate a single expression (content inside {})
    fn evaluate_expr(expr: &str, captures: &Captures) -> String {
        let expr = expr.trim();

        // Check for simple numeric reference like "0", "1", "2"
        if let Ok(group_num) = expr.parse::<usize>() {
            return captures
                .get(group_num)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();
        }

        // Parse expression with function calls
        parse_and_evaluate(expr, captures)
    }

    /// Parse and evaluate an expression like `g(1).replace('-', '').lower()`
    fn parse_and_evaluate(expr: &str, captures: &Captures) -> String {
        let mut chars = expr.chars().peekable();
        let mut value = String::new();

        // Skip whitespace
        while chars.peek() == Some(&' ') {
            chars.next();
        }

        // Parse initial value (must start with g(n))
        if expr.starts_with("g(") {
            // Parse g(n)
            chars.next(); // 'g'
            chars.next(); // '('

            let mut num_str = String::new();
            while let Some(&c) = chars.peek() {
                if c == ')' {
                    chars.next();
                    break;
                }
                num_str.push(c);
                chars.next();
            }

            if let Ok(group_num) = num_str.trim().parse::<usize>() {
                value = captures
                    .get(group_num)
                    .map(|m| m.as_str().to_string())
                    .unwrap_or_default();
            }
        } else {
            // Unknown expression start
            return expr.to_string();
        }

        // Parse method chain
        while chars.peek().is_some() {
            // Skip whitespace
            while chars.peek() == Some(&' ') {
                chars.next();
            }

            // Expect '.'
            if chars.peek() != Some(&'.') {
                break;
            }
            chars.next(); // consume '.'

            // Parse method name
            let mut method_name = String::new();
            while let Some(&c) = chars.peek() {
                if c == '(' || c == ' ' {
                    break;
                }
                method_name.push(c);
                chars.next();
            }

            // Skip whitespace
            while chars.peek() == Some(&' ') {
                chars.next();
            }

            // Parse arguments if present
            let args = if chars.peek() == Some(&'(') {
                chars.next(); // consume '('
                parse_args(&mut chars)
            } else {
                vec![]
            };

            // Apply method
            value = apply_method(&value, &method_name, &args);
        }

        value
    }

    /// Parse function arguments from the char iterator
    fn parse_args(chars: &mut std::iter::Peekable<std::str::Chars>) -> Vec<String> {
        let mut args = Vec::new();
        let mut current_arg = String::new();
        let mut in_string = false;
        let mut string_char = '"';

        for c in chars.by_ref() {
            if c == ')' && !in_string {
                // End of arguments
                let arg = current_arg.trim().to_string();
                if !arg.is_empty() {
                    args.push(parse_string_literal(&arg));
                }
                break;
            } else if (c == '"' || c == '\'') && !in_string {
                in_string = true;
                string_char = c;
                current_arg.push(c);
            } else if c == string_char && in_string {
                in_string = false;
                current_arg.push(c);
            } else if c == ',' && !in_string {
                let arg = current_arg.trim().to_string();
                if !arg.is_empty() {
                    args.push(parse_string_literal(&arg));
                }
                current_arg.clear();
            } else {
                current_arg.push(c);
            }
        }

        args
    }

    /// Parse a string literal, removing quotes
    fn parse_string_literal(s: &str) -> String {
        let s = s.trim();
        if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
            s[1..s.len() - 1].to_string()
        } else {
            s.to_string()
        }
    }

    /// Apply a method to a value
    fn apply_method(value: &str, method: &str, args: &[String]) -> String {
        match method {
            "replace" => {
                if args.len() >= 2 {
                    value.replace(&args[0], &args[1])
                } else if args.len() == 1 {
                    value.replace(&args[0], "")
                } else {
                    value.to_string()
                }
            }
            "lower" | "lowercase" => value.to_lowercase(),
            "upper" | "uppercase" => value.to_uppercase(),
            "trim" => value.trim().to_string(),
            "trim_start" | "ltrim" => value.trim_start().to_string(),
            "trim_end" | "rtrim" => value.trim_end().to_string(),
            _ => value.to_string(),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use regex::Regex;

        fn make_captures<'a>(pattern: &str, text: &'a str) -> Option<Captures<'a>> {
            Regex::new(pattern).ok()?.captures(text)
        }

        #[test]
        fn test_simple_substitution() {
            let caps = make_captures(r"(\d+)", "hello 123 world").unwrap();
            assert_eq!(evaluate("value: {1}", &caps), "value: 123");
        }

        #[test]
        fn test_g_function() {
            let caps = make_captures(r"(\d+)", "hello 123 world").unwrap();
            assert_eq!(evaluate("{g(1)}", &caps), "123");
            assert_eq!(evaluate("{g(0)}", &caps), "123");
        }

        #[test]
        fn test_replace_function() {
            let caps = make_captures(r"([\d\-]+)", "isbn:978-3-16-148410-0").unwrap();
            assert_eq!(evaluate("{g(1).replace('-', '')}", &caps), "9783161484100");
        }

        #[test]
        fn test_lower_function() {
            let caps = make_captures(r"(\w+)", "HELLO").unwrap();
            assert_eq!(evaluate("{g(1).lower()}", &caps), "hello");
        }

        #[test]
        fn test_upper_function() {
            let caps = make_captures(r"(\w+)", "hello").unwrap();
            assert_eq!(evaluate("{g(1).upper()}", &caps), "HELLO");
        }

        #[test]
        fn test_trim_function() {
            let caps = make_captures(r"(.+)", "  hello  ").unwrap();
            assert_eq!(evaluate("{g(1).trim()}", &caps), "hello");
        }

        #[test]
        fn test_chained_functions() {
            let caps = make_captures(r"([\d\-]+)", "978-3-16").unwrap();
            assert_eq!(evaluate("{g(1).replace('-', '').lower()}", &caps), "978316");
        }

        #[test]
        fn test_mixed_template() {
            let caps = make_captures(r"isbn:([\d\-]+)", "isbn:978-3-16").unwrap();
            assert_eq!(
                evaluate("isbn://{g(1).replace('-', '')}", &caps),
                "isbn://978316"
            );
        }

        #[test]
        fn test_multiple_expressions() {
            let caps = make_captures(r"(\w+):(\w+)", "key:VALUE").unwrap();
            assert_eq!(
                evaluate("{g(1).upper()}={g(2).lower()}", &caps),
                "KEY=value"
            );
        }
    }
}

/// Compiled query router rule with pre-compiled regex
#[derive(Debug, Clone)]
pub struct CompiledRouterRule {
    regex: Regex,
    substitution: String,
    target_field: String,
    mode: RoutingMode,
}

impl CompiledRouterRule {
    /// Create a new compiled router rule
    pub fn new(rule: &QueryRouterRule) -> Result<Self, String> {
        let regex = Regex::new(&rule.pattern)
            .map_err(|e| format!("Invalid regex pattern '{}': {}", rule.pattern, e))?;

        Ok(Self {
            regex,
            substitution: rule.substitution.clone(),
            target_field: rule.target_field.clone(),
            mode: rule.mode,
        })
    }

    /// Try to match and transform a query
    pub fn try_match(&self, query: &str) -> Option<RoutedQuery> {
        let captures = self.regex.captures(query)?;

        // Use the template evaluator for substitution
        let result = template::evaluate(&self.substitution, &captures);

        Some(RoutedQuery {
            query: result,
            target_field: self.target_field.clone(),
            mode: self.mode,
        })
    }

    /// Get the target field name
    pub fn target_field(&self) -> &str {
        &self.target_field
    }

    /// Get the routing mode
    pub fn mode(&self) -> RoutingMode {
        self.mode
    }
}

/// Query field router that holds multiple routing rules
#[derive(Debug, Clone, Default)]
pub struct QueryFieldRouter {
    rules: Vec<CompiledRouterRule>,
}

impl QueryFieldRouter {
    /// Create a new empty router
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Create a router from a list of rules
    pub fn from_rules(rules: &[QueryRouterRule]) -> Result<Self, String> {
        let compiled: Result<Vec<_>, _> = rules.iter().map(CompiledRouterRule::new).collect();
        Ok(Self { rules: compiled? })
    }

    /// Add a rule to the router
    pub fn add_rule(&mut self, rule: &QueryRouterRule) -> Result<(), String> {
        self.rules.push(CompiledRouterRule::new(rule)?);
        Ok(())
    }

    /// Check if router has any rules
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Get the number of rules
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// Try to route a query, returning the first matching rule's result
    pub fn route(&self, query: &str) -> Option<RoutedQuery> {
        for rule in &self.rules {
            if let Some(routed) = rule.try_match(query) {
                return Some(routed);
            }
        }
        None
    }

    /// Try to route a query, returning all matching rules' results
    pub fn route_all(&self, query: &str) -> Vec<RoutedQuery> {
        self.rules
            .iter()
            .filter_map(|rule| rule.try_match(query))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_doi_routing() {
        let rule = QueryRouterRule {
            pattern: r"(10\.\d{4,}/[^\s]+)".to_string(),
            substitution: "doi://{1}".to_string(),
            target_field: "uri".to_string(),
            mode: RoutingMode::Exclusive,
        };

        let compiled = CompiledRouterRule::new(&rule).unwrap();

        // Should match DOI
        let result = compiled.try_match("10.1234/abc.123").unwrap();
        assert_eq!(result.query, "doi://10.1234/abc.123");
        assert_eq!(result.target_field, "uri");
        assert_eq!(result.mode, RoutingMode::Exclusive);

        // Should not match non-DOI
        assert!(compiled.try_match("hello world").is_none());
    }

    #[test]
    fn test_full_match_substitution() {
        let rule = QueryRouterRule {
            pattern: r"^#(\d+)$".to_string(),
            substitution: "{1}".to_string(),
            target_field: "issue_number".to_string(),
            mode: RoutingMode::Exclusive,
        };

        let compiled = CompiledRouterRule::new(&rule).unwrap();

        let result = compiled.try_match("#42").unwrap();
        assert_eq!(result.query, "42");
        assert_eq!(result.target_field, "issue_number");
    }

    #[test]
    fn test_multiple_capture_groups() {
        let rule = QueryRouterRule {
            pattern: r"(\w+):(\w+)".to_string(),
            substitution: "field={1} value={2}".to_string(),
            target_field: "custom".to_string(),
            mode: RoutingMode::Additional,
        };

        let compiled = CompiledRouterRule::new(&rule).unwrap();

        let result = compiled.try_match("author:smith").unwrap();
        assert_eq!(result.query, "field=author value=smith");
        assert_eq!(result.mode, RoutingMode::Additional);
    }

    #[test]
    fn test_router_with_multiple_rules() {
        let rules = vec![
            QueryRouterRule {
                pattern: r"^doi:(10\.\d{4,}/[^\s]+)$".to_string(),
                substitution: "doi://{1}".to_string(),
                target_field: "uri".to_string(),
                mode: RoutingMode::Exclusive,
            },
            QueryRouterRule {
                pattern: r"^pmid:(\d+)$".to_string(),
                substitution: "pubmed://{1}".to_string(),
                target_field: "uri".to_string(),
                mode: RoutingMode::Exclusive,
            },
        ];

        let router = QueryFieldRouter::from_rules(&rules).unwrap();

        // Match first rule
        let result = router.route("doi:10.1234/test").unwrap();
        assert_eq!(result.query, "doi://10.1234/test");

        // Match second rule
        let result = router.route("pmid:12345678").unwrap();
        assert_eq!(result.query, "pubmed://12345678");

        // No match
        assert!(router.route("random query").is_none());
    }

    #[test]
    fn test_invalid_regex() {
        let rule = QueryRouterRule {
            pattern: r"[invalid".to_string(),
            substitution: "{0}".to_string(),
            target_field: "test".to_string(),
            mode: RoutingMode::Exclusive,
        };

        assert!(CompiledRouterRule::new(&rule).is_err());
    }

    #[test]
    fn test_routing_mode_default() {
        let mode: RoutingMode = Default::default();
        assert_eq!(mode, RoutingMode::Additional);
    }
}
