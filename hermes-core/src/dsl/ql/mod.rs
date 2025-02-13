//! Query language parser using pest
//!
//! Supports:
//! - Term queries: `rust` or `title:rust`
//! - Phrase queries: `"hello world"` or `title:"hello world"`
//! - Boolean operators: `AND`, `OR`, `NOT` (or `&&`, `||`, `-`)
//! - Grouping: `(rust OR python) AND programming`
//! - Default fields for unqualified terms

use pest::Parser;
use pest_derive::Parser;
use std::sync::Arc;

use super::query_field_router::{QueryFieldRouter, RoutingMode};
use super::schema::{Field, Schema};
use crate::query::{BooleanQuery, Query, TermQuery};
use crate::tokenizer::{BoxedTokenizer, TokenizerRegistry};

#[derive(Parser)]
#[grammar = "dsl/ql/ql.pest"]
struct QueryParser;

/// Parsed query that can be converted to a Query trait object
#[derive(Debug, Clone)]
pub enum ParsedQuery {
    Term {
        field: Option<String>,
        term: String,
    },
    Phrase {
        field: Option<String>,
        phrase: String,
    },
    /// Dense vector ANN query
    Ann {
        field: String,
        vector: Vec<f32>,
        nprobe: usize,
        rerank: f32,
    },
    /// Sparse vector query
    Sparse {
        field: String,
        vector: Vec<(u32, f32)>,
    },
    And(Vec<ParsedQuery>),
    Or(Vec<ParsedQuery>),
    Not(Box<ParsedQuery>),
}

/// Query language parser with schema awareness
pub struct QueryLanguageParser {
    schema: Arc<Schema>,
    default_fields: Vec<Field>,
    tokenizers: Arc<TokenizerRegistry>,
    /// Optional query field router for routing queries based on regex patterns
    field_router: Option<QueryFieldRouter>,
}

impl QueryLanguageParser {
    pub fn new(
        schema: Arc<Schema>,
        default_fields: Vec<Field>,
        tokenizers: Arc<TokenizerRegistry>,
    ) -> Self {
        Self {
            schema,
            default_fields,
            tokenizers,
            field_router: None,
        }
    }

    /// Create a parser with a query field router
    pub fn with_router(
        schema: Arc<Schema>,
        default_fields: Vec<Field>,
        tokenizers: Arc<TokenizerRegistry>,
        router: QueryFieldRouter,
    ) -> Self {
        Self {
            schema,
            default_fields,
            tokenizers,
            field_router: Some(router),
        }
    }

    /// Set the query field router
    pub fn set_router(&mut self, router: QueryFieldRouter) {
        self.field_router = Some(router);
    }

    /// Get the query field router
    pub fn router(&self) -> Option<&QueryFieldRouter> {
        self.field_router.as_ref()
    }

    /// Parse a query string into a Query
    ///
    /// Supports query language syntax (field:term, AND, OR, NOT, grouping)
    /// and plain text (tokenized and searched across default fields).
    ///
    /// If a query field router is configured, the query is first checked against
    /// routing rules. If a rule matches:
    /// - In exclusive mode: only the target field is queried with the substituted value
    /// - In additional mode: both the target field and default fields are queried
    pub fn parse(&self, query_str: &str) -> Result<Box<dyn Query>, String> {
        let query_str = query_str.trim();
        if query_str.is_empty() {
            return Err("Empty query".to_string());
        }

        // Check if query matches any routing rules
        if let Some(router) = &self.field_router
            && let Some(routed) = router.route(query_str)
        {
            return self.build_routed_query(
                &routed.query,
                &routed.target_field,
                routed.mode,
                query_str,
            );
        }

        // No routing match - parse normally
        self.parse_normal(query_str)
    }

    /// Build a query from a routed match
    fn build_routed_query(
        &self,
        routed_query: &str,
        target_field: &str,
        mode: RoutingMode,
        original_query: &str,
    ) -> Result<Box<dyn Query>, String> {
        // Validate target field exists
        let _field_id = self
            .schema
            .get_field(target_field)
            .ok_or_else(|| format!("Unknown target field: {}", target_field))?;

        // Build query for the target field with the substituted value
        let target_query = self.build_term_query(Some(target_field), routed_query)?;

        match mode {
            RoutingMode::Exclusive => {
                // Only query the target field
                Ok(target_query)
            }
            RoutingMode::Additional => {
                // Query both target field and default fields
                let mut bool_query = BooleanQuery::new();
                bool_query = bool_query.should(target_query);

                // Also parse the original query against default fields
                if let Ok(default_query) = self.parse_normal(original_query) {
                    bool_query = bool_query.should(default_query);
                }

                Ok(Box::new(bool_query))
            }
        }
    }

    /// Parse query without routing (normal parsing path)
    fn parse_normal(&self, query_str: &str) -> Result<Box<dyn Query>, String> {
        // Try parsing as query language first
        match self.parse_query_string(query_str) {
            Ok(parsed) => self.build_query(&parsed),
            Err(_) => {
                // If grammar parsing fails, treat as plain text
                // Split by whitespace and create OR of terms
                self.parse_plain_text(query_str)
            }
        }
    }

    /// Parse plain text as implicit OR of tokenized terms
    fn parse_plain_text(&self, text: &str) -> Result<Box<dyn Query>, String> {
        if self.default_fields.is_empty() {
            return Err("No default fields configured".to_string());
        }

        let tokenizer = self.get_tokenizer(self.default_fields[0]);
        let tokens: Vec<String> = tokenizer
            .tokenize(text)
            .into_iter()
            .map(|t| t.text.to_lowercase())
            .collect();

        if tokens.is_empty() {
            return Err("No tokens in query".to_string());
        }

        let mut bool_query = BooleanQuery::new();
        for token in &tokens {
            for &field_id in &self.default_fields {
                bool_query = bool_query.should(TermQuery::text(field_id, token));
            }
        }
        Ok(Box::new(bool_query))
    }

    fn parse_query_string(&self, query_str: &str) -> Result<ParsedQuery, String> {
        let pairs = QueryParser::parse(Rule::query, query_str)
            .map_err(|e| format!("Parse error: {}", e))?;

        let query_pair = pairs.into_iter().next().ok_or("No query found")?;

        // query = { SOI ~ or_expr ~ EOI }
        self.parse_or_expr(query_pair.into_inner().next().unwrap())
    }

    fn parse_or_expr(&self, pair: pest::iterators::Pair<Rule>) -> Result<ParsedQuery, String> {
        let mut inner = pair.into_inner();
        let first = self.parse_and_expr(inner.next().unwrap())?;

        let rest: Vec<ParsedQuery> = inner
            .filter(|p| p.as_rule() == Rule::and_expr)
            .map(|p| self.parse_and_expr(p))
            .collect::<Result<Vec<_>, _>>()?;

        if rest.is_empty() {
            Ok(first)
        } else {
            let mut all = vec![first];
            all.extend(rest);
            Ok(ParsedQuery::Or(all))
        }
    }

    fn parse_and_expr(&self, pair: pest::iterators::Pair<Rule>) -> Result<ParsedQuery, String> {
        let mut inner = pair.into_inner();
        let first = self.parse_primary(inner.next().unwrap())?;

        let rest: Vec<ParsedQuery> = inner
            .filter(|p| p.as_rule() == Rule::primary)
            .map(|p| self.parse_primary(p))
            .collect::<Result<Vec<_>, _>>()?;

        if rest.is_empty() {
            Ok(first)
        } else {
            let mut all = vec![first];
            all.extend(rest);
            Ok(ParsedQuery::And(all))
        }
    }

    fn parse_primary(&self, pair: pest::iterators::Pair<Rule>) -> Result<ParsedQuery, String> {
        let mut negated = false;
        let mut inner_query = None;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::not_op => negated = true,
                Rule::group => {
                    let or_expr = inner.into_inner().next().unwrap();
                    inner_query = Some(self.parse_or_expr(or_expr)?);
                }
                Rule::ann_query => {
                    inner_query = Some(self.parse_ann_query(inner)?);
                }
                Rule::sparse_query => {
                    inner_query = Some(self.parse_sparse_query(inner)?);
                }
                Rule::phrase_query => {
                    inner_query = Some(self.parse_phrase_query(inner)?);
                }
                Rule::term_query => {
                    inner_query = Some(self.parse_term_query(inner)?);
                }
                _ => {}
            }
        }

        let query = inner_query.ok_or("No query in primary")?;

        if negated {
            Ok(ParsedQuery::Not(Box::new(query)))
        } else {
            Ok(query)
        }
    }

    fn parse_term_query(&self, pair: pest::iterators::Pair<Rule>) -> Result<ParsedQuery, String> {
        let mut field = None;
        let mut term = String::new();

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::field_spec => {
                    field = Some(inner.into_inner().next().unwrap().as_str().to_string());
                }
                Rule::term => {
                    term = inner.as_str().to_string();
                }
                _ => {}
            }
        }

        Ok(ParsedQuery::Term { field, term })
    }

    fn parse_phrase_query(&self, pair: pest::iterators::Pair<Rule>) -> Result<ParsedQuery, String> {
        let mut field = None;
        let mut phrase = String::new();

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::field_spec => {
                    field = Some(inner.into_inner().next().unwrap().as_str().to_string());
                }
                Rule::quoted_string => {
                    let s = inner.as_str();
                    phrase = s[1..s.len() - 1].to_string();
                }
                _ => {}
            }
        }

        Ok(ParsedQuery::Phrase { field, phrase })
    }

    /// Parse an ANN query: field:ann([1.0, 2.0, 3.0], nprobe=32, rerank=3)
    fn parse_ann_query(&self, pair: pest::iterators::Pair<Rule>) -> Result<ParsedQuery, String> {
        let mut field = String::new();
        let mut vector = Vec::new();
        let mut nprobe = 32usize;
        let mut rerank = 3.0f32;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::field_spec => {
                    field = inner.into_inner().next().unwrap().as_str().to_string();
                }
                Rule::vector_array => {
                    for num in inner.into_inner() {
                        if num.as_rule() == Rule::number
                            && let Ok(v) = num.as_str().parse::<f32>()
                        {
                            vector.push(v);
                        }
                    }
                }
                Rule::ann_params => {
                    for param in inner.into_inner() {
                        if param.as_rule() == Rule::ann_param {
                            // ann_param = { ("nprobe" | "rerank") ~ "=" ~ number }
                            let param_str = param.as_str();
                            if let Some(eq_pos) = param_str.find('=') {
                                let name = &param_str[..eq_pos];
                                let value = &param_str[eq_pos + 1..];
                                match name {
                                    "nprobe" => nprobe = value.parse().unwrap_or(0),
                                    "rerank" => rerank = value.parse().unwrap_or(0.0),
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(ParsedQuery::Ann {
            field,
            vector,
            nprobe,
            rerank,
        })
    }

    /// Parse a sparse vector query: field:sparse({1: 0.5, 5: 0.3})
    fn parse_sparse_query(&self, pair: pest::iterators::Pair<Rule>) -> Result<ParsedQuery, String> {
        let mut field = String::new();
        let mut vector = Vec::new();

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::field_spec => {
                    field = inner.into_inner().next().unwrap().as_str().to_string();
                }
                Rule::sparse_map => {
                    for entry in inner.into_inner() {
                        if entry.as_rule() == Rule::sparse_entry {
                            let mut entry_inner = entry.into_inner();
                            if let (Some(idx), Some(weight)) =
                                (entry_inner.next(), entry_inner.next())
                                && let (Ok(i), Ok(w)) =
                                    (idx.as_str().parse::<u32>(), weight.as_str().parse::<f32>())
                            {
                                vector.push((i, w));
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(ParsedQuery::Sparse { field, vector })
    }

    fn build_query(&self, parsed: &ParsedQuery) -> Result<Box<dyn Query>, String> {
        use crate::query::{DenseVectorQuery, SparseVectorQuery};

        match parsed {
            ParsedQuery::Term { field, term } => self.build_term_query(field.as_deref(), term),
            ParsedQuery::Phrase { field, phrase } => {
                self.build_phrase_query(field.as_deref(), phrase)
            }
            ParsedQuery::Ann {
                field,
                vector,
                nprobe,
                rerank,
            } => {
                let field_id = self
                    .schema
                    .get_field(field)
                    .ok_or_else(|| format!("Unknown field: {}", field))?;
                let query = DenseVectorQuery::new(field_id, vector.clone())
                    .with_nprobe(*nprobe)
                    .with_rerank_factor(*rerank);
                Ok(Box::new(query))
            }
            ParsedQuery::Sparse { field, vector } => {
                let field_id = self
                    .schema
                    .get_field(field)
                    .ok_or_else(|| format!("Unknown field: {}", field))?;
                let query = SparseVectorQuery::new(field_id, vector.clone());
                Ok(Box::new(query))
            }
            ParsedQuery::And(queries) => {
                let mut bool_query = BooleanQuery::new();
                for q in queries {
                    bool_query = bool_query.must(self.build_query(q)?);
                }
                Ok(Box::new(bool_query))
            }
            ParsedQuery::Or(queries) => {
                let mut bool_query = BooleanQuery::new();
                for q in queries {
                    bool_query = bool_query.should(self.build_query(q)?);
                }
                Ok(Box::new(bool_query))
            }
            ParsedQuery::Not(inner) => {
                // NOT query needs a context - wrap in a match-all with must_not
                let mut bool_query = BooleanQuery::new();
                bool_query = bool_query.must_not(self.build_query(inner)?);
                Ok(Box::new(bool_query))
            }
        }
    }

    fn build_term_query(&self, field: Option<&str>, term: &str) -> Result<Box<dyn Query>, String> {
        if let Some(field_name) = field {
            // Field-qualified term: tokenize using field's tokenizer
            let field_id = self
                .schema
                .get_field(field_name)
                .ok_or_else(|| format!("Unknown field: {}", field_name))?;
            // Validate field type — TermQuery only works on text fields
            if let Some(entry) = self.schema.get_field_entry(field_id) {
                use crate::dsl::FieldType;
                if entry.field_type != FieldType::Text {
                    return Err(format!(
                        "Term query requires a text field, but '{}' is {:?}. Use range query for numeric fields.",
                        field_name, entry.field_type
                    ));
                }
            }
            let tokenizer = self.get_tokenizer(field_id);
            let tokens: Vec<String> = tokenizer
                .tokenize(term)
                .into_iter()
                .map(|t| t.text.to_lowercase())
                .collect();

            if tokens.is_empty() {
                return Err("No tokens in term".to_string());
            }

            if tokens.len() == 1 {
                Ok(Box::new(TermQuery::text(field_id, &tokens[0])))
            } else {
                // Multiple tokens from single term - AND them together
                let mut bool_query = BooleanQuery::new();
                for token in &tokens {
                    bool_query = bool_query.must(TermQuery::text(field_id, token));
                }
                Ok(Box::new(bool_query))
            }
        } else if !self.default_fields.is_empty() {
            // Unqualified term: tokenize and search across default fields
            let tokenizer = self.get_tokenizer(self.default_fields[0]);
            let tokens: Vec<String> = tokenizer
                .tokenize(term)
                .into_iter()
                .map(|t| t.text.to_lowercase())
                .collect();

            if tokens.is_empty() {
                return Err("No tokens in term".to_string());
            }

            // Build SHOULD query across all default fields for each token
            let mut bool_query = BooleanQuery::new();
            for token in &tokens {
                for &field_id in &self.default_fields {
                    bool_query = bool_query.should(TermQuery::text(field_id, token));
                }
            }
            Ok(Box::new(bool_query))
        } else {
            Err("No field specified and no default fields configured".to_string())
        }
    }

    fn build_phrase_query(
        &self,
        field: Option<&str>,
        phrase: &str,
    ) -> Result<Box<dyn Query>, String> {
        // For phrase queries, tokenize and create AND query of terms
        let field_id = if let Some(field_name) = field {
            self.schema
                .get_field(field_name)
                .ok_or_else(|| format!("Unknown field: {}", field_name))?
        } else if !self.default_fields.is_empty() {
            self.default_fields[0]
        } else {
            return Err("No field specified and no default fields configured".to_string());
        };

        let tokenizer = self.get_tokenizer(field_id);
        let tokens: Vec<String> = tokenizer
            .tokenize(phrase)
            .into_iter()
            .map(|t| t.text.to_lowercase())
            .collect();

        if tokens.is_empty() {
            return Err("No tokens in phrase".to_string());
        }

        if tokens.len() == 1 {
            return Ok(Box::new(TermQuery::text(field_id, &tokens[0])));
        }

        // Create AND query for all tokens (simplified phrase matching)
        let mut bool_query = BooleanQuery::new();
        for token in &tokens {
            bool_query = bool_query.must(TermQuery::text(field_id, token));
        }

        // If no field specified and multiple default fields, wrap in OR
        if field.is_none() && self.default_fields.len() > 1 {
            let mut outer = BooleanQuery::new();
            for &f in &self.default_fields {
                let tokenizer = self.get_tokenizer(f);
                let tokens: Vec<String> = tokenizer
                    .tokenize(phrase)
                    .into_iter()
                    .map(|t| t.text.to_lowercase())
                    .collect();

                let mut field_query = BooleanQuery::new();
                for token in &tokens {
                    field_query = field_query.must(TermQuery::text(f, token));
                }
                outer = outer.should(field_query);
            }
            return Ok(Box::new(outer));
        }

        Ok(Box::new(bool_query))
    }

    fn get_tokenizer(&self, field: Field) -> BoxedTokenizer {
        // Get tokenizer name from schema field entry, fallback to "lowercase"
        let tokenizer_name = self
            .schema
            .get_field_entry(field)
            .and_then(|entry| entry.tokenizer.as_deref())
            .unwrap_or("lowercase");

        self.tokenizers
            .get(tokenizer_name)
            .unwrap_or_else(|| Box::new(crate::tokenizer::LowercaseTokenizer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::SchemaBuilder;
    use crate::tokenizer::TokenizerRegistry;

    fn setup() -> (Arc<Schema>, Vec<Field>, Arc<TokenizerRegistry>) {
        let mut builder = SchemaBuilder::default();
        let title = builder.add_text_field("title", true, true);
        let body = builder.add_text_field("body", true, true);
        let schema = Arc::new(builder.build());
        let tokenizers = Arc::new(TokenizerRegistry::default());
        (schema, vec![title, body], tokenizers)
    }

    #[test]
    fn test_simple_term() {
        let (schema, default_fields, tokenizers) = setup();
        let parser = QueryLanguageParser::new(schema, default_fields, tokenizers);

        // Should parse without error - creates BooleanQuery across default fields
        let _query = parser.parse("rust").unwrap();
    }

    #[test]
    fn test_field_term() {
        let (schema, default_fields, tokenizers) = setup();
        let parser = QueryLanguageParser::new(schema, default_fields, tokenizers);

        // Should parse field:term syntax
        let _query = parser.parse("title:rust").unwrap();
    }

    #[test]
    fn test_boolean_and() {
        let (schema, default_fields, tokenizers) = setup();
        let parser = QueryLanguageParser::new(schema, default_fields, tokenizers);

        // Should parse AND boolean query
        let _query = parser.parse("rust AND programming").unwrap();
    }

    #[test]
    fn test_match_query() {
        let (schema, default_fields, tokenizers) = setup();
        let parser = QueryLanguageParser::new(schema, default_fields, tokenizers);

        // Should tokenize and create boolean query
        let _query = parser.parse("hello world").unwrap();
    }

    #[test]
    fn test_phrase_query() {
        let (schema, default_fields, tokenizers) = setup();
        let parser = QueryLanguageParser::new(schema, default_fields, tokenizers);

        // Should parse quoted phrase
        let _query = parser.parse("\"hello world\"").unwrap();
    }

    #[test]
    fn test_boolean_or() {
        let (schema, default_fields, tokenizers) = setup();
        let parser = QueryLanguageParser::new(schema, default_fields, tokenizers);

        // Should parse OR boolean query
        let _query = parser.parse("rust OR python").unwrap();
    }

    #[test]
    fn test_complex_query() {
        let (schema, default_fields, tokenizers) = setup();
        let parser = QueryLanguageParser::new(schema, default_fields, tokenizers);

        // Should parse complex boolean with grouping
        let _query = parser.parse("(rust OR python) AND programming").unwrap();
    }

    #[test]
    fn test_router_exclusive_mode() {
        use crate::dsl::query_field_router::{QueryFieldRouter, QueryRouterRule, RoutingMode};

        let mut builder = SchemaBuilder::default();
        let _title = builder.add_text_field("title", true, true);
        let _uri = builder.add_text_field("uri", true, true);
        let schema = Arc::new(builder.build());
        let tokenizers = Arc::new(TokenizerRegistry::default());

        let router = QueryFieldRouter::from_rules(&[QueryRouterRule {
            pattern: r"^doi:(10\.\d{4,}/[^\s]+)$".to_string(),
            substitution: "doi://{1}".to_string(),
            target_field: "uri".to_string(),
            mode: RoutingMode::Exclusive,
        }])
        .unwrap();

        let parser = QueryLanguageParser::with_router(schema, vec![], tokenizers, router);

        // Should route DOI query to uri field
        let _query = parser.parse("doi:10.1234/test.123").unwrap();
    }

    #[test]
    fn test_router_additional_mode() {
        use crate::dsl::query_field_router::{QueryFieldRouter, QueryRouterRule, RoutingMode};

        let mut builder = SchemaBuilder::default();
        let title = builder.add_text_field("title", true, true);
        let _uri = builder.add_text_field("uri", true, true);
        let schema = Arc::new(builder.build());
        let tokenizers = Arc::new(TokenizerRegistry::default());

        let router = QueryFieldRouter::from_rules(&[QueryRouterRule {
            pattern: r"#(\d+)".to_string(),
            substitution: "{1}".to_string(),
            target_field: "uri".to_string(),
            mode: RoutingMode::Additional,
        }])
        .unwrap();

        let parser = QueryLanguageParser::with_router(schema, vec![title], tokenizers, router);

        // Should route to both uri field and default fields
        let _query = parser.parse("#42").unwrap();
    }

    #[test]
    fn test_router_no_match_falls_through() {
        use crate::dsl::query_field_router::{QueryFieldRouter, QueryRouterRule, RoutingMode};

        let mut builder = SchemaBuilder::default();
        let title = builder.add_text_field("title", true, true);
        let _uri = builder.add_text_field("uri", true, true);
        let schema = Arc::new(builder.build());
        let tokenizers = Arc::new(TokenizerRegistry::default());

        let router = QueryFieldRouter::from_rules(&[QueryRouterRule {
            pattern: r"^doi:".to_string(),
            substitution: "{0}".to_string(),
            target_field: "uri".to_string(),
            mode: RoutingMode::Exclusive,
        }])
        .unwrap();

        let parser = QueryLanguageParser::with_router(schema, vec![title], tokenizers, router);

        // Should NOT match and fall through to normal parsing
        let _query = parser.parse("rust programming").unwrap();
    }

    #[test]
    fn test_router_invalid_target_field() {
        use crate::dsl::query_field_router::{QueryFieldRouter, QueryRouterRule, RoutingMode};

        let mut builder = SchemaBuilder::default();
        let _title = builder.add_text_field("title", true, true);
        let schema = Arc::new(builder.build());
        let tokenizers = Arc::new(TokenizerRegistry::default());

        let router = QueryFieldRouter::from_rules(&[QueryRouterRule {
            pattern: r"test".to_string(),
            substitution: "{0}".to_string(),
            target_field: "nonexistent".to_string(),
            mode: RoutingMode::Exclusive,
        }])
        .unwrap();

        let parser = QueryLanguageParser::with_router(schema, vec![], tokenizers, router);

        // Should fail because target field doesn't exist
        let result = parser.parse("test");
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.contains("Unknown target field"));
    }

    #[test]
    fn test_parse_ann_query() {
        let mut builder = SchemaBuilder::default();
        let embedding = builder.add_dense_vector_field("embedding", 128, true, true);
        let schema = Arc::new(builder.build());
        let tokenizers = Arc::new(TokenizerRegistry::default());

        let parser = QueryLanguageParser::new(schema, vec![embedding], tokenizers);

        // Parse ANN query
        let result = parser.parse_query_string("embedding:ann([1.0, 2.0, 3.0], nprobe=32)");
        assert!(result.is_ok(), "Failed to parse ANN query: {:?}", result);

        if let Ok(ParsedQuery::Ann {
            field,
            vector,
            nprobe,
            rerank,
        }) = result
        {
            assert_eq!(field, "embedding");
            assert_eq!(vector, vec![1.0, 2.0, 3.0]);
            assert_eq!(nprobe, 32);
            assert_eq!(rerank, 3.0); // default
        } else {
            panic!("Expected Ann query, got: {:?}", result);
        }
    }

    #[test]
    fn test_parse_sparse_query() {
        let mut builder = SchemaBuilder::default();
        let sparse = builder.add_text_field("sparse", true, true);
        let schema = Arc::new(builder.build());
        let tokenizers = Arc::new(TokenizerRegistry::default());

        let parser = QueryLanguageParser::new(schema, vec![sparse], tokenizers);

        // Parse sparse query
        let result = parser.parse_query_string("sparse:sparse({1: 0.5, 5: 0.3})");
        assert!(result.is_ok(), "Failed to parse sparse query: {:?}", result);

        if let Ok(ParsedQuery::Sparse { field, vector }) = result {
            assert_eq!(field, "sparse");
            assert_eq!(vector, vec![(1, 0.5), (5, 0.3)]);
        } else {
            panic!("Expected Sparse query, got: {:?}", result);
        }
    }
}
