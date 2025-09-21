//! Query language parser using pest
//!
//! Supports:
//! - Term queries: `rust` or `title:rust`
//! - Phrase queries: `"hello world"` or `title:"hello world"`
//! - Boolean operators: `AND`, `OR`, `NOT` (or `&&`, `||`, `!`)
//! - Required/prohibited clauses: `+rust -python`, including phrases and groups
//! - Grouping: `(rust OR python) AND programming`
//! - Default fields for unqualified terms

use pest::Parser;
use pest_derive::Parser;
use std::sync::Arc;

use super::query_field_router::{QueryFieldRouter, RoutingMode};
use super::schema::{Field, Schema};
use crate::query::{
    BooleanQuery, DEFAULT_DENSE_RERANK_FACTOR, PhraseQuery, PrefixQuery, Query, RegexQuery,
    TermQuery, WildcardQuery,
};
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
    /// Prefix query — matches terms starting with a given prefix
    Prefix {
        field: Option<String>,
        prefix: String,
    },
    /// Whole-term wildcard filter with explicit pattern syntax.
    Wildcard {
        field: Option<String>,
        pattern: String,
    },
    /// Whole-term regular expression; pattern case is preserved.
    Regex {
        field: Option<String>,
        pattern: String,
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
    Required(Box<ParsedQuery>),
    Prohibited(Box<ParsedQuery>),
    /// Preserve the scope of clause modifiers inside parentheses.
    Group(Box<ParsedQuery>),
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
        self.parse_with_mode(query_str, false)
    }

    /// Parse query syntax without falling back to tokenized plain text.
    /// Configured field routing still applies.
    pub fn parse_strict(&self, query_str: &str) -> Result<Box<dyn Query>, String> {
        self.parse_with_mode(query_str, true)
    }

    fn parse_with_mode(&self, query_str: &str, strict: bool) -> Result<Box<dyn Query>, String> {
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
                strict,
            );
        }

        // No routing match - parse normally
        self.parse_normal(query_str, strict)
    }

    /// Build a query from a routed match
    fn build_routed_query(
        &self,
        routed_query: &str,
        target_field: &str,
        mode: RoutingMode,
        original_query: &str,
        strict: bool,
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
                // An additional route must retain the original query or its
                // error; dropping it silently changes the requested semantics.
                bool_query = bool_query.should(self.parse_normal(original_query, strict)?);

                Ok(Box::new(bool_query))
            }
        }
    }

    /// Parse query without routing (normal parsing path)
    fn parse_normal(&self, query_str: &str, strict: bool) -> Result<Box<dyn Query>, String> {
        // Try parsing as query language first
        match self.parse_query_string(query_str) {
            Ok(parsed) => self.build_query(&parsed),
            Err(error)
                if strict
                    || query_str.contains('\\')
                    || query_str.match_indices("regex").any(|(at, _)| {
                        (at == 0
                            || query_str[..at].ends_with(|ch: char| {
                                ch.is_whitespace() || matches!(ch, ':' | '+' | '-' | '(')
                            }))
                            && query_str[at + "regex".len()..]
                                .trim_start()
                                .starts_with('(')
                    })
                    || query_str
                        .split(|c: char| c.is_whitespace() || c == '(')
                        .any(|word| word.len() > 1 && word.starts_with(['+', '-'])) =>
            {
                Err(error)
            }
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

        let mut bool_query = BooleanQuery::new();
        for &field_id in &self.default_fields {
            for token in self.get_tokenizer(field_id).tokenize(text) {
                bool_query =
                    bool_query.should(TermQuery::text(field_id, &token.text.to_lowercase()));
            }
        }
        if bool_query.should.is_empty() {
            return Err("No tokens in query".to_string());
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
        let mut modifier = None;
        let mut inner_query = None;

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::not_op | Rule::required_op | Rule::prohibited_op => {
                    modifier = Some(inner.as_rule());
                }
                Rule::group => {
                    let or_expr = inner.into_inner().next().unwrap();
                    inner_query = Some(ParsedQuery::Group(Box::new(self.parse_or_expr(or_expr)?)));
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
                Rule::wildcard_query | Rule::regex_query => {
                    let is_regex = inner.as_rule() == Rule::regex_query;
                    let mut field = None;
                    let mut pattern = String::new();
                    for part in inner.into_inner() {
                        match part.as_rule() {
                            Rule::field_spec => {
                                field = Some(part.into_inner().next().unwrap().as_str().to_owned())
                            }
                            Rule::quoted_string => {
                                pattern = serde_json::from_str(part.as_str())
                                    .map_err(|error| format!("Invalid pattern string: {error}"))?
                            }
                            _ => {}
                        }
                    }
                    inner_query = Some(if is_regex {
                        ParsedQuery::Regex { field, pattern }
                    } else {
                        ParsedQuery::Wildcard { field, pattern }
                    });
                }
                Rule::term_pattern_query => {
                    inner_query = Some(self.parse_term_pattern(inner)?);
                }
                Rule::term_query => {
                    inner_query = Some(self.parse_term_query(inner)?);
                }
                _ => {}
            }
        }

        let query = inner_query.ok_or("No query in primary")?;

        Ok(match modifier {
            Some(Rule::not_op) => ParsedQuery::Not(Box::new(query)),
            Some(Rule::required_op) => ParsedQuery::Required(Box::new(query)),
            Some(Rule::prohibited_op) => ParsedQuery::Prohibited(Box::new(query)),
            _ => query,
        })
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
                    let source = inner.as_str();
                    if source.contains('\\') {
                        term.reserve(source.len());
                        let mut chars = source.chars();
                        while let Some(ch) = chars.next() {
                            term.push(if ch == '\\' {
                                chars.next().ok_or("Dangling term escape")?
                            } else {
                                ch
                            });
                        }
                    } else {
                        term = source.to_owned();
                    }
                }
                _ => {}
            }
        }

        Ok(ParsedQuery::Term { field, term })
    }

    fn parse_term_pattern(&self, pair: pest::iterators::Pair<Rule>) -> Result<ParsedQuery, String> {
        let mut field = None;
        let mut prefix = String::new();

        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::field_spec => {
                    field = Some(inner.into_inner().next().unwrap().as_str().to_string());
                }
                Rule::term_pattern => {
                    prefix = inner.as_str().to_string();
                }
                _ => {}
            }
        }

        if let Some(literal) = prefix.strip_suffix('*')
            && !literal.is_empty()
            && !literal.contains(['*', '?', '\\'])
        {
            Ok(ParsedQuery::Prefix {
                field,
                prefix: literal.to_owned(),
            })
        } else {
            Ok(ParsedQuery::Wildcard {
                field,
                pattern: prefix,
            })
        }
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

    /// Parse an ANN query: field:ann([1.0, 2.0, 3.0], nprobe=32, rerank=2)
    fn parse_ann_query(&self, pair: pest::iterators::Pair<Rule>) -> Result<ParsedQuery, String> {
        let mut field = String::new();
        let mut vector = Vec::new();
        let mut nprobe = 32usize;
        let mut rerank = DEFAULT_DENSE_RERANK_FACTOR;

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
            ParsedQuery::Prefix { field, prefix } => {
                self.build_prefix_query(field.as_deref(), prefix)
            }
            ParsedQuery::Wildcard { field, pattern } | ParsedQuery::Regex { field, pattern } => {
                let explicit = field
                    .as_ref()
                    .map(|name| {
                        self.schema
                            .get_field(name)
                            .ok_or_else(|| format!("Unknown field: {name}"))
                    })
                    .transpose()?;
                let fields = explicit
                    .as_ref()
                    .map_or(self.default_fields.as_slice(), std::slice::from_ref);
                if fields.is_empty() {
                    return Err("No field specified and no default fields configured".into());
                }
                let build = |field| -> Result<Box<dyn Query>, String> {
                    if matches!(parsed, ParsedQuery::Regex { .. }) {
                        Ok(Box::new(
                            RegexQuery::new(field, pattern).map_err(|error| error.to_string())?,
                        ))
                    } else {
                        Ok(Box::new(
                            WildcardQuery::text(field, pattern)
                                .map_err(|error| error.to_string())?,
                        ))
                    }
                };
                if explicit.is_some() {
                    return build(fields[0]);
                }
                let mut query = BooleanQuery::new();
                for &field in fields {
                    query.should.push(Arc::from(build(field)?));
                }
                Ok(Box::new(query))
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
                let mut query = SparseVectorQuery::new(field_id, vector.clone());
                if let Some(config) = self
                    .schema
                    .get_field_entry(field_id)
                    .and_then(|entry| entry.sparse_vector_config.as_ref())
                    .and_then(|config| config.query_config.as_ref())
                {
                    if let Some(gamma) = config.lsp_gamma {
                        query = query.with_lsp_gamma(gamma);
                    }
                    query = query
                        .with_heap_factor(config.heap_factor)
                        .with_seismic_cut(config.seismic_cut)
                        .with_seismic_factor(config.seismic_factor)
                        .with_exhaustive(config.exhaustive);
                }
                Ok(Box::new(query))
            }
            ParsedQuery::And(queries) => {
                let mut bool_query = BooleanQuery::new();
                for q in queries {
                    bool_query = match q {
                        ParsedQuery::Prohibited(inner) => {
                            bool_query.must_not(self.build_query(inner)?)
                        }
                        _ => bool_query.must(self.build_query(q)?),
                    };
                }
                Ok(Box::new(bool_query))
            }
            ParsedQuery::Or(queries) => {
                let mut bool_query = BooleanQuery::new();
                for q in queries {
                    bool_query = match q {
                        ParsedQuery::Required(inner) => bool_query.must(self.build_query(inner)?),
                        ParsedQuery::Prohibited(inner) => {
                            bool_query.must_not(self.build_query(inner)?)
                        }
                        _ => bool_query.should(self.build_query(q)?),
                    };
                }
                Ok(Box::new(bool_query))
            }
            ParsedQuery::Required(inner) | ParsedQuery::Group(inner) => self.build_query(inner),
            ParsedQuery::Not(inner) | ParsedQuery::Prohibited(inner) => {
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
            // Each default field must use the same tokenizer as its indexed text.
            // Preserve the unqualified term's OR semantics across fields and tokens.
            let mut bool_query = BooleanQuery::new();
            for &field_id in &self.default_fields {
                let tokens = self.get_tokenizer(field_id).tokenize(term);

                // Preserve term decomposition for the owning planner, including
                // when nested in a larger Boolean query.
                if tokens.len() == 1 && self.default_fields.len() == 1 {
                    return Ok(Box::new(TermQuery::text(
                        field_id,
                        &tokens[0].text.to_lowercase(),
                    )));
                }

                for token in tokens {
                    bool_query =
                        bool_query.should(TermQuery::text(field_id, &token.text.to_lowercase()));
                }
            }
            // An empty token stream in one field must not discard other fields.
            if bool_query.should.is_empty() {
                return Err("No tokens in term".to_string());
            }
            Ok(Box::new(bool_query))
        } else {
            Err("No field specified and no default fields configured".to_string())
        }
    }

    fn build_prefix_query(
        &self,
        field: Option<&str>,
        prefix: &str,
    ) -> Result<Box<dyn Query>, String> {
        if let Some(field_name) = field {
            let field_id = self
                .schema
                .get_field(field_name)
                .ok_or_else(|| format!("Unknown field: {}", field_name))?;
            Ok(Box::new(PrefixQuery::text(field_id, prefix)))
        } else if !self.default_fields.is_empty() {
            // Unqualified prefix: OR across default fields
            let mut bool_query = BooleanQuery::new();
            for &field_id in &self.default_fields {
                bool_query = bool_query.should(PrefixQuery::text(field_id, prefix));
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
        let explicit = field
            .map(|name| {
                self.schema
                    .get_field(name)
                    .ok_or_else(|| format!("Unknown field: {name}"))
            })
            .transpose()?;
        let fields = explicit
            .as_ref()
            .map_or(self.default_fields.as_slice(), std::slice::from_ref);
        if fields.is_empty() {
            return Err("No field specified and no default fields configured".into());
        }
        let build = |field_id| -> Result<Option<Box<dyn Query>>, String> {
            let tokens: Vec<_> = self
                .get_tokenizer(field_id)
                .tokenize(phrase)
                .into_iter()
                .map(|token| (token.position, token.text.to_lowercase().into_bytes()))
                .collect();
            if tokens.is_empty() {
                return Ok(None);
            }
            if tokens.len() == 1 {
                return Ok(Some(Box::new(TermQuery::new(
                    field_id,
                    tokens[0].1.clone(),
                ))));
            }
            PhraseQuery::validate_positions(&self.schema, field_id)
                .map_err(|error| error.to_string())?;
            Ok(Some(Box::new(PhraseQuery::with_offsets(field_id, tokens))))
        };
        if fields.len() == 1 {
            return build(fields[0])?.ok_or_else(|| "No tokens in phrase".into());
        }
        let mut outer = BooleanQuery::new();
        let mut populated = false;
        for &field in fields {
            if let Some(query) = build(field)? {
                outer = outer.should(query);
                populated = true;
            }
        }
        if !populated {
            return Err("No tokens in phrase".into());
        }
        Ok(Box::new(outer))
    }

    fn get_tokenizer(&self, field: Field) -> BoxedTokenizer {
        // Get tokenizer name from schema field entry, fallback to "simple"
        let tokenizer_name = self
            .schema
            .get_field_entry(field)
            .and_then(|entry| entry.tokenizer.as_deref())
            .unwrap_or("simple");

        self.tokenizers
            .get(tokenizer_name)
            .unwrap_or_else(|| Box::new(crate::tokenizer::SimpleTokenizer))
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
        builder.set_positions(title, crate::dsl::PositionMode::TokenPosition);
        builder.set_positions(body, crate::dsl::PositionMode::TokenPosition);
        let schema = Arc::new(builder.build());
        let tokenizers = Arc::new(TokenizerRegistry::default());
        (schema, vec![title, body], tokenizers)
    }

    #[cfg(feature = "native")]
    #[tokio::test]
    async fn unqualified_search_uses_each_fields_tokenizer_in_either_order() {
        use crate::{Document, Index, IndexConfig, IndexWriter, RamDirectory};

        let cases = [
            (
                ["en: text<en_stem>", "zh: text<lex(segmenter: unicode)>"],
                vec![
                    [
                        "The quick brown fox jumps over the lazy dog",
                        "那只敏捷的棕毛狐狸跃过了那只懒狗",
                    ],
                    ["unrelated", "无关"],
                ],
                "棕毛狐狸",
                "zh:棕毛狐狸",
                vec![0],
            ),
            (
                ["raw: text<simple>", "stemmed: text<en_stem>"],
                vec![["running", ""], ["", "running"], ["unrelated", "unrelated"]],
                "running",
                "raw:running OR stemmed:running",
                vec![0, 1],
            ),
            (
                [
                    "filtered: text<lex(default: en, stop_words: true)>",
                    "raw: text<simple>",
                ],
                vec![["", "the"], ["unrelated", "unrelated"]],
                "the",
                "raw:the",
                vec![0],
            ),
        ];
        for (fields, documents, term, qualified, expected) in cases {
            for reverse in [false, true] {
                let order = if reverse { [1, 0] } else { [0, 1] };
                let schema = crate::parse_schema(&format!(
                    "index test {{ field {} [indexed, stored] field {} [indexed, stored] }}",
                    fields[order[0]], fields[order[1]],
                ))
                .unwrap();
                let field_ids =
                    fields.map(|field| schema.get_field(field.split(':').next().unwrap()).unwrap());
                let directory = RamDirectory::new();
                let config = IndexConfig::default();
                let mut writer = IndexWriter::create(directory.clone(), schema, config.clone())
                    .await
                    .unwrap();
                for values in &documents {
                    let mut document = Document::new();
                    for (field, value) in field_ids.iter().zip(values) {
                        if !value.is_empty() {
                            document.add_text(*field, *value);
                        }
                    }
                    writer.add_document(document).unwrap();
                }
                writer.commit().await.unwrap();
                let index = Index::open(directory, config).await.unwrap();
                let parser = index.query_parser();
                // A comma forces the permissive parser's plain-text fallback.
                let fallback = format!("{term},");
                assert!(parser.parse_query_string(&fallback).is_err());
                for text in [qualified, term, fallback.as_str()] {
                    let result = index.query(text, 10).await.unwrap();
                    let mut actual: Vec<_> =
                        result.hits.iter().map(|hit| hit.address.doc_id).collect();
                    actual.sort_unstable();
                    assert_eq!(actual, expected, "query={text}, reverse={reverse}");
                }
                let strict = parser.parse_strict(term).unwrap();
                let mut actual: Vec<_> = index
                    .search(strict.as_ref(), 10)
                    .await
                    .unwrap()
                    .hits
                    .iter()
                    .map(|hit| hit.address.doc_id)
                    .collect();
                actual.sort_unstable();
                assert_eq!(actual, expected, "strict query={term}, reverse={reverse}");
            }
        }
    }

    #[test]
    fn unqualified_search_rejects_queries_with_no_tokens_in_any_default_field() {
        let schema = Arc::new(
            crate::parse_schema(
                "index test {
                field first: text<en_stem_stop>
                field second: text<lex(default: en, stop_words: true)>
            }",
            )
            .unwrap(),
        );
        let fields = vec![
            schema.get_field("first").unwrap(),
            schema.get_field("second").unwrap(),
        ];
        let parser =
            QueryLanguageParser::new(schema, fields, Arc::new(TokenizerRegistry::default()));
        assert_eq!(parser.parse("the").err().unwrap(), "No tokens in term");
        assert_eq!(parser.parse("the,").err().unwrap(), "No tokens in query");
    }

    #[test]
    fn unqualified_single_field_term_preserves_planner_decomposition() {
        let (schema, fields, tokenizers) = setup();
        let parser = QueryLanguageParser::new(schema, vec![fields[0]], tokenizers);
        let query = parser.parse("RUNNING").unwrap();
        let crate::query::QueryDecomposition::TextTerm(info) = query.decompose() else {
            panic!("single-field term must remain decomposable");
        };
        assert_eq!(info.field, fields[0]);
        assert_eq!(info.term, b"running");
    }

    #[test]
    fn sparse_query_language_preserves_schema_lsp_gamma_including_exhaustive_zero() {
        for gamma in [None, Some(0), Some(7)] {
            let setting = gamma.map_or(String::new(), |gamma| {
                format!(", query<lsp_gamma: {gamma}, exhaustive: false>")
            });
            let schema = crate::parse_schema(&format!(
                "index test {{ field emb: sparse_vector [indexed<format: bmp, dims: 16{setting}>] }}"
            )).unwrap();
            let parser = QueryLanguageParser::new(
                Arc::new(schema),
                vec![],
                Arc::new(TokenizerRegistry::default()),
            );
            let query = parser.parse("emb:sparse({0: 1.0})").unwrap();
            let crate::query::QueryDecomposition::SparseTerms(infos) = query.decompose() else {
                panic!("sparse syntax did not produce a sparse query");
            };
            assert_eq!(infos.len(), 1);
            assert_eq!(infos[0].lsp_gamma, gamma);
            assert!(!infos[0].exhaustive);
        }
    }

    #[cfg(feature = "native")]
    #[tokio::test]
    async fn parsed_sparse_search_uses_the_fields_exhaustive_or_bounded_lsp_policy() {
        use crate::query::SparseVectorQuery;
        use crate::{Document, Index, IndexConfig, IndexWriter, RamDirectory};
        for gamma in [0, 1] {
            let schema = crate::parse_schema(&format!(
                "index test {{ field emb: sparse_vector [indexed<format: bmp, dims: 16, bmp_block_size: 1, query<lsp_gamma: {gamma}>>] }}"
            )).unwrap();
            let field = schema.get_field("emb").unwrap();
            let directory = RamDirectory::new();
            let config = IndexConfig::default();
            let mut writer = IndexWriter::create(directory.clone(), schema, config.clone())
                .await
                .unwrap();
            // Two superblocks: one of eight weaker documents and one winner.
            for doc in 0..9 {
                let mut document = Document::new();
                document.add_sparse_vector(field, vec![(0, if doc == 8 { 5.0 } else { 0.1 })]);
                writer.add_document(document).unwrap();
            }
            writer.commit().await.unwrap();
            let index = Index::open(directory, config).await.unwrap();
            let query = index.query_parser().parse("emb:sparse({0: 1.0})").unwrap();
            let actual = index.search(query.as_ref(), 9).await.unwrap();
            let expected = index
                .search(
                    &SparseVectorQuery::new(field, vec![(0, 1.0)]).with_lsp_gamma(gamma),
                    9,
                )
                .await
                .unwrap();
            let hits = |result: crate::query::SearchResponse| {
                result
                    .hits
                    .into_iter()
                    .map(|hit| (hit.address.doc_id, hit.score.to_bits()))
                    .collect::<Vec<_>>()
            };
            assert_eq!(actual.hits.len(), if gamma == 0 { 9 } else { 1 });
            assert_eq!(hits(actual), hits(expected));
        }
    }

    #[test]
    fn sparse_query_language_preserves_schema_exhaustive_policy() {
        for exhaustive in [false, true] {
            let setting = format!(", query<exhaustive: {exhaustive}>");
            let schema = crate::parse_schema(&format!(
                "index test {{ field emb: sparse_vector [indexed<format: seismic, dims: 16{setting}>] }}"
            )).unwrap();
            let parser = QueryLanguageParser::new(
                Arc::new(schema),
                vec![],
                Arc::new(TokenizerRegistry::default()),
            );
            let query = parser.parse("emb:sparse({0: 1.0})").unwrap();
            let crate::query::QueryDecomposition::SparseTerms(infos) = query.decompose() else {
                panic!("sparse syntax did not produce a sparse query");
            };
            assert_eq!(infos.len(), 1);
            assert_eq!(infos[0].exhaustive, exhaustive);
        }
    }

    #[cfg(feature = "native")]
    #[tokio::test]
    async fn parsed_sparse_search_uses_the_fields_exhaustive_policy() {
        use crate::query::SparseVectorQuery;
        use crate::{Document, Index, IndexConfig, IndexWriter, RamDirectory};
        for exhaustive in [false, true] {
            let schema = crate::parse_schema(&format!(
                "index test {{ field emb: sparse_vector [indexed<format: seismic, dims: 16, seismic_postings: 1, seismic_cluster_size: 1, query<exhaustive: {exhaustive}>>] }}"
            )).unwrap();
            let field = schema.get_field("emb").unwrap();
            let directory = RamDirectory::new();
            let config = IndexConfig::default();
            let mut writer = IndexWriter::create(directory.clone(), schema, config.clone())
                .await
                .unwrap();
            // Two superblocks: one of eight weaker documents and one winner.
            for doc in 0..9 {
                let mut document = Document::new();
                document.add_sparse_vector(field, vec![(0, if doc == 8 { 5.0 } else { 0.1 })]);
                writer.add_document(document).unwrap();
            }
            writer.commit().await.unwrap();
            let index = Index::open(directory, config).await.unwrap();
            let query = index.query_parser().parse("emb:sparse({0: 1.0})").unwrap();
            let actual = index.search(query.as_ref(), 9).await.unwrap();
            let expected = index
                .search(
                    &SparseVectorQuery::new(field, vec![(0, 1.0)]).with_exhaustive(exhaustive),
                    9,
                )
                .await
                .unwrap();
            let hits = |result: crate::query::SearchResponse| {
                result
                    .hits
                    .into_iter()
                    .map(|hit| (hit.address.doc_id, hit.score.to_bits()))
                    .collect::<Vec<_>>()
            };
            assert_eq!(actual.hits.len(), if exhaustive { 9 } else { 1 });
            assert_eq!(hits(actual), hits(expected));
        }
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
            assert_eq!(rerank, 2.0); // default
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

    #[test]
    fn test_parse_prefix_simple() {
        let (schema, default_fields, tokenizers) = setup();
        let parser = QueryLanguageParser::new(schema, default_fields, tokenizers);

        // Simple prefix: title:abc*
        let result = parser.parse_query_string("title:abc*");
        assert!(result.is_ok(), "Failed to parse prefix query: {:?}", result);
        if let Ok(ParsedQuery::Prefix { field, prefix }) = result {
            assert_eq!(field, Some("title".to_string()));
            assert_eq!(prefix, "abc");
        } else {
            panic!("Expected Prefix query, got: {:?}", result);
        }
    }

    #[test]
    fn test_parse_prefix_url() {
        let mut builder = SchemaBuilder::default();
        let _site = builder.add_text_field("site", true, true);
        let schema = Arc::new(builder.build());
        let tokenizers = Arc::new(TokenizerRegistry::default());
        let parser = QueryLanguageParser::new(schema, vec![], tokenizers);

        // URL prefix: site:https://reddit.com/r/Transhumanism*
        let result = parser.parse_query_string("site:https://reddit.com/r/Transhumanism*");
        assert!(
            result.is_ok(),
            "Failed to parse URL prefix query: {:?}",
            result
        );
        if let Ok(ParsedQuery::Prefix { field, prefix }) = result {
            assert_eq!(field, Some("site".to_string()));
            assert_eq!(prefix, "https://reddit.com/r/Transhumanism");
        } else {
            panic!("Expected Prefix query, got: {:?}", result);
        }
    }

    #[test]
    fn test_parse_prefix_unqualified() {
        let (schema, default_fields, tokenizers) = setup();
        let parser = QueryLanguageParser::new(schema, default_fields, tokenizers);

        // Unqualified prefix: transhuman*
        let result = parser.parse_query_string("transhuman*");
        assert!(
            result.is_ok(),
            "Failed to parse unqualified prefix: {:?}",
            result
        );
        if let Ok(ParsedQuery::Prefix { field, prefix }) = result {
            assert_eq!(field, None);
            assert_eq!(prefix, "transhuman");
        } else {
            panic!("Expected Prefix query, got: {:?}", result);
        }
    }

    #[test]
    fn test_prefix_query_builds() {
        let (schema, default_fields, tokenizers) = setup();
        let parser = QueryLanguageParser::new(schema, default_fields, tokenizers);

        // Should build without error
        let _query = parser.parse("title:abc*").unwrap();
    }

    #[test]
    fn test_prefix_in_boolean() {
        let (schema, default_fields, tokenizers) = setup();
        let parser = QueryLanguageParser::new(schema, default_fields, tokenizers);

        // Prefix in boolean: rust AND title:abc*
        let _query = parser.parse("rust AND title:abc*").unwrap();
    }

    #[test]
    fn test_prefix_mixed_with_terms() {
        let mut builder = SchemaBuilder::default();
        let title = builder.add_text_field("title", true, true);
        let _site = builder.add_text_field("site", true, true);
        let schema = Arc::new(builder.build());
        let tokenizers = Arc::new(TokenizerRegistry::default());
        let parser = QueryLanguageParser::new(schema, vec![title], tokenizers);

        // Mixed: prefix + free-text terms (implicit OR)
        let result =
            parser.parse_query_string("site:https://reddit.com/r/Transhumanism* longevity drugs");
        assert!(
            result.is_ok(),
            "Failed to parse mixed prefix+terms: {:?}",
            result
        );
        // Should be Or([Prefix, Term, Term])
        if let Ok(ParsedQuery::Or(parts)) = &result {
            assert_eq!(parts.len(), 3, "Expected 3 parts, got: {:?}", parts);
            assert!(
                matches!(&parts[0], ParsedQuery::And(v) if v.len() == 1 && matches!(&v[0], ParsedQuery::Prefix { .. }))
                    || matches!(&parts[0], ParsedQuery::Prefix { .. }),
                "First part should be prefix: {:?}",
                parts[0]
            );
        } else {
            panic!("Expected Or query, got: {:?}", result);
        }

        // Should also build into a Query without error
        let _query = parser
            .parse("site:https://reddit.com/r/Transhumanism* longevity drugs")
            .unwrap();
    }

    #[test]
    fn boolean_keywords_do_not_split_longer_terms() {
        let (schema, default_fields, tokenizers) = setup();
        let parser = QueryLanguageParser::new(schema, default_fields, tokenizers);
        for word in ["NOTHING", "NOT_name", "NOT-thing", "NOT42", "NOTé"] {
            let parsed = parser.parse_query_string(word).unwrap();
            assert!(
                matches!(&parsed, ParsedQuery::Term { term, .. } if term == word),
                "{word}: {parsed:?}"
            );
        }
        for word in ["ANDROID", "ORCHID", "AND_name", "OR-thing", "AND42", "ORé"] {
            let parsed = parser.parse_query_string(&format!("alpha {word}")).unwrap();
            let ParsedQuery::Or(parts) = &parsed else {
                panic!("{word}: {parsed:?}");
            };
            assert_eq!(parts.len(), 2, "{word}: {parsed:?}");
            assert!(
                matches!(&parts[1], ParsedQuery::Term { term, .. } if term == word),
                "{word}: {parsed:?}"
            );
        }
        assert!(
            matches!(parser.parse_query_string("NOT:term").unwrap(), ParsedQuery::Term { field: Some(field), term } if field == "NOT" && term == "term")
        );
        for prefix in ["NOT", "NOT/path", "NOT@example", "NOT.thing"] {
            assert!(
                matches!(parser.parse_query_string(&format!("{prefix}*")).unwrap(), ParsedQuery::Prefix { prefix: value, .. } if value == prefix)
            );
        }
        assert!(matches!(
            parser.parse_query_string("alpha AND(beta)").unwrap(),
            ParsedQuery::And(_)
        ));
        assert!(matches!(
            parser.parse_query_string("NOT(alpha)").unwrap(),
            ParsedQuery::Not(_)
        ));
        assert!(matches!(
            parser.parse_query_string("alpha OR(beta)").unwrap(),
            ParsedQuery::Or(_)
        ));
    }

    #[test]
    fn test_implicit_or_plain_terms() {
        let (schema, default_fields, tokenizers) = setup();
        let parser = QueryLanguageParser::new(schema, default_fields, tokenizers);

        // Space-separated terms: implicit OR
        let result = parser.parse_query_string("hello world");
        assert!(result.is_ok(), "Failed to parse implicit OR: {:?}", result);
        if let Ok(ParsedQuery::Or(parts)) = &result {
            assert_eq!(parts.len(), 2);
        } else {
            panic!("Expected Or query, got: {:?}", result);
        }
    }
}
