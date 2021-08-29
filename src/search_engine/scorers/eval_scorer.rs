use crate::errors::{Error, SummaResult, ValidationError};
use crate::search_engine::scorers::SegmentEvalScorer;
use fasteval2::Evaler;
use std::collections::HashSet;
use tantivy::schema::Schema;
use tantivy::SegmentReader;

lazy_static! {
    static ref RESERVED_WORDS: HashSet<&'static str> = HashSet::from(["now", "original_score", "fastsigm"]);
}

/// Instantiates `SegmentEvalScorer` for each segment
pub struct EvalScorer {
    eval_expr: String,
    schema: Schema,
    parser: fasteval2::Parser,
    var_names: Vec<String>,
}

impl EvalScorer {
    pub fn new(eval_expr: &str, schema: &Schema) -> SummaResult<EvalScorer> {
        let parser = fasteval2::Parser::new();

        // Create `Slab` for extracting variable names
        let mut slab = fasteval2::Slab::new();
        let parsed = parser.parse(eval_expr, &mut slab.ps)?.from(&slab.ps);
        let mut var_names = vec![];
        for var_name in parsed.var_names(&slab).iter().filter(|var_name| !RESERVED_WORDS.contains((*var_name).as_str())) {
            let field = schema.get_field(var_name);
            let field = field.ok_or_else(|| ValidationError::MissingField(var_name.to_owned()))?;
            if !schema.get_field_entry(field).is_fast() {
                return Err(Error::Validation(ValidationError::RequiredFastField(var_name.to_owned())));
            }
            var_names.push(var_name.to_owned());
        }

        Ok(EvalScorer {
            eval_expr: eval_expr.to_owned(),
            schema: schema.clone(),
            parser,
            var_names,
        })
    }

    pub fn get_for_segment_reader(&self, segment_reader: &SegmentReader) -> SummaResult<SegmentEvalScorer> {
        SegmentEvalScorer::for_segment(segment_reader, &self.schema, &self.parser, &self.eval_expr, &self.var_names)
    }
}
