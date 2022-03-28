use fasteval2::Evaler;
use tantivy::schema::Schema;
use tantivy::SegmentReader;

use crate::errors::{SummaResult, ValidationError};
use crate::scorers::SegmentEvalScorer;

/// Instantiates `SegmentEvalScorer` for each segment
pub(crate) struct EvalScorer {
    eval_expr: String,
    schema: Schema,
    parser: fasteval2::Parser,
    var_names: Vec<String>,
}

impl EvalScorer {
    pub fn new(eval_expr: &str, schema: &Schema) -> SummaResult<EvalScorer> {
        static RESERVED_WORDS: [&str; 4] = ["now", "original_score", "fastsigm", "iqpr"];
        let parser = fasteval2::Parser::new();

        // Create `Slab` for extracting variable names
        let mut slab = fasteval2::Slab::new();
        let parsed = parser.parse(eval_expr, &mut slab.ps)?.from(&slab.ps);
        let mut var_names = vec![];
        for var_name in parsed
            .var_names(&slab)
            .iter()
            .filter(|var_name| !RESERVED_WORDS.contains(&(*var_name).as_str()))
        {
            let field = schema.get_field(var_name)?;
            if !schema.get_field_entry(field).is_fast() {
                return Err(ValidationError::RequiredFastField(var_name.to_owned()).into());
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

    /// Instantiates `SegmentEvalScorer` for passed segment
    pub fn get_for_segment_reader(&self, segment_reader: &SegmentReader) -> SummaResult<SegmentEvalScorer> {
        SegmentEvalScorer::for_segment(segment_reader, &self.schema, &self.parser, &self.eval_expr, &self.var_names)
    }

    /// Instantiates `SegmentEvalScorer` for passed segment in async way
    pub async fn get_for_segment_reader_async(&self, segment_reader: &SegmentReader) -> SummaResult<SegmentEvalScorer> {
        SegmentEvalScorer::for_segment_async(segment_reader, &self.schema, &self.parser, &self.eval_expr, &self.var_names).await
    }
}
