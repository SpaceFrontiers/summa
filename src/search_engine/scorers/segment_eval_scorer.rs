use super::fast_field_iterator::{FastFieldIterator, FastFieldIteratorImpl};
use crate::errors::{Error, SummaResult, ValidationError};
use fasteval2::{Compiler, Evaler, Instruction};
use std::time::{SystemTime, UNIX_EPOCH};
use tantivy::schema::{FieldType, Schema};
use tantivy::{DocId, Score, SegmentReader};

pub struct SegmentEvalScorer {
    slab: fasteval2::Slab,
    compiled: Instruction,
    boxed_original_score: Box<f64>,
    _boxed_now: Box<f64>,
    fast_fields_iterators: Vec<Box<dyn FastFieldIterator>>,
    namespace: fn(&str, Vec<f64>) -> Option<f64>,
}

fn fast_field_to_iter(schema: &Schema, segment_reader: &SegmentReader, field_name: &str) -> SummaResult<Box<dyn FastFieldIterator>> {
    let field = schema.get_field(field_name).ok_or_else(|| Error::FieldDoesNotExist(field_name.to_owned()))?;
    let field_type = schema.get_field_entry(field).field_type();
    let fast_field = match field_type {
        FieldType::U64(_) => {
            FastFieldIteratorImpl::from_fast_field_reader(segment_reader.fast_fields().u64(field).map_err(|_| ValidationError::InvalidFastFieldType {
                field: field_name.to_owned(),
                field_type: field_type.to_owned(),
            })?)
        }
        FieldType::I64(_) => {
            FastFieldIteratorImpl::from_fast_field_reader(segment_reader.fast_fields().i64(field).map_err(|_| ValidationError::InvalidFastFieldType {
                field: field_name.to_owned(),
                field_type: field_type.to_owned(),
            })?)
        }
        FieldType::F64(_) => {
            FastFieldIteratorImpl::from_fast_field_reader(segment_reader.fast_fields().f64(field).map_err(|_| ValidationError::InvalidFastFieldType {
                field: field_name.to_owned(),
                field_type: field_type.to_owned(),
            })?)
        }
        FieldType::Date(_) => {
            FastFieldIteratorImpl::from_fast_field_reader(segment_reader.fast_fields().date(field).map_err(|_| ValidationError::InvalidFastFieldType {
                field: field_name.to_owned(),
                field_type: field_type.to_owned(),
            })?)
        }
        field_type => return Err(Error::InvalidFieldType(field_name.to_owned(), field_type.to_owned())),
    };
    Ok(fast_field)
}

impl SegmentEvalScorer {
    #[inline]
    pub fn for_segment(
        segment_reader: &SegmentReader,
        schema: &Schema,
        parser: &fasteval2::Parser,
        eval_expr: &str,
        var_names: &Vec<String>,
    ) -> SummaResult<SegmentEvalScorer> {
        let mut slab = fasteval2::Slab::new();

        let mut namespace = |name: &str, args: Vec<f64>| -> Option<f64> {
            match name {
                "fastsigm" => {
                    let x = args[0].abs();
                    let a = args.get(1).unwrap_or(&1f64);
                    Some(x / (*a + x))
                }
                _ => None,
            }
        };

        let boxed_original_score = Box::new(0f64);
        let boxed_now = Box::new(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64());

        // Set default variables
        unsafe {
            slab.ps.add_unsafe_var("original_score".to_owned(), boxed_original_score.as_ref());
            slab.ps.add_unsafe_var("now".to_owned(), boxed_now.as_ref());
        }

        let mut fast_fields_iterators = vec![];

        // Set fast fields
        for var_name in var_names {
            fast_fields_iterators.push(fast_field_to_iter(schema, segment_reader, var_name)?);
            unsafe {
                slab.ps.add_unsafe_var(var_name.to_owned(), fast_fields_iterators.last().unwrap().value());
            }
        }
        let compiled = parser
            .parse(eval_expr, &mut slab.ps)?
            .from(&slab.ps)
            .compile(&slab.ps, &mut slab.cs, &mut namespace);

        Ok(SegmentEvalScorer {
            slab,
            compiled,
            boxed_original_score,
            _boxed_now: boxed_now,
            fast_fields_iterators,
            namespace,
        })
    }
    pub(crate) fn score(&mut self, doc_id: DocId, original_score: Score) -> f64 {
        *self.boxed_original_score = original_score as f64;
        for fast_field_iterator in self.fast_fields_iterators.iter_mut() {
            fast_field_iterator.advance(doc_id)
        }
        if let fasteval2::IUnsafeVar { ptr, .. } = self.compiled {
            unsafe { *ptr }
        } else {
            self.compiled.eval(&self.slab, &mut self.namespace).unwrap()
        }
    }
}
