use fasteval2::{Compiler, Evaler, Instruction};
use tantivy::schema::{FieldType, Schema};
use tantivy::{DocId, SegmentReader};

use super::fast_field_iterator::{FastFieldIterator, FastFieldIteratorImpl};
use crate::errors::{Error, SummaResult, ValidationError};
use crate::page_rank::inverse_quantized_page_rank;

/// Responsible for evaluation `fasteval` formula against documents and fastfields to receive document score
pub(crate) struct SegmentEvalScorer {
    slab: fasteval2::Slab,
    compiled: Instruction,
    boxed_original_score: Box<f64>,
    _boxed_now: Box<f64>,
    fast_fields_iterators: Vec<Box<dyn FastFieldIterator>>,
    namespace: fn(&str, Vec<f64>) -> Option<f64>,
}

fn fast_field_to_iter(schema: &Schema, segment_reader: &SegmentReader, field_name: &str) -> SummaResult<Box<dyn FastFieldIterator>> {
    let field = schema.get_field(field_name)?;
    let field_type = schema.get_field_entry(field).field_type();
    let fast_field = match field_type {
        FieldType::U64(_) => FastFieldIteratorImpl::from_fast_field_reader(segment_reader.fast_fields().u64(field_name).map_err(|tantivy_error| {
            ValidationError::InvalidFastFieldType {
                field: field_name.to_owned(),
                field_type: field_type.to_owned(),
                tantivy_error,
            }
        })?),
        FieldType::I64(_) => FastFieldIteratorImpl::from_fast_field_reader(segment_reader.fast_fields().i64(field_name).map_err(|tantivy_error| {
            ValidationError::InvalidFastFieldType {
                field: field_name.to_owned(),
                field_type: field_type.to_owned(),
                tantivy_error,
            }
        })?),
        FieldType::F64(_) => FastFieldIteratorImpl::from_fast_field_reader(segment_reader.fast_fields().f64(field_name).map_err(|tantivy_error| {
            ValidationError::InvalidFastFieldType {
                field: field_name.to_owned(),
                field_type: field_type.to_owned(),
                tantivy_error,
            }
        })?),
        FieldType::Date(_) => FastFieldIteratorImpl::from_fast_field_reader(segment_reader.fast_fields().date(field_name).map_err(|tantivy_error| {
            ValidationError::InvalidFastFieldType {
                field: field_name.to_owned(),
                field_type: field_type.to_owned(),
                tantivy_error,
            }
        })?),
        field_type => return Err(Error::InvalidFieldType(field_name.to_owned(), field_type.to_owned())),
    };
    Ok(fast_field)
}

async fn fast_field_to_iter_async(schema: &Schema, segment_reader: &SegmentReader, field_name: &str) -> SummaResult<Box<dyn FastFieldIterator>> {
    let field = schema.get_field(field_name)?;
    let field_type = schema.get_field_entry(field).field_type();
    let fast_field = match field_type {
        FieldType::U64(_) => FastFieldIteratorImpl::from_fast_field_reader(segment_reader.fast_fields().u64_async(field_name).await?),
        FieldType::I64(_) => FastFieldIteratorImpl::from_fast_field_reader(segment_reader.fast_fields().i64_async(field_name).await?),
        FieldType::F64(_) => FastFieldIteratorImpl::from_fast_field_reader(segment_reader.fast_fields().f64_async(field_name).await?),
        FieldType::Date(_) => FastFieldIteratorImpl::from_fast_field_reader(segment_reader.fast_fields().date_async(field_name).await?),
        field_type => return Err(Error::InvalidFieldType(field_name.to_owned(), field_type.to_owned())),
    };
    Ok(fast_field)
}

impl SegmentEvalScorer {
    /// Create `SegmentEvalScorer` for segment
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
                "iqpr" => {
                    let x = args[0].abs() as u64;
                    Some(inverse_quantized_page_rank(x))
                }
                _ => None,
            }
        };

        let boxed_original_score = Box::new(0f64);
        let boxed_now = Box::new(instant::now() / 1000.0);

        // Set default variables
        unsafe {
            slab.ps.add_unsafe_var("original_score".to_owned(), boxed_original_score.as_ref());
            slab.ps.add_unsafe_var("now".to_owned(), boxed_now.as_ref());
        }

        let mut fast_fields_iterators = vec![];

        // Set fast fields
        for var_name in var_names {
            let fast_field_iterator = fast_field_to_iter(schema, segment_reader, var_name)?;
            unsafe {
                slab.ps.add_unsafe_var(var_name.to_owned(), fast_field_iterator.value());
            }
            fast_fields_iterators.push(fast_field_iterator);
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

    #[inline]
    pub async fn for_segment_async(
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
                "iqpr" => {
                    let x = args[0].abs() as u64;
                    Some(inverse_quantized_page_rank(x))
                }
                _ => None,
            }
        };

        let boxed_original_score = Box::new(0f64);
        let boxed_now = Box::new(instant::now() / 1000.0);

        // Set default variables
        unsafe {
            slab.ps.add_unsafe_var("original_score".to_owned(), boxed_original_score.as_ref());
            slab.ps.add_unsafe_var("now".to_owned(), boxed_now.as_ref());
        }

        let mut fast_fields_iterators = vec![];

        // Set fast fields
        for var_name in var_names {
            let fast_field_iterator = fast_field_to_iter_async(schema, segment_reader, var_name).await?;
            unsafe {
                slab.ps.add_unsafe_var(var_name.to_owned(), fast_field_iterator.value());
            }
            fast_fields_iterators.push(fast_field_iterator);
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

    pub(crate) fn score(&mut self, doc_id: DocId, original_score: f32) -> f64 {
        *self.boxed_original_score = original_score as f64;
        for fast_field_iterator in self.fast_fields_iterators.iter_mut() {
            fast_field_iterator.advance(doc_id)
        }
        if let fasteval2::IUnsafeVar { ptr, .. } = self.compiled {
            unsafe { *ptr }
        } else {
            self.compiled.eval(&self.slab, &mut self.namespace).expect("undefined variable")
        }
    }
}
