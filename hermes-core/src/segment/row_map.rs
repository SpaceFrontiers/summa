//! One budgeted document permutation shared by every compaction encoder.
use crate::{Error, Result};

pub(crate) struct RowMap {
    old_to_new: Vec<u32>,
    pub(crate) new_to_old: Vec<u32>,
}

impl RowMap {
    pub(crate) fn new(
        num_docs: u32,
        max_live_docs: u32,
        keep: impl Fn(u32) -> bool,
        budget: usize,
    ) -> Result<Self> {
        if max_live_docs > num_docs {
            return Err(Error::Corruption(
                "live row count exceeds physical rows".into(),
            ));
        }
        let bytes = (num_docs as usize)
            .checked_add(max_live_docs as usize)
            .and_then(|rows| rows.checked_mul(4))
            .ok_or_else(|| Error::Schema("compaction row map size overflow".into()))?;
        if bytes > budget {
            return Err(Error::Schema(format!(
                "compaction row maps require up to {bytes} bytes; budget is {budget}"
            )));
        }
        let mut old_to_new = vec![u32::MAX; num_docs as usize];
        let mut new_to_old = Vec::with_capacity(max_live_docs as usize);
        for old in 0..num_docs {
            if keep(old) {
                if new_to_old.len() == max_live_docs as usize {
                    return Err(Error::Corruption(
                        "row map exceeds admitted live count".into(),
                    ));
                }
                old_to_new[old as usize] = new_to_old.len() as u32;
                new_to_old.push(old);
            }
        }
        Ok(Self {
            old_to_new,
            new_to_old,
        })
    }

    pub(crate) fn get(&self, old: u32) -> Option<u32> {
        self.old_to_new
            .get(old as usize)
            .copied()
            .filter(|&id| id != u32::MAX)
    }

    pub(crate) fn len(&self) -> u32 {
        self.new_to_old.len() as u32
    }

    pub(crate) fn memory_bytes(&self) -> usize {
        (self.old_to_new.capacity() + self.new_to_old.capacity()) * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mostly_deleted_rows_fit_the_live_map_budget_without_growing_it() {
        let rows = RowMap::new(64, 1, |doc| doc == 63, 260).unwrap();
        assert_eq!(rows.memory_bytes(), 260);
        assert_eq!(rows.get(63), Some(0));
        assert_eq!(rows.get(0), None);
        assert_eq!(rows.new_to_old, [63]);
        assert!(RowMap::new(64, 1, |_| true, 260).is_err());
        assert!(RowMap::new(64, 65, |_| true, 1024).is_err());
    }
}
