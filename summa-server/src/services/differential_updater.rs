use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ipfs_api::response::FilesEntry;
use itertools::Itertools;
use summa_core::components::ComponentFile;

#[derive(Debug)]
pub enum RequiredOperation {
    Add(ComponentFile),
    Remove(FilesEntry),
}

impl RequiredOperation {
    pub fn path(&self) -> &Path {
        match self {
            RequiredOperation::Add(component_file) => component_file.path(),
            RequiredOperation::Remove(files_entry) => Path::new(&files_entry.name),
        }
    }
}

impl PartialEq for RequiredOperation {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (RequiredOperation::Add(a), RequiredOperation::Add(b)) => a.path().eq(b.path()),
            (RequiredOperation::Remove(a), RequiredOperation::Remove(b)) => a.name.eq(&b.name),
            _ => false,
        }
    }
}

pub struct DifferentialUpdater {
    source_files: Vec<FilesEntry>,
}

impl DifferentialUpdater {
    pub fn from_source(source: impl IntoIterator<Item = FilesEntry>) -> DifferentialUpdater {
        DifferentialUpdater {
            source_files: source.into_iter().collect(),
        }
    }

    pub fn target_state(self, target_files: impl IntoIterator<Item = ComponentFile>) -> Vec<RequiredOperation> {
        let mut target_files: HashMap<PathBuf, ComponentFile> =
            HashMap::from_iter(target_files.into_iter().map(|index_file| (index_file.path().to_path_buf(), index_file)));
        let mut required_operations = vec![];
        for source_file in self.source_files {
            match target_files.remove(Path::new(&source_file.name)) {
                None => required_operations.push(RequiredOperation::Remove(source_file)),
                Some(index_file) => match index_file {
                    ComponentFile::SegmentComponent(_) => {}
                    ComponentFile::Other(_) => required_operations.push(RequiredOperation::Add(index_file.clone())),
                },
            }
        }
        required_operations.extend(target_files.into_values().map(|index_file| RequiredOperation::Add(index_file)));
        required_operations.into_iter().sorted_by_cached_key(|f| f.path().to_path_buf()).collect()
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ipfs_api::response::FilesEntry;
    use summa_core::components::{ComponentFile, SegmentComponent};

    use crate::services::differential_updater::RequiredOperation;
    use crate::services::DifferentialUpdater;

    #[test]
    fn test_new_files() {
        let old_files = [
            FilesEntry {
                name: "0.pos".to_string(),
                typ: 0,
                size: 17,
                hash: "deadbeef11".to_string(),
            },
            FilesEntry {
                name: "1.pos".to_string(),
                typ: 0,
                size: 23,
                hash: "deadbeef12".to_string(),
            },
            FilesEntry {
                name: ".managed.json".to_string(),
                typ: 0,
                size: 31,
                hash: "deadbeef13".to_string(),
            },
            FilesEntry {
                name: "meta.json".to_string(),
                typ: 0,
                size: 61,
                hash: "deadbeef14".to_string(),
            },
        ];
        let new_files = [
            ComponentFile::SegmentComponent(SegmentComponent {
                path: PathBuf::from("1.pos"),
                segment_component: tantivy::SegmentComponent::Postings,
            }),
            ComponentFile::SegmentComponent(SegmentComponent {
                path: PathBuf::from("2.pos"),
                segment_component: tantivy::SegmentComponent::Postings,
            }),
            ComponentFile::Other(PathBuf::from(".managed.json")),
            ComponentFile::Other(PathBuf::from("meta.json")),
            ComponentFile::Other(PathBuf::from("hotcache.bin")),
        ];
        assert_eq!(
            DifferentialUpdater::from_source(old_files).target_state(new_files),
            vec![
                RequiredOperation::Add(ComponentFile::Other(PathBuf::from(".managed.json"))),
                RequiredOperation::Remove(FilesEntry {
                    name: "0.pos".to_string(),
                    typ: 0,
                    size: 17,
                    hash: "deadbeef11".to_string(),
                }),
                RequiredOperation::Add(ComponentFile::SegmentComponent(SegmentComponent {
                    path: PathBuf::from("2.pos"),
                    segment_component: tantivy::SegmentComponent::Postings
                })),
                RequiredOperation::Add(ComponentFile::Other(PathBuf::from("hotcache.bin"))),
                RequiredOperation::Add(ComponentFile::Other(PathBuf::from("meta.json"))),
            ]
        )
    }
}
