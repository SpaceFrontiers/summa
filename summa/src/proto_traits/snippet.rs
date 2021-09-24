//! `From` trait implmentations used for creation of `proto::Score`

use crate::proto;

impl From<tantivy::Snippet> for proto::Snippet {
    fn from(snippet: tantivy::Snippet) -> Self {
        proto::Snippet {
            fragment: snippet.fragment().as_bytes().to_vec(),
            highlights: snippet
                .highlighted()
                .iter()
                .map(|r| proto::Highlight {
                    from: r.start as u32,
                    to: r.end as u32,
                })
                .collect(),
            html: snippet.to_html(),
        }
    }
}
