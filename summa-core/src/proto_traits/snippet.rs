use summa_proto::proto;

use crate::proto_traits::Wrapper;

impl From<tantivy::Snippet> for Wrapper<proto::Snippet> {
    fn from(snippet: tantivy::Snippet) -> Self {
        Wrapper::from(proto::Snippet {
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
        })
    }
}
