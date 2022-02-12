use std::collections::HashMap;

use futures::future::join_all;
use tantivy::query::Query;
use tantivy::Searcher;

pub struct SnippetGeneratorConfig {
    searcher: Searcher,
    query: Box<dyn Query>,
    snippet_configs: HashMap<String, u32>,
}

impl SnippetGeneratorConfig {
    pub fn new(searcher: Searcher, query: Box<dyn Query>, snippet_configs: HashMap<String, u32>) -> SnippetGeneratorConfig {
        SnippetGeneratorConfig {
            searcher,
            query,
            snippet_configs,
        }
    }

    pub fn as_tantivy(&self) -> Vec<(String, tantivy::SnippetGenerator)> {
        self.snippet_configs
            .iter()
            .filter_map(|(field_name, max_num_chars)| {
                self.searcher.schema().get_field(field_name).ok().map(|snippet_field| {
                    let mut snippet_generator =
                        tantivy::SnippetGenerator::create(&self.searcher, &*self.query, snippet_field).expect("Snippet generator cannot be created");
                    snippet_generator.set_max_num_chars(*max_num_chars as usize);
                    (field_name.to_string(), snippet_generator)
                })
            })
            .collect()
    }

    pub async fn as_tantivy_async(&self) -> Vec<(String, tantivy::SnippetGenerator)> {
        let futures = self.snippet_configs.iter().filter_map(|(field_name, max_num_chars)| {
            self.searcher.schema().get_field(field_name).ok().map(|snippet_field| async move {
                let mut snippet_generator = tantivy::SnippetGenerator::create_async(&self.searcher, &self.query, snippet_field)
                    .await
                    .expect("Snippet generator cannot be created");
                snippet_generator.set_max_num_chars(*max_num_chars as usize);
                (field_name.to_string(), snippet_generator)
            })
        });
        join_all(futures).await
    }
}
