//! Scaling probe: add_document cost versus document shape for a chunked
//! lex-tokenized text field. Run with --release --nocapture.

use std::sync::Arc;
use std::time::Instant;

use summa_core::dsl::sdl::parse_sdl;
use summa_core::tokenizer::TokenizerRegistry;
use summa_core::{Document, SegmentBuilder, SegmentBuilderConfig};

fn xorshift(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn shape(kind: &str, seed: &mut u64, words: usize) -> String {
    let mut out = String::with_capacity(words * 8);
    match kind {
        "english" => {
            for _ in 0..words {
                out.push_str(&format!("word{} ", xorshift(seed) % 5000));
            }
        }
        "unique" => {
            for _ in 0..words {
                out.push_str(&format!("u{:x} ", xorshift(seed)));
            }
        }
        "repeated" => {
            for _ in 0..words {
                out.push_str("same ");
            }
        }
        "longtoken" => {
            // one run of letters without any break, `words` * 6 bytes long
            for _ in 0..words {
                out.push_str("abcdef");
            }
        }
        "chinese" => {
            let chars: Vec<char> = "的一是不了人我在有他这为之大来以个中上们到说国和地也子时道出而要于就下得可你年生自会那后能对着事其里所去行过家十用发天如然作方成者多日都三小军二无同么经法当起与好看学进种将还分此心前面又定见只主没公从".chars().collect();
            for _ in 0..words {
                out.push(chars[(xorshift(seed) % chars.len() as u64) as usize]);
            }
        }
        "japanese" => {
            let parts = [
                "量子",
                "コンピュータ",
                "の",
                "研究",
                "を",
                "食べました",
                "学校",
                "東京",
                "で",
                "は",
            ];
            for _ in 0..words {
                out.push_str(parts[(xorshift(seed) % parts.len() as u64) as usize]);
            }
        }
        "numbers" => {
            for _ in 0..words {
                out.push_str(&format!("{},", xorshift(seed) % 100000));
            }
        }
        "mixed" => {
            for i in 0..words {
                if i % 3 == 0 {
                    out.push_str("Über-straße/2024 ");
                } else {
                    out.push_str(&format!("wörd{} ", xorshift(seed) % 3000));
                }
            }
        }
        _ => unreachable!(),
    }
    out
}

#[test]
#[ignore]
fn chunked_lex_add_document_scaling() {
    let sdl = r#"index t {
  field id: text<raw_ci> [indexed, stored, fast, primary]
  field languages: text<raw_ci> [indexed, stored, fast]
  field content: text<lex(by: languages, default: en, stop_words: true, han: simplified)> [indexed<chunked, token_position>]
}"#;
    let defs = parse_sdl(sdl).unwrap();
    let schema = Arc::new(defs[0].to_schema());
    let registry = TokenizerRegistry::new();
    let content = schema.get_field("content").unwrap();
    let id = schema.get_field("id").unwrap();
    let languages = schema.get_field("languages").unwrap();
    let kinds: Vec<&str> = std::env::var("SHAPES")
        .map(|s| s.split(',').map(|k| k.to_string()).collect::<Vec<_>>())
        .unwrap_or_else(|_| {
            vec![
                "english".into(),
                "unique".into(),
                "repeated".into(),
                "longtoken".into(),
                "chinese".into(),
                "japanese".into(),
                "numbers".into(),
                "mixed".into(),
            ]
        })
        .into_iter()
        .map(|k| Box::leak(k.into_boxed_str()) as &str)
        .collect();
    for kind in kinds {
        let single: bool = std::env::var("SINGLE").is_ok();
        for &chunks in &[500usize, 2000] {
            let (chunks, words_per_chunk) = if single {
                (1usize, 400 * chunks)
            } else {
                (chunks, 400)
            };
            let mut builder =
                SegmentBuilder::new(schema.clone(), SegmentBuilderConfig::default()).unwrap();
            for (field, entry) in schema.fields() {
                if let Some(ref name) = entry.tokenizer
                    && let Some(tok) = registry.get(name)
                {
                    builder.set_tokenizer(field, tok);
                }
            }
            let mut seed = 0x9e3779b97f4a7c15u64;
            let mut doc = Document::new();
            doc.add_text(id, format!("doc-{kind}-{chunks}"));
            doc.add_text(
                languages,
                if kind == "chinese" {
                    "zh"
                } else if kind == "japanese" {
                    "ja"
                } else {
                    "en"
                },
            );
            let mut bytes = 0;
            for _ in 0..chunks {
                let text = shape(kind, &mut seed, words_per_chunk);
                bytes += text.len();
                doc.add_text(content, text);
            }
            let started = Instant::now();
            builder.add_document(doc).unwrap();
            eprintln!(
                "{kind:10} chunks={chunks:5} words/chunk={words_per_chunk:7} bytes={:8} add_document={:?}",
                bytes,
                started.elapsed(),
            );
        }
    }
}
