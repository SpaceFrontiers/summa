use std::path::PathBuf;

use ahash::AHashMap;
use hermes_tokenizer::Tokenizer;
use tokenizers::models::bpe::BPE;
use tokenizers::normalizers::unicode::NFC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::{AddedToken, Tokenizer as HfTokenizer};

fn byte_to_unicode() -> [char; 256] {
    let allowed: Vec<u8> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut table = ['\0'; 256];
    for &byte in &allowed {
        table[byte as usize] = byte as char;
    }
    let mut offset = 0;
    for byte in 0..=255u8 {
        if table[byte as usize] == '\0' {
            table[byte as usize] = char::from_u32(256 + offset).unwrap();
            offset += 1;
        }
    }
    table
}

fn fixture() -> (String, HfTokenizer) {
    let byte_chars = byte_to_unicode();
    let mut vocab = AHashMap::new();
    vocab.insert("<eos>".to_owned(), 0);
    for (byte, character) in byte_chars.iter().enumerate() {
        vocab.insert(character.to_string(), byte as u32 + 1);
    }

    let merge_defs = [
        ("h", "e", "he"),
        ("he", "l", "hel"),
        ("hel", "l", "hell"),
        ("hell", "o", "hello"),
        ("Ġ", "hello", "Ġhello"),
        ("w", "o", "wo"),
        ("wo", "r", "wor"),
        ("wor", "l", "worl"),
        ("worl", "d", "world"),
        ("Ġ", "world", "Ġworld"),
        ("â", "Ģ", "âĢ"),
        ("âĢ", "Ļ", "âĢĻ"),
        ("âĢ", "Ķ", "âĢĶ"),
        ("âĢ", "¦", "âĢ¦"),
        ("Ġ", "âĢĶ", "ĠâĢĶ"),
    ];
    let mut merges = Vec::new();
    for (left, right, merged) in merge_defs {
        let id = vocab.len() as u32;
        vocab.insert(merged.to_owned(), id);
        merges.push((left.to_owned(), right.to_owned()));
    }

    let model = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .unwrap();
    let mut tokenizer = HfTokenizer::new(model);
    tokenizer.with_normalizer(Some(NFC)).unwrap();
    let byte_level = ByteLevel::new(false, true, true);
    tokenizer.with_pre_tokenizer(Some(byte_level));
    tokenizer.with_decoder(Some(byte_level));
    tokenizer
        .add_special_tokens([AddedToken::from("<eos>", true)])
        .unwrap();
    tokenizer
        .add_tokens([
            AddedToken::from("  ", false),
            AddedToken::from("<strip>", false).lstrip(true).rstrip(true),
        ])
        .unwrap();
    tokenizer.with_post_processor(Some(
        TemplateProcessing::builder()
            .try_single("<eos> $A <eos>")
            .unwrap()
            .special_tokens(vec![("<eos>", 0)])
            .build()
            .unwrap(),
    ));
    let json = tokenizer.to_string(false).unwrap();
    (json, tokenizer)
}

fn assert_text_parity(ours: &Tokenizer, hf: &HfTokenizer, text: &str) {
    for add_special_tokens in [false, true] {
        let actual = ours.encode(text, add_special_tokens).unwrap();
        let expected = hf.encode(text, add_special_tokens).unwrap();
        assert_eq!(
            actual,
            expected.get_ids(),
            "encoding differs for {text:?} (add_special_tokens={add_special_tokens})"
        );
        for skip_special_tokens in [false, true] {
            assert_eq!(
                ours.decode(&actual, skip_special_tokens).unwrap(),
                hf.decode(&actual, skip_special_tokens).unwrap(),
                "decoding differs for {text:?} (skip_special_tokens={skip_special_tokens})"
            );
        }
    }
}

#[test]
fn generated_byte_level_bpe_matches_hugging_face() {
    let (json, hf) = fixture();
    let ours = Tokenizer::from_bytes(json.as_bytes()).unwrap();
    let cases = [
        "",
        "hello",
        " hello world",
        "Hello, world!",
        "a   b\t\n",
        "a \t <strip> \n b",
        "<eos>",
        "before <eos> after",
        "caf\u{e9}",
        "cafe\u{301}",
        "Русский 中文 العربية",
        "emoji: 🦀🚀",
        "\0 control \u{85}",
    ];
    for text in cases {
        assert_text_parity(&ours, &hf, text);
    }

    assert_eq!(ours.vocab_size(), hf.get_vocab_size(true));
    for id in 0..ours.vocab_size() as u32 {
        assert_eq!(ours.id_to_token(id), hf.id_to_token(id), "token ID {id}");
    }
    for token in ["<eos>", "hello", "Ġhello", "  ", "<strip>", "🦀"] {
        assert_eq!(
            ours.token_to_id(token),
            hf.token_to_id(token),
            "token {token:?}"
        );
    }
    for (piece, display) in [("âĢĻ", "’"), ("ĠâĢĶ", " —"), ("âĢ¦", "…"), ("Ċ", "\n")]
    {
        let id = ours.token_to_id(piece).unwrap();
        assert_eq!(ours.decode(&[id], false).unwrap(), display);
        assert_eq!(ours.id_to_token(id).as_deref(), Some(piece));
    }
}

#[test]
fn parallel_batch_preserves_hugging_face_ids_and_order() {
    let (json, hf) = fixture();
    let ours = Tokenizer::from_bytes(json.as_bytes()).unwrap();
    let texts: Vec<String> = (0..32)
        .map(|index| format!("doc-{index}: hello world 🦀 ").repeat(1_600))
        .collect();
    assert!(texts.iter().map(String::len).sum::<usize>() > 1 << 20);

    let actual = ours.encode_batch(texts.clone(), false).unwrap();
    let expected = hf.encode_batch(texts, false).unwrap();
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual, expected.get_ids(), "batch document {index}");
    }
}

#[test]
fn unsupported_model_fails_at_load_time() {
    let error = Tokenizer::from_bytes(
        br#"{"model":{"type":"WordPiece","vocab":{"[UNK]":0},"unk_token":"[UNK]"}}"#,
    )
    .err()
    .expect("WordPiece must be rejected");
    assert!(error.to_string().contains("WordPiece"), "{error:#}");
}

#[test]
fn current_checkpoint_tokenizer_matches_hugging_face_when_available() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../.context/inference-step19000/tokenizer.json");
    if !path.exists() {
        eprintln!("checkpoint tokenizer not present; skipping local differential");
        return;
    }

    let ours = Tokenizer::from_file(&path).unwrap();
    let hf = HfTokenizer::from_file(&path).unwrap();
    let mut cases = vec![
        String::new(),
        "Summarize the retrieved evidence in three sentences.".to_owned(),
        "fn main() { println!(\"hello\"); }".to_owned(),
        "Русский 中文 العربية हिन्दी".to_owned(),
        "NFC café; NFD cafe\u{301}; emoji 🦀🚀".to_owned(),
        " \t  whitespace\n\n                        end".to_owned(),
        "literal <|endoftext|> and <|padding|>".to_owned(),
    ];
    let alphabet = [
        'a', 'Z', '0', ' ', '\n', '\t', '.', ',', 'é', '\u{301}', 'Ж', '中', 'ع', '🦀',
    ];
    let mut state = 0xD1FF_E2E3_91A5_77C3u64;
    for length in 0..128 {
        let mut text = String::new();
        for _ in 0..length {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            text.push(alphabet[(state as usize) % alphabet.len()]);
        }
        cases.push(text);
    }
    for text in &cases {
        assert_text_parity(&ours, &hf, text);
    }

    let actual = ours.encode_batch(cases.clone(), false).unwrap();
    let expected = hf.encode_batch(cases, false).unwrap();
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual,
            expected.get_ids(),
            "checkpoint batch document {index}"
        );
    }
}
