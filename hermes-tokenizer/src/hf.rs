//! Loader for Hugging Face `tokenizer.json` byte-level BPE artifacts.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use rustc_hash::FxBuildHasher;
use serde::Deserialize;
use serde_json::Value;

use crate::bpe;
use crate::pretokenize::PretokenizerType;
use crate::token::TokenId;

pub(crate) struct LoadedTokenizer {
    pub engine: bpe::tiktoken::Tokenizer,
    pub pieces: Vec<Option<String>>,
    pub token_to_id: HashMap<String, u32>,
    pub special_ids: HashSet<u32>,
    pub prefix_ids: Vec<u32>,
    pub suffix_ids: Vec<u32>,
}

#[derive(Deserialize)]
struct TokenizerJson {
    model: Model,
    #[serde(default)]
    added_tokens: Vec<AddedToken>,
    #[serde(default)]
    pre_tokenizer: Option<PreTokenizerJson>,
    #[serde(default)]
    normalizer: Option<NormalizerJson>,
    #[serde(default)]
    post_processor: Option<Value>,
    #[serde(default)]
    decoder: Option<ComponentJson>,
}

#[derive(Deserialize)]
struct Model {
    #[serde(rename = "type")]
    model_type: String,
    vocab: HashMap<String, u32>,
    merges: Vec<[String; 2]>,
    #[serde(default)]
    dropout: Option<f32>,
    #[serde(default)]
    unk_token: Option<String>,
    #[serde(default)]
    continuing_subword_prefix: Option<String>,
    #[serde(default)]
    end_of_word_suffix: Option<String>,
    #[serde(default)]
    fuse_unk: bool,
    #[serde(default)]
    byte_fallback: bool,
    #[serde(default)]
    ignore_merges: bool,
}

#[derive(Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
    #[serde(default)]
    special: bool,
    #[serde(default)]
    single_word: bool,
    #[serde(default = "default_true")]
    normalized: bool,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    rstrip: bool,
}

#[derive(Deserialize)]
struct NormalizerJson {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    normalizers: Vec<NormalizerJson>,
}

#[derive(Deserialize)]
struct PreTokenizerJson {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    pretokenizers: Vec<PreTokenizerJson>,
    #[serde(default)]
    pattern: Option<PatternJson>,
    #[serde(default)]
    add_prefix_space: Option<bool>,
    #[serde(default)]
    use_regex: Option<bool>,
}

#[derive(Deserialize)]
struct ComponentJson {
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Deserialize)]
struct PatternJson {
    #[serde(rename = "Regex", default)]
    regex: Option<String>,
}

#[derive(Deserialize)]
struct ModelTypeProbe {
    #[serde(default)]
    model: Option<ModelTypeOnly>,
}

#[derive(Deserialize)]
struct ModelTypeOnly {
    #[serde(rename = "type")]
    model_type: Option<String>,
}

fn default_true() -> bool {
    true
}

pub(crate) fn load_file(path: impl AsRef<Path>) -> Result<LoadedTokenizer> {
    let path = path.as_ref();
    let data = std::fs::read(path)
        .with_context(|| format!("failed to read tokenizer {}", path.display()))?;
    load_slice(&data).with_context(|| format!("failed to parse tokenizer {}", path.display()))
}

pub(crate) fn load_slice(data: &[u8]) -> Result<LoadedTokenizer> {
    reject_unsupported_model(data)?;
    let tokenizer: TokenizerJson = serde_json::from_slice(data)
        .map_err(|error| anyhow::anyhow!("failed to parse tokenizer JSON: {error}"))?;
    build(tokenizer)
}

fn reject_unsupported_model(data: &[u8]) -> Result<()> {
    let Ok(ModelTypeProbe { model: Some(model) }) = serde_json::from_slice::<ModelTypeProbe>(data)
    else {
        return Ok(());
    };
    let family = model.model_type.filter(|model_type| model_type != "BPE");
    ensure!(
        family.is_none(),
        "unsupported tokenizer model type {}; hermes-tokenizer supports byte-level BPE",
        family.unwrap_or_default()
    );
    Ok(())
}

fn build(tokenizer: TokenizerJson) -> Result<LoadedTokenizer> {
    ensure!(
        tokenizer.model.model_type == "BPE",
        "unsupported tokenizer model type {} (expected BPE)",
        tokenizer.model.model_type
    );
    ensure!(
        tokenizer.model.dropout.is_none(),
        "BPE dropout is not supported"
    );
    ensure!(
        tokenizer.model.unk_token.is_none(),
        "BPE unk_token is not supported"
    );
    ensure!(
        tokenizer.model.continuing_subword_prefix.is_none(),
        "BPE continuing_subword_prefix is not supported"
    );
    ensure!(
        tokenizer.model.end_of_word_suffix.is_none(),
        "BPE end_of_word_suffix is not supported"
    );
    ensure!(!tokenizer.model.fuse_unk, "BPE fuse_unk is not supported");
    ensure!(
        !tokenizer.model.byte_fallback,
        "SentencePiece-style byte_fallback tokenizers are not supported"
    );
    validate_added_tokens(&tokenizer.added_tokens)?;
    validate_pretokenizer(&tokenizer.pre_tokenizer)?;
    validate_decoder(&tokenizer.decoder)?;

    let (_, unicode_to_byte) = byte_unicode_tables();
    let max_model_id = tokenizer.model.vocab.values().copied().max().unwrap_or(0) as usize;
    let max_added_id = tokenizer
        .added_tokens
        .iter()
        .map(|token| token.id as usize)
        .max()
        .unwrap_or(0);
    let vocab_len = max_model_id.max(max_added_id) + 1;
    let mut vocab: Vec<Arc<[u8]>> = vec![Arc::from([]); vocab_len];
    let mut pieces = vec![None; vocab_len];
    let mut token_to_id =
        HashMap::with_capacity(tokenizer.model.vocab.len() + tokenizer.added_tokens.len());
    let mut vocab_inverse: HashMap<Arc<[u8]>, TokenId, FxBuildHasher> =
        HashMap::with_capacity_and_hasher(tokenizer.model.vocab.len(), FxBuildHasher);

    for (piece, &id) in &tokenizer.model.vocab {
        let bytes: Arc<[u8]> = bytelevel_piece_to_bytes(piece, &unicode_to_byte).into();
        vocab[id as usize] = Arc::clone(&bytes);
        pieces[id as usize] = Some(piece.clone());
        token_to_id.insert(piece.clone(), id);
        ensure!(
            vocab_inverse
                .insert(Arc::clone(&bytes), TokenId::from(id))
                .is_none(),
            "multiple vocabulary entries decode to the same bytes as {piece:?}"
        );
    }

    let mut special_ids = HashSet::new();
    for token in &tokenizer.added_tokens {
        let id = token.id as usize;
        if let Some(existing) = &pieces[id] {
            ensure!(
                existing == &token.content,
                "added token {:?} reuses token ID {} belonging to {:?}",
                token.content,
                token.id,
                existing
            );
        }
        if vocab[id].is_empty() {
            vocab[id] = token.content.as_bytes().into();
        }
        pieces[id] = Some(token.content.clone());
        token_to_id.insert(token.content.clone(), token.id);
        if token.special {
            special_ids.insert(token.id);
        }
    }
    ensure!(
        pieces.iter().all(Option::is_some),
        "tokenizer vocabulary IDs must be dense"
    );

    let mut merge_entries = Vec::with_capacity(tokenizer.model.merges.len());
    for [left, right] in &tokenizer.model.merges {
        let left_bytes = bytelevel_piece_to_bytes(left, &unicode_to_byte);
        let right_bytes = bytelevel_piece_to_bytes(right, &unicode_to_byte);
        let left_id = *vocab_inverse
            .get(left_bytes.as_slice())
            .ok_or_else(|| anyhow::anyhow!("merge references unknown left token {left:?}"))?;
        let right_id = *vocab_inverse
            .get(right_bytes.as_slice())
            .ok_or_else(|| anyhow::anyhow!("merge references unknown right token {right:?}"))?;
        let mut merged_bytes = left_bytes;
        merged_bytes.extend_from_slice(&right_bytes);
        let merged_id = *vocab_inverse.get(merged_bytes.as_slice()).ok_or_else(|| {
            anyhow::anyhow!("merge {left:?} + {right:?} has no merged vocabulary token")
        })?;
        merge_entries.push((left_id, right_id, merged_id));
    }

    let byte_remapping = bpe::ByteRemapping::from_byte_vocab(&vocab)?;
    let vocab = vocab.into_iter().map(|entry| entry.to_vec()).collect();
    let ids_are_ranks = merge_entries.is_sorted_by_key(|&(_, _, merged)| merged);
    let mut engine = if ids_are_ranks {
        let mut merges = HashMap::with_capacity_and_hasher(merge_entries.len(), FxBuildHasher);
        for (left, right, merged) in merge_entries {
            merges.entry((left, right)).or_insert(merged);
        }
        bpe::tiktoken::Tokenizer::new(merges, vocab, byte_remapping)
    } else {
        let mut merges = bpe::tiktoken::RankedMerges::with_capacity_and_hasher(
            merge_entries.len(),
            FxBuildHasher,
        );
        for (rank, (left, right, merged)) in merge_entries.into_iter().enumerate() {
            merges
                .entry(bpe::ranked_merge_key(left, right))
                .or_insert((merged, rank as u32));
        }
        bpe::tiktoken::Tokenizer::new_ranked(merges, vocab, byte_remapping)
    };

    engine.set_pretokenizer_type(detect_pretokenizer(&tokenizer.pre_tokenizer)?);
    engine.set_normalize_nfc(detect_nfc(&tokenizer.normalizer)?);
    engine.set_add_prefix_space(detect_add_prefix_space(&tokenizer.pre_tokenizer));
    engine.set_ignore_merges(tokenizer.model.ignore_merges);
    engine.set_added_tokens(
        tokenizer
            .added_tokens
            .iter()
            .map(|token| bpe::tiktoken::AddedTokenDef {
                content: token.content.as_bytes().into(),
                id: TokenId::from(token.id),
                lstrip: token.lstrip,
                rstrip: token.rstrip,
            })
            .collect(),
    );

    let (prefix_ids, suffix_ids) = post_processor_ids(tokenizer.post_processor.as_ref())?;
    Ok(LoadedTokenizer {
        engine,
        pieces,
        token_to_id,
        special_ids,
        prefix_ids,
        suffix_ids,
    })
}

fn validate_added_tokens(tokens: &[AddedToken]) -> Result<()> {
    for token in tokens {
        ensure!(
            !token.single_word,
            "added token {:?} uses unsupported single_word matching",
            token.content
        );
        ensure!(
            !token.normalized || token.content.is_ascii(),
            "normalized non-ASCII added token {:?} is not supported",
            token.content
        );
    }
    Ok(())
}

fn validate_pretokenizer(pretokenizer: &Option<PreTokenizerJson>) -> Result<()> {
    fn walk(node: &PreTokenizerJson) -> Result<()> {
        ensure!(
            node.kind != "ByteLevel" || node.use_regex != Some(false),
            "ByteLevel pre_tokenizer with use_regex=false is not supported"
        );
        for child in &node.pretokenizers {
            walk(child)?;
        }
        Ok(())
    }

    pretokenizer.as_ref().map_or(Ok(()), walk)
}

fn validate_decoder(decoder: &Option<ComponentJson>) -> Result<()> {
    let decoder = decoder
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("byte-level BPE requires a ByteLevel decoder"))?;
    ensure!(
        decoder.kind == "ByteLevel",
        "unsupported decoder type {} (expected ByteLevel)",
        decoder.kind
    );
    Ok(())
}

fn detect_nfc(normalizer: &Option<NormalizerJson>) -> Result<bool> {
    fn walk(normalizer: &NormalizerJson) -> Result<bool> {
        match normalizer.kind.as_str() {
            "NFC" => Ok(true),
            "Sequence" => normalizer
                .normalizers
                .iter()
                .try_fold(false, |found, child| Ok(found | walk(child)?)),
            other => Err(anyhow::anyhow!("unsupported normalizer type {other}")),
        }
    }

    normalizer.as_ref().map_or(Ok(false), walk)
}

fn detect_pretokenizer(pretokenizer: &Option<PreTokenizerJson>) -> Result<PretokenizerType> {
    fn split_regexes<'a>(node: &'a PreTokenizerJson, output: &mut Vec<&'a str>) {
        if node.kind == "Split"
            && let Some(PatternJson { regex: Some(regex) }) = &node.pattern
        {
            output.push(regex);
        }
        for child in &node.pretokenizers {
            split_regexes(child, output);
        }
    }

    let pretokenizer = pretokenizer
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("byte-level BPE requires a pre_tokenizer"))?;
    let mut regexes = Vec::new();
    split_regexes(pretokenizer, &mut regexes);
    if regexes.is_empty() {
        fn has_byte_level(node: &PreTokenizerJson) -> bool {
            node.kind == "ByteLevel" || node.pretokenizers.iter().any(has_byte_level)
        }
        ensure!(
            has_byte_level(pretokenizer),
            "unsupported pre_tokenizer type {} (no Split regex found)",
            pretokenizer.kind
        );
        return Ok(PretokenizerType::GPT2);
    }
    PretokenizerType::from_split_regexes(&regexes)
        .ok_or_else(|| anyhow::anyhow!("unknown pre_tokenizer Split regexes: {regexes:?}"))
}

fn detect_add_prefix_space(pretokenizer: &Option<PreTokenizerJson>) -> bool {
    fn walk(node: &PreTokenizerJson) -> bool {
        (node.kind == "ByteLevel" && node.add_prefix_space == Some(true))
            || node.pretokenizers.iter().any(walk)
    }

    pretokenizer.as_ref().is_some_and(walk)
}

fn byte_unicode_tables() -> ([char; 256], HashMap<char, u8>) {
    let allowed: Vec<u8> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut byte_to_unicode = ['\0'; 256];
    for &byte in &allowed {
        byte_to_unicode[byte as usize] = byte as char;
    }
    let mut offset = 0;
    for byte in 0..=255u8 {
        if byte_to_unicode[byte as usize] == '\0' {
            byte_to_unicode[byte as usize] = char::from_u32(256 + offset).unwrap();
            offset += 1;
        }
    }
    let unicode_to_byte = byte_to_unicode
        .iter()
        .enumerate()
        .map(|(byte, &character)| (character, byte as u8))
        .collect();
    (byte_to_unicode, unicode_to_byte)
}

fn bytelevel_piece_to_bytes(piece: &str, unicode_to_byte: &HashMap<char, u8>) -> Vec<u8> {
    if piece
        .chars()
        .all(|character| unicode_to_byte.contains_key(&character))
    {
        piece
            .chars()
            .map(|character| unicode_to_byte[&character])
            .collect()
    } else {
        piece.as_bytes().to_vec()
    }
}

fn post_processor_ids(processor: Option<&Value>) -> Result<(Vec<u32>, Vec<u32>)> {
    let Some(processor) = processor else {
        return Ok((Vec::new(), Vec::new()));
    };
    let kind = processor
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    match kind {
        "" | "ByteLevel" => Ok((Vec::new(), Vec::new())),
        "Sequence" => {
            let mut prefix = Vec::new();
            let mut suffix = Vec::new();
            for child in processor
                .get("processors")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let (mut child_prefix, child_suffix) = post_processor_ids(Some(child))?;
                child_prefix.extend(prefix);
                prefix = child_prefix;
                suffix.extend(child_suffix);
            }
            Ok((prefix, suffix))
        }
        "TemplateProcessing" => template_processor_ids(processor),
        other => Err(anyhow::anyhow!("unsupported post_processor type {other}")),
    }
}

fn template_processor_ids(processor: &Value) -> Result<(Vec<u32>, Vec<u32>)> {
    let specials = processor
        .get("special_tokens")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("TemplateProcessing is missing special_tokens"))?;
    let single = processor
        .get("single")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("TemplateProcessing is missing single"))?;
    let mut prefix = Vec::new();
    let mut suffix = Vec::new();
    let mut saw_sequence = false;
    for item in single {
        if item.get("Sequence").is_some() {
            saw_sequence = true;
            continue;
        }
        let special = item
            .get("SpecialToken")
            .and_then(|value| value.get("id"))
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("unsupported TemplateProcessing item {item}"))?;
        let ids = specials
            .get(special)
            .and_then(|value| value.get("ids"))
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("TemplateProcessing token {special:?} has no ids"))?;
        let target = if saw_sequence {
            &mut suffix
        } else {
            &mut prefix
        };
        for id in ids {
            target.push(
                id.as_u64()
                    .and_then(|id| u32::try_from(id).ok())
                    .ok_or_else(|| anyhow::anyhow!("invalid special-token ID {id}"))?,
            );
        }
    }
    Ok((prefix, suffix))
}
