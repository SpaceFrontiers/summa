use anyhow::Result;
use hermes_tokenizer::Tokenizer as CoreTokenizer;
use std::path::Path;

// Training and inference resolve EOS through the same tokenizer wrapper, so
// the stop/sequence-boundary token cannot drift between executables.
const EOS_NAMES: &[&str] = &[
    "<eos>",
    "<|endoftext|>",
    "</s>",
    "<|end_of_text|>",
    "<|eot_id|>",
];

pub struct Tokenizer {
    inner: CoreTokenizer,
    eos_token_id: u32,
}

impl Tokenizer {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let inner = CoreTokenizer::from_file(path).map_err(|e| anyhow::anyhow!("{e:#}"))?;
        let vocab_size = inner.vocab_size();
        anyhow::ensure!(vocab_size > 0, "tokenizer has an empty vocabulary");
        let resolve = |names: &[&str]| names.iter().find_map(|n| inner.token_to_id(n));
        let eos_token_id = resolve(EOS_NAMES).ok_or_else(|| {
            anyhow::anyhow!(
                "tokenizer defines none of the supported EOS tokens: {}",
                EOS_NAMES.join(", ")
            )
        })?;
        Ok(Self {
            inner,
            eos_token_id,
        })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.inner
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("{e:#}"))
    }

    pub fn encode_batch(
        &self,
        texts: Vec<String>,
        add_special_tokens: bool,
    ) -> Result<Vec<Vec<u32>>> {
        self.inner
            .encode_batch(texts, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("{e:#}"))
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("{e:#}"))
    }

    /// Vocabulary piece for one token ID, without decoder cleanup. This is
    /// retained in traces so diagnostics never lose the tokenizer's exact
    /// representation.
    pub fn token_piece(&self, id: u32) -> Result<String> {
        self.inner
            .id_to_token(id)
            .ok_or_else(|| anyhow::anyhow!("tokenizer has no vocabulary entry for token ID {id}"))
    }

    /// Human-readable rendering of one vocabulary piece using the tokenizer's
    /// configured decoder. Byte-level vocabularies represent punctuation and
    /// whitespace with glyph sequences such as `âĢĻ`, `Ġ`, and `Ċ`; exposing
    /// those implementation markers in a token inspector is misleading.
    ///
    /// The raw piece remains available through [`Self::token_piece`]. A token
    /// that contains only part of a multi-byte character can still decode as a
    /// replacement glyph because its text genuinely spans token boundaries.
    pub fn display_piece(&self, id: u32) -> Result<String> {
        self.inner.decode(&[id], false).map_err(|e| {
            anyhow::anyhow!("failed to decode vocabulary entry for token ID {id}: {e}")
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }
}
