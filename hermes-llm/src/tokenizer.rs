use anyhow::Result;
use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;

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
    inner: HfTokenizer,
    eos_token_id: u32,
}

impl Tokenizer {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let inner = HfTokenizer::from_file(path).map_err(|e| anyhow::anyhow!("{}", e))?;
        let vocab_size = inner.get_vocab_size(true);
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
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn encode_batch(
        &self,
        texts: Vec<String>,
        add_special_tokens: bool,
    ) -> Result<Vec<Vec<u32>>> {
        let encodings = self
            .inner
            .encode_batch(texts, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(encodings
            .into_iter()
            .map(|encoding| encoding.get_ids().to_vec())
            .collect())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("{}", e))
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }
}
