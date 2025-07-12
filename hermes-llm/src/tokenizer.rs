use anyhow::Result;
use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;

// Common special-token spellings across tokenizer families — kept in lockstep
// with hermes-train/src/hermes_train/tokenizer.py so train and serve resolve
// the same ids.
const EOS_NAMES: &[&str] = &[
    "<eos>",
    "<|endoftext|>",
    "</s>",
    "<|end_of_text|>",
    "<|eot_id|>",
];
const PAD_NAMES: &[&str] = &["<pad>", "<|padding|>", "<|pad|>"];
const BOS_NAMES: &[&str] = &["<bos>", "<|begin_of_text|>", "<s>", "<|startoftext|>"];

pub struct Tokenizer {
    inner: HfTokenizer,
    pad_token_id: u32,
    bos_token_id: u32,
    eos_token_id: u32,
}

impl Tokenizer {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let inner = HfTokenizer::from_file(path).map_err(|e| anyhow::anyhow!("{}", e))?;
        let vocab_size = inner.get_vocab_size(true) as u32;
        anyhow::ensure!(vocab_size > 0, "tokenizer has an empty vocabulary");
        // EOS is load-bearing (stop token); resolve it from known names before
        // falling back to a raw id, and warn loudly if the fallback is used.
        let resolve = |names: &[&str]| names.iter().find_map(|n| inner.token_to_id(n));
        let eos_token_id = resolve(EOS_NAMES).unwrap_or_else(|| {
            let fallback = 2.min(vocab_size - 1);
            tracing::warn!(
                "no known EOS token in tokenizer; falling back to id {fallback}. \
                 Generation stop-on-EOS may be wrong — set it explicitly."
            );
            fallback
        });
        let pad_token_id = resolve(PAD_NAMES).unwrap_or(eos_token_id);
        let bos_token_id = resolve(BOS_NAMES).unwrap_or(eos_token_id);
        Ok(Self {
            inner,
            pad_token_id,
            bos_token_id,
            eos_token_id,
        })
    }

    pub fn from_pretrained(identifier: &str) -> Result<Self> {
        // Use hf-hub to download the tokenizer file (uses rustls, no openssl)
        let client = hf_hub::HFClientSync::new()?;
        let (owner, name) = hf_hub::split_id(identifier);
        let repo = client.model(owner, name);
        let tokenizer_path = repo
            .download_file()
            .filename("tokenizer.json")
            .send()
            .map_err(|e| anyhow::anyhow!("Failed to download tokenizer: {}", e))?;
        Self::from_file(tokenizer_path)
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn encode_batch(&self, texts: &[&str], add_special_tokens: bool) -> Result<Vec<Vec<u32>>> {
        let encodings = self
            .inner
            .encode_batch(texts.to_vec(), add_special_tokens)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(encodings
            .into_iter()
            .map(|e| e.get_ids().to_vec())
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

    pub fn pad_token_id(&self) -> u32 {
        self.pad_token_id
    }

    pub fn bos_token_id(&self) -> u32 {
        self.bos_token_id
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub fn set_pad_token_id(&mut self, id: u32) {
        self.pad_token_id = id;
    }

    pub fn set_bos_token_id(&mut self, id: u32) {
        self.bos_token_id = id;
    }

    pub fn set_eos_token_id(&mut self, id: u32) {
        self.eos_token_id = id;
    }
}
