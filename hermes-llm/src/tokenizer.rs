use anyhow::Result;
use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;

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
        Ok(Self {
            inner,
            pad_token_id: 0,
            bos_token_id: 1,
            eos_token_id: 2.min(vocab_size - 1),
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
