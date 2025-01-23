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
        let inner =
            HfTokenizer::from_pretrained(identifier, None).map_err(|e| anyhow::anyhow!("{}", e))?;
        let vocab_size = inner.get_vocab_size(true) as u32;
        Ok(Self {
            inner,
            pad_token_id: 0,
            bos_token_id: 1,
            eos_token_id: 2.min(vocab_size - 1),
        })
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

pub struct BPETrainer {
    vocab_size: usize,
    min_frequency: u32,
    special_tokens: Vec<String>,
}

impl BPETrainer {
    pub fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            min_frequency: 2,
            special_tokens: vec![
                "<pad>".to_string(),
                "<bos>".to_string(),
                "<eos>".to_string(),
                "<unk>".to_string(),
            ],
        }
    }

    pub fn with_min_frequency(mut self, freq: u32) -> Self {
        self.min_frequency = freq;
        self
    }

    pub fn with_special_tokens(mut self, tokens: Vec<String>) -> Self {
        self.special_tokens = tokens;
        self
    }

    pub fn train_from_files(&self, files: &[&str], output_path: &str) -> Result<Tokenizer> {
        use tokenizers::models::bpe::{BPE, BpeTrainerBuilder};
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;
        use tokenizers::tokenizer::Trainer;

        let special_tokens: Vec<tokenizers::AddedToken> = self
            .special_tokens
            .iter()
            .map(|s| tokenizers::AddedToken::from(s.as_str(), true))
            .collect();

        let mut trainer = BpeTrainerBuilder::default()
            .vocab_size(self.vocab_size)
            .min_frequency(self.min_frequency as u64)
            .special_tokens(special_tokens.clone())
            .build();

        let mut model = BPE::default();

        for file in files {
            let content = std::fs::read_to_string(file)?;
            let lines: Vec<&str> = content.lines().collect();
            trainer
                .feed(lines.iter().copied(), |s| Ok(vec![s.to_owned()]))
                .map_err(|e| anyhow::anyhow!("{}", e))?;
        }

        trainer
            .train(&mut model)
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        let mut tokenizer = HfTokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));
        tokenizer.add_special_tokens(&special_tokens);

        tokenizer
            .save(output_path, true)
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        Tokenizer::from_file(output_path)
    }

    pub fn train_from_texts(&self, texts: &[&str], output_path: &str) -> Result<Tokenizer> {
        use tokenizers::models::bpe::{BPE, BpeTrainerBuilder};
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;
        use tokenizers::tokenizer::Trainer;

        let special_tokens: Vec<tokenizers::AddedToken> = self
            .special_tokens
            .iter()
            .map(|s| tokenizers::AddedToken::from(s.as_str(), true))
            .collect();

        let mut trainer = BpeTrainerBuilder::default()
            .vocab_size(self.vocab_size)
            .min_frequency(self.min_frequency as u64)
            .special_tokens(special_tokens.clone())
            .build();

        let mut model = BPE::default();

        trainer
            .feed(texts.iter().copied(), |s| Ok(vec![s.to_owned()]))
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        trainer
            .train(&mut model)
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        let mut tokenizer = HfTokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));
        tokenizer.add_special_tokens(&special_tokens);

        tokenizer
            .save(output_path, true)
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        Tokenizer::from_file(output_path)
    }
}
