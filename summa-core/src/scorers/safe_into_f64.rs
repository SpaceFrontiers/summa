use tantivy::DateTime;

pub(crate) trait SafeIntoF64 {
    fn safe_into_f64(self) -> f64;
}

impl SafeIntoF64 for u64 {
    fn safe_into_f64(self) -> f64 {
        self as f64
    }
}

impl SafeIntoF64 for i64 {
    fn safe_into_f64(self) -> f64 {
        self as f64
    }
}

impl SafeIntoF64 for f64 {
    fn safe_into_f64(self) -> f64 {
        self
    }
}

impl SafeIntoF64 for DateTime {
    fn safe_into_f64(self) -> f64 {
        self.into_timestamp_secs() as f64
    }
}
