pub mod aggregation;
pub mod compression;
pub mod merge_policy;
pub mod order;
pub mod snippet;
pub mod sort_by_field;

pub struct Wrapper<T>(T);

impl<T> Wrapper<T> {
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T> From<T> for Wrapper<T> {
    fn from(value: T) -> Self {
        Wrapper(value)
    }
}
