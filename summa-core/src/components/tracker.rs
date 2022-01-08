pub trait Tracker: Clone {
    fn get_status(&self) -> String;
    fn set_status(&self, new_status: &str);
}

#[derive(Clone, Default)]
pub struct NoTracker {}
impl Tracker for NoTracker {
    fn get_status(&self) -> String {
        unimplemented!()
    }
    fn set_status(&self, _: &str) {}
}
