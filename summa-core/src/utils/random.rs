use rand::distributions::Alphanumeric;
use rand::thread_rng;
use rand::Rng;

/// Generates 12-characters string for `session_id` and `request_id`
pub fn generate_request_id() -> String {
    let mut rng = thread_rng();
    std::iter::repeat(()).map(|()| rng.sample(Alphanumeric) as char).take(12).collect()
}
