#![cfg(feature = "native")]

/// Exercise the public environment initializer in child processes so tests do
/// not mutate process-global environment or the once-initialized pin policy.
#[test]
fn pin_budget_environment_is_checked_and_never_wraps() {
    const EXPECTED: &str = "HERMES_TEST_EXPECTED_PIN_BYTES";
    if let Ok(expected) = std::env::var(EXPECTED) {
        let policy = hermes_core::segment::pin::PinPolicy::from_env();
        assert_eq!(policy.budget_bytes, expected.parse::<u64>().unwrap());
        return;
    }
    for (input, expected) in [
        (None, 0),
        (Some("0"), 0),
        (Some("128"), 128 * 1024 * 1024),
        (Some("invalid"), 0),
        (Some("-1"), 0),
        (Some("18446744073709551615"), 0),
    ] {
        let mut child = std::process::Command::new(std::env::current_exe().unwrap());
        child
            .args([
                "--exact",
                "pin_budget_environment_is_checked_and_never_wraps",
                "--nocapture",
            ])
            .env(EXPECTED, expected.to_string());
        match input {
            Some(value) => {
                child.env("HERMES_PIN_METADATA_BUDGET_MB", value);
            }
            None => {
                child.env_remove("HERMES_PIN_METADATA_BUDGET_MB");
            }
        }
        let output = child.output().unwrap();
        assert!(
            output.status.success(),
            "budget {input:?}: {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
