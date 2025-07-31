//! Fail-soft asynchronous NVIDIA device telemetry.
//!
//! The training thread remains the sole owner of the metric journal. This
//! module owns one persistent `nvidia-smi` process and a background reader,
//! then exposes bounded, non-blocking samples for the trainer to drain at safe
//! boundaries. A missing tool, malformed output, process exit, or a full
//! channel is observable without ever blocking model execution.

use std::collections::VecDeque;
use std::ffi::OsStr;
use std::io::{BufRead, BufReader, Read};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail, ensure};

use crate::metrics::{DeviceUtilizationMetrics, MetricEvent};

// Enough for a multi-minute sleep/checkpoint boundary at the default cadence,
// while remaining a fixed and tiny memory budget.
const DEFAULT_CHANNEL_CAPACITY: usize = 256;
const MAX_CHANNEL_CAPACITY: usize = 4_096;
const MAX_DIAGNOSTICS: usize = 16;
const MAX_DIAGNOSTIC_BYTES: usize = 4 * 1024;
const MAX_SAMPLE_BYTES: usize = 64 * 1024;
const MEBIBYTE: u64 = 1024 * 1024;

/// Exact process/query configuration. The device selector is passed as a
/// single `--id` argument, never through a shell. NVIDIA accepts a physical
/// index, GPU UUID, or PCI bus ID.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NvidiaSmiSamplerConfig {
    pub interval: Duration,
    pub physical_device: String,
    pub channel_capacity: usize,
}

impl NvidiaSmiSamplerConfig {
    pub fn new(interval: Duration, physical_device: impl Into<String>) -> Result<Self> {
        let value = Self {
            interval,
            physical_device: physical_device.into(),
            channel_capacity: DEFAULT_CHANNEL_CAPACITY,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.interval.is_zero(),
            "device sampling interval must be positive"
        );
        ensure!(
            self.interval.as_millis() <= u64::MAX as u128,
            "device sampling interval does not fit nvidia-smi"
        );
        validate_physical_device_selector(&self.physical_device)?;
        ensure!(
            (1..=MAX_CHANNEL_CAPACITY).contains(&self.channel_capacity),
            "device sample channel capacity must be in 1..={MAX_CHANNEL_CAPACITY}"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeviceSample {
    pub collected_at_unix_ms: u64,
    pub metrics: DeviceUtilizationMetrics,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DeviceSamplerDiagnostic {
    InvalidSample(String),
    ProcessExited(String),
    DroppedSamples(u64),
    DroppedDiagnostics(u64),
}

impl std::fmt::Display for DeviceSamplerDiagnostic {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSample(message) => {
                write!(formatter, "discarded invalid nvidia-smi sample: {message}")
            }
            Self::ProcessExited(message) => write!(formatter, "nvidia-smi stopped: {message}"),
            Self::DroppedSamples(count) => {
                write!(
                    formatter,
                    "dropped {count} device samples because the channel was full"
                )
            }
            Self::DroppedDiagnostics(count) => write!(
                formatter,
                "dropped {count} device-sampler diagnostics because the diagnostic buffer was full"
            ),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeviceSamplerDrain {
    pub samples: Vec<DeviceSample>,
    pub diagnostics: Vec<DeviceSamplerDiagnostic>,
}

/// Running persistent sampler. Dropping it kills and reaps `nvidia-smi`, then
/// joins both output readers. Runtime errors are returned by [`Self::drain`]
/// rather than panicking or poisoning training state.
pub struct NvidiaSmiSampler {
    samples: Receiver<DeviceSample>,
    diagnostics: Arc<Mutex<VecDeque<DeviceSamplerDiagnostic>>>,
    dropped_samples: Arc<AtomicU64>,
    dropped_diagnostics: Arc<AtomicU64>,
    stopping: Arc<AtomicBool>,
    child: Arc<Mutex<Option<Child>>>,
    worker: Option<JoinHandle<()>>,
}

impl NvidiaSmiSampler {
    pub fn start(config: NvidiaSmiSamplerConfig) -> Result<Self> {
        Self::start_with_program(config, OsStr::new("nvidia-smi"))
    }

    /// Attempt startup without turning observability into a training
    /// dependency. The diagnostic is intentionally returned to the caller so
    /// it can be logged on the trainer thread.
    pub fn start_fail_soft(config: NvidiaSmiSamplerConfig) -> (Option<Self>, Option<String>) {
        Self::start_fail_soft_with_program(config, OsStr::new("nvidia-smi"))
    }

    fn start_fail_soft_with_program(
        config: NvidiaSmiSamplerConfig,
        program: &OsStr,
    ) -> (Option<Self>, Option<String>) {
        match Self::start_with_program(config, program) {
            Ok(sampler) => (Some(sampler), None),
            Err(error) => (None, Some(format!("{error:#}"))),
        }
    }

    fn start_with_program(config: NvidiaSmiSamplerConfig, program: &OsStr) -> Result<Self> {
        config.validate()?;
        let loop_millis = u64::try_from(config.interval.as_millis())
            .context("device sampling interval does not fit nvidia-smi")?;
        ensure!(
            loop_millis > 0,
            "device sampling interval rounds to zero milliseconds"
        );

        let mut command = Command::new(program);
        command
            .arg("--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu")
            .arg("--format=csv,noheader,nounits")
            .arg("--id")
            .arg(&config.physical_device)
            .arg(format!("--loop-ms={loop_millis}"))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            // A private process group lets Drop terminate helper descendants
            // which inherited the pipes, too; otherwise joining the readers
            // could wait on an orphan after nvidia-smi itself exits.
            command.process_group(0);
        }
        let mut child = command
            .spawn()
            .with_context(|| format!("start {}", program.to_string_lossy()))?;
        let stdout = child
            .stdout
            .take()
            .context("nvidia-smi stdout is unavailable")?;
        let stderr = child
            .stderr
            .take()
            .context("nvidia-smi stderr is unavailable")?;
        let child = Arc::new(Mutex::new(Some(child)));
        let stopping = Arc::new(AtomicBool::new(false));
        let diagnostics = Arc::new(Mutex::new(VecDeque::new()));
        let dropped_samples = Arc::new(AtomicU64::new(0));
        let dropped_diagnostics = Arc::new(AtomicU64::new(0));
        let (sender, samples) = mpsc::sync_channel(config.channel_capacity);

        let worker_child = Arc::clone(&child);
        let worker_stopping = Arc::clone(&stopping);
        let worker_diagnostics = Arc::clone(&diagnostics);
        let worker_dropped_samples = Arc::clone(&dropped_samples);
        let worker_dropped_diagnostics = Arc::clone(&dropped_diagnostics);
        let interval = config.interval;
        let worker = thread::Builder::new()
            .name("hermes-nvidia-smi".into())
            .spawn(move || {
                run_reader(
                    stdout,
                    stderr,
                    sender,
                    interval,
                    &worker_child,
                    &worker_stopping,
                    &worker_diagnostics,
                    &worker_dropped_samples,
                    &worker_dropped_diagnostics,
                );
            });
        let worker = match worker {
            Ok(worker) => worker,
            Err(error) => {
                stop_child(&child);
                return Err(error).context("start nvidia-smi reader thread");
            }
        };

        Ok(Self {
            samples,
            diagnostics,
            dropped_samples,
            dropped_diagnostics,
            stopping,
            child,
            worker: Some(worker),
        })
    }

    /// Drain every sample currently available without waiting for another
    /// process read. Drop counters are exchanged atomically so each loss is
    /// reported once.
    pub fn drain(&mut self) -> DeviceSamplerDrain {
        let samples = self.samples.try_iter().collect::<Vec<_>>();
        let mut diagnostics = lock_unpoisoned(&self.diagnostics)
            .drain(..)
            .collect::<Vec<_>>();
        let dropped_samples = self.dropped_samples.swap(0, Ordering::Relaxed);
        if dropped_samples > 0 {
            diagnostics.push(DeviceSamplerDiagnostic::DroppedSamples(dropped_samples));
        }
        let dropped_diagnostics = self.dropped_diagnostics.swap(0, Ordering::Relaxed);
        if dropped_diagnostics > 0 {
            diagnostics.push(DeviceSamplerDiagnostic::DroppedDiagnostics(
                dropped_diagnostics,
            ));
        }
        DeviceSamplerDrain {
            samples,
            diagnostics,
        }
    }

    /// Stop and reap the child before returning the final bounded batch. This
    /// is used before the terminal checkpoint so no successful-run sample is
    /// left behind after checkpoint state commits its metric prefix.
    pub fn shutdown_and_drain(&mut self) -> DeviceSamplerDrain {
        self.stop();
        self.drain()
    }

    fn stop(&mut self) {
        self.stopping.store(true, Ordering::Release);
        stop_child(&self.child);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Drop for NvidiaSmiSampler {
    fn drop(&mut self) {
        self.stop();
    }
}

#[allow(clippy::too_many_arguments)]
fn run_reader(
    stdout: impl Read + Send + 'static,
    stderr: impl Read + Send + 'static,
    sender: SyncSender<DeviceSample>,
    interval: Duration,
    child: &Arc<Mutex<Option<Child>>>,
    stopping: &AtomicBool,
    diagnostics: &Mutex<VecDeque<DeviceSamplerDiagnostic>>,
    dropped_samples: &AtomicU64,
    dropped_diagnostics: &AtomicU64,
) {
    let stderr_reader = thread::Builder::new()
        .name("hermes-nvidia-smi-stderr".into())
        .spawn(move || read_bounded(stderr, MAX_DIAGNOSTIC_BYTES));
    let mut output = BufReader::new(stdout);
    let mut line = Vec::new();
    let mut last_collected_at_unix_ms = 0_u64;
    loop {
        match read_sample_line_bounded(&mut output, &mut line, MAX_SAMPLE_BYTES) {
            Ok(None) => break,
            Ok(Some(line)) => match collection_timestamp(&mut last_collected_at_unix_ms).and_then(
                |collected_at_unix_ms| {
                    parse_nvidia_smi_line(line, interval, collected_at_unix_ms)
                        .map(|metrics| (collected_at_unix_ms, metrics))
                },
            ) {
                Ok((collected_at_unix_ms, metrics)) => match sender.try_send(DeviceSample {
                    collected_at_unix_ms,
                    metrics,
                }) {
                    Ok(()) => {}
                    Err(TrySendError::Full(_)) => {
                        dropped_samples.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(TrySendError::Disconnected(_)) => break,
                },
                Err(error) => push_diagnostic(
                    diagnostics,
                    dropped_diagnostics,
                    DeviceSamplerDiagnostic::InvalidSample(error.to_string()),
                ),
            },
            Err(error) => {
                if !stopping.load(Ordering::Acquire) {
                    push_diagnostic(
                        diagnostics,
                        dropped_diagnostics,
                        DeviceSamplerDiagnostic::InvalidSample(error.to_string()),
                    );
                }
                break;
            }
        }
    }

    let unexpected_eof = !stopping.load(Ordering::Acquire);
    let mut status = {
        let mut child = lock_unpoisoned(child);
        child
            .as_mut()
            .and_then(|process| process.try_wait().ok().flatten())
    };
    // A broken or wrapped helper can close stdout while remaining alive with
    // stderr open. Joining the stderr reader first would then block forever.
    // Terminate and reap the private process group before the join; telemetry
    // remains fail-soft and Drop can safely call the same idempotent cleanup.
    if unexpected_eof && status.is_none() {
        status = stop_child(child);
    }
    let stderr = stderr_reader
        .ok()
        .and_then(|reader| reader.join().ok())
        .unwrap_or_default();
    if unexpected_eof {
        let status = status.map_or_else(
            || "stdout closed while the process status was unavailable".to_owned(),
            |status| format!("process exited with {status}"),
        );
        let message = if stderr.trim().is_empty() {
            status
        } else {
            format!("{status}: {}", stderr.trim())
        };
        push_diagnostic(
            diagnostics,
            dropped_diagnostics,
            DeviceSamplerDiagnostic::ProcessExited(message),
        );
    }
}

fn read_sample_line_bounded<'a>(
    reader: &mut (impl BufRead + ?Sized),
    output: &'a mut Vec<u8>,
    maximum_bytes: usize,
) -> Result<Option<&'a str>> {
    ensure!(
        maximum_bytes > 0,
        "device sample byte limit must be positive"
    );
    output.clear();
    let capture_bytes = maximum_bytes
        .checked_add(1)
        .context("device sample byte limit overflows usize")?;
    loop {
        let available = reader
            .fill_buf()
            .context("failed to read nvidia-smi stdout")?;
        if available.is_empty() {
            break;
        }
        let through_delimiter = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |position| position + 1);
        let remaining = capture_bytes
            .checked_sub(output.len())
            .context("device sample capture length overflow")?;
        let copied = through_delimiter.min(remaining);
        output.extend_from_slice(&available[..copied]);
        reader.consume(copied);
        if copied < through_delimiter
            || output.len() == capture_bytes
            || output.last() == Some(&b'\n')
        {
            break;
        }
    }
    if output.is_empty() {
        return Ok(None);
    }
    let payload_bytes = output
        .len()
        .checked_sub(usize::from(output.last() == Some(&b'\n')))
        .context("device sample byte count underflows usize")?;
    ensure!(
        payload_bytes <= maximum_bytes,
        "nvidia-smi sample exceeds the maximum of {maximum_bytes} bytes"
    );
    let line = std::str::from_utf8(output).context("nvidia-smi sample is not UTF-8")?;
    Ok(Some(line))
}

/// Terminate and reap the sampler's private process group, returning its exit
/// status. Takes the handle so the raw group signal is issued at most once:
/// reaping frees the pid, and the kernel may hand the same pid — and therefore
/// the same process-group id — to an unrelated process afterwards.
fn stop_child(child: &Mutex<Option<Child>>) -> Option<ExitStatus> {
    let mut guard = lock_unpoisoned(child);
    let mut process = guard.take()?;
    #[cfg(unix)]
    {
        let process_group = -(process.id() as i32);
        // SAFETY: the child was placed in a private process group before
        // spawn; a negative pid targets only that group, and the process is
        // still unreaped here so that group id cannot have been recycled.
        // Failure is harmless because `Child::kill` below remains the fallback.
        unsafe {
            libc::kill(process_group, libc::SIGKILL);
        }
    }
    let _ = process.kill();
    process.wait().ok()
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn push_diagnostic(
    diagnostics: &Mutex<VecDeque<DeviceSamplerDiagnostic>>,
    dropped: &AtomicU64,
    diagnostic: DeviceSamplerDiagnostic,
) {
    let mut diagnostics = lock_unpoisoned(diagnostics);
    if diagnostics.len() == MAX_DIAGNOSTICS {
        diagnostics.pop_front();
        dropped.fetch_add(1, Ordering::Relaxed);
    }
    diagnostics.push_back(diagnostic);
}

fn read_bounded(mut input: impl Read, limit: usize) -> String {
    let mut bytes = Vec::new();
    let _ = input.by_ref().take(limit as u64).read_to_end(&mut bytes);
    String::from_utf8_lossy(&bytes).into_owned()
}

pub fn validate_physical_device_selector(selector: &str) -> Result<()> {
    ensure!(!selector.is_empty(), "physical GPU selector is empty");
    ensure!(
        !selector.starts_with('-'),
        "physical GPU selector must not look like an option"
    );
    ensure!(selector.len() <= 128, "physical GPU selector is too long");
    ensure!(
        selector.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':' | b'/')
        }),
        "physical GPU selector contains unsupported characters"
    );
    Ok(())
}

/// Parse one `csv,noheader,nounits` result. Required utilization and memory
/// fields reject `N/A`; optional power and temperature retain `None` when the
/// driver cannot report them.
pub fn parse_nvidia_smi_line(
    line: &str,
    interval: Duration,
    sampled_at_unix_ms: u64,
) -> Result<DeviceUtilizationMetrics> {
    ensure!(!interval.is_zero(), "device sample interval is zero");
    let fields = line.trim().split(',').map(str::trim).collect::<Vec<_>>();
    ensure!(
        fields.len() == 6,
        "expected 6 nvidia-smi fields, observed {}",
        fields.len()
    );
    let device_index = parse_required::<u32>(fields[0], "device index")?;
    let gpu_utilization_percent = parse_required::<f64>(fields[1], "GPU utilization")?;
    ensure!(
        gpu_utilization_percent.is_finite() && (0.0..=100.0).contains(&gpu_utilization_percent),
        "GPU utilization is outside 0..=100"
    );
    let memory_used_bytes = parse_mebibytes(fields[2], "used GPU memory")?;
    let memory_total_bytes = parse_mebibytes(fields[3], "total GPU memory")?;
    ensure!(
        memory_total_bytes > 0 && memory_used_bytes <= memory_total_bytes,
        "GPU memory usage exceeds total memory or total memory is zero"
    );
    let power_watts = parse_optional_f64(fields[4], "GPU power")?;
    if let Some(power) = power_watts {
        ensure!(power >= 0.0, "GPU power is negative");
    }
    let temperature_celsius = parse_optional_f64(fields[5], "GPU temperature")?;
    let metrics = DeviceUtilizationMetrics {
        sampled_at_unix_ms,
        device_index,
        sample_window_seconds: interval.as_secs_f64(),
        gpu_utilization_percent,
        sm_active_percent: None,
        tensor_core_active_percent: None,
        memory_bandwidth_percent: None,
        memory_used_bytes,
        memory_total_bytes,
        power_watts,
        temperature_celsius,
    };
    MetricEvent::DeviceUtilization(metrics.clone()).validate()?;
    Ok(metrics)
}

fn collection_timestamp(last: &mut u64) -> Result<u64> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_millis()
        .try_into()
        .context("Unix timestamp does not fit u64 milliseconds")?;
    *last = (*last).max(now);
    Ok(*last)
}

fn parse_required<T>(value: &str, label: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    if unavailable(value) {
        bail!("{label} is unavailable");
    }
    value
        .parse()
        .map_err(|error| anyhow::anyhow!("parse {label}: {error}"))
}

fn parse_mebibytes(value: &str, label: &str) -> Result<u64> {
    let mebibytes = parse_required::<u64>(value, label)?;
    mebibytes
        .checked_mul(MEBIBYTE)
        .with_context(|| format!("{label} overflows bytes"))
}

fn parse_optional_f64(value: &str, label: &str) -> Result<Option<f64>> {
    if unavailable(value) {
        return Ok(None);
    }
    let value = value
        .parse::<f64>()
        .with_context(|| format!("parse {label}"))?;
    ensure!(value.is_finite(), "{label} is not finite");
    Ok(Some(value))
}

fn unavailable(value: &str) -> bool {
    matches!(value, "N/A" | "[N/A]" | "Not Supported")
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Cursor;
    use std::path::Path;
    use std::time::{Duration, Instant};

    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    use super::*;

    #[test]
    fn sample_reader_bounds_payload_and_validates_utf8() {
        let mut output = Vec::new();
        let mut exact = Cursor::new(b"12345678\nnext".to_vec());
        assert_eq!(
            read_sample_line_bounded(&mut exact, &mut output, 8).unwrap(),
            Some("12345678\n")
        );

        let mut oversized = Cursor::new(b"123456789\n".to_vec());
        let error = read_sample_line_bounded(&mut oversized, &mut output, 8)
            .unwrap_err()
            .to_string();
        assert!(error.contains("maximum of 8 bytes"), "{error}");
        assert_eq!(output.len(), 9);

        let mut invalid = Cursor::new(vec![0xff, b'\n']);
        let error = read_sample_line_bounded(&mut invalid, &mut output, 8)
            .unwrap_err()
            .to_string();
        assert!(error.contains("not UTF-8"), "{error}");
    }

    #[test]
    fn parser_converts_units_and_accepts_unavailable_optional_fields() {
        let metrics = parse_nvidia_smi_line(
            "2, 87, 1024, 40960, 245.75, 63\n",
            Duration::from_millis(1500),
            1234,
        )
        .unwrap();
        assert_eq!(metrics.device_index, 2);
        assert_eq!(metrics.sampled_at_unix_ms, 1234);
        assert_eq!(metrics.gpu_utilization_percent, 87.0);
        assert_eq!(metrics.memory_used_bytes, 1024 * MEBIBYTE);
        assert_eq!(metrics.memory_total_bytes, 40960 * MEBIBYTE);
        assert_eq!(metrics.power_watts, Some(245.75));
        assert_eq!(metrics.temperature_celsius, Some(63.0));
        assert_eq!(metrics.sample_window_seconds, 1.5);

        let metrics =
            parse_nvidia_smi_line("0, 0, 0, 81920, N/A, [N/A]", Duration::from_secs(1), 1235)
                .unwrap();
        assert_eq!(metrics.power_watts, None);
        assert_eq!(metrics.temperature_celsius, None);
    }

    #[test]
    fn parser_rejects_invalid_required_measurements() {
        for line in [
            "0, 101, 0, 10, 1, 1",
            "0, N/A, 0, 10, 1, 1",
            "0, 10, 11, 10, 1, 1",
            "0, 10, 1, 10, -1, 1",
            "0, 10, 1, 10, 1",
        ] {
            assert!(
                parse_nvidia_smi_line(line, Duration::from_secs(1), 1).is_err(),
                "accepted {line:?}"
            );
        }
    }

    #[test]
    fn selectors_are_single_safe_nvidia_identifiers() {
        for selector in ["0", "GPU-1234-abcd", "0000:81:00.0", "MIG-GPU-a/1/2"] {
            validate_physical_device_selector(selector).unwrap();
        }
        for selector in ["", "0,1", "--help", "0\n1", "gpu name"] {
            assert!(
                validate_physical_device_selector(selector).is_err(),
                "accepted {selector:?}"
            );
        }
    }

    #[test]
    fn sampler_channel_allocation_is_operationally_bounded() {
        let mut config = NvidiaSmiSamplerConfig::new(Duration::from_secs(1), "0").unwrap();
        config.channel_capacity = MAX_CHANNEL_CAPACITY + 1;
        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains("1..="), "{error}");
    }

    #[test]
    fn unavailable_program_is_fail_soft_and_observable() {
        let config = NvidiaSmiSamplerConfig::new(Duration::from_secs(1), "0").unwrap();
        let (sampler, diagnostic) = NvidiaSmiSampler::start_fail_soft_with_program(
            config,
            OsStr::new("/definitely/missing/hermes-nvidia-smi"),
        );
        assert!(sampler.is_none());
        let diagnostic = diagnostic.unwrap();
        assert!(diagnostic.contains("start"), "{diagnostic}");
    }

    #[cfg(unix)]
    #[test]
    fn unexpected_process_exit_is_reported_without_panicking() {
        let directory = tempfile::tempdir().unwrap();
        let program = directory.path().join("exiting-nvidia-smi");
        executable(
            &program,
            "#!/bin/sh\nprintf '0, 50, 1, 10, 20, 30\\n'\nprintf 'driver failed\\n' >&2\nexit 7\n",
        );
        let config = NvidiaSmiSamplerConfig::new(Duration::from_secs(1), "0").unwrap();
        let mut sampler =
            NvidiaSmiSampler::start_with_program(config, program.as_os_str()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let drain = sampler.drain();
            if drain.diagnostics.iter().any(|diagnostic| {
                matches!(diagnostic, DeviceSamplerDiagnostic::ProcessExited(message) if message.contains("driver failed"))
            }) {
                break;
            }
            assert!(Instant::now() < deadline, "process exit was not reported");
            thread::sleep(Duration::from_millis(10));
        }
    }

    #[cfg(unix)]
    #[test]
    fn stdout_eof_kills_a_live_helper_before_joining_stderr() {
        let directory = tempfile::tempdir().unwrap();
        let program = directory.path().join("closed-stdout-nvidia-smi");
        executable(
            &program,
            "#!/bin/sh\nprintf 'helper closed stdout\\n' >&2\nexec 1>&-\nexec sleep 30\n",
        );
        let config = NvidiaSmiSamplerConfig::new(Duration::from_secs(1), "0").unwrap();
        let started = Instant::now();
        let mut sampler =
            NvidiaSmiSampler::start_with_program(config, program.as_os_str()).unwrap();
        // Full workspace runs exercise several CPU-heavy index tests in
        // parallel. Allow scheduler contention without weakening the hang
        // check: the deliberately stuck helper sleeps for 30 seconds.
        let timeout = Duration::from_secs(5);
        let deadline = Instant::now() + timeout;
        loop {
            let drain = sampler.drain();
            if drain.diagnostics.iter().any(|diagnostic| {
                matches!(diagnostic, DeviceSamplerDiagnostic::ProcessExited(message) if message.contains("helper closed stdout"))
            }) {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "stdout EOF did not terminate the live helper"
            );
            thread::sleep(Duration::from_millis(10));
        }
        drop(sampler);
        assert!(started.elapsed() < timeout);
    }

    #[cfg(unix)]
    #[test]
    fn persistent_sampler_is_bounded_observable_and_stops_cleanly() {
        let directory = tempfile::tempdir().unwrap();
        let program = directory.path().join("fake-nvidia-smi");
        executable(
            &program,
            "#!/bin/sh\nprintf 'malformed\\n'\ni=0\nwhile [ \"$i\" -lt 64 ]; do\n  printf '3, 92, 2048, 40960, 255.5, 67\\n'\n  i=$((i + 1))\ndone\nsleep 30\n",
        );
        let mut config = NvidiaSmiSamplerConfig::new(Duration::from_millis(250), "3").unwrap();
        config.channel_capacity = 1;
        let mut sampler =
            NvidiaSmiSampler::start_with_program(config, program.as_os_str()).unwrap();

        let deadline = Instant::now() + Duration::from_secs(2);
        let drain = loop {
            let drain = sampler.drain();
            if !drain.samples.is_empty()
                && drain.diagnostics.iter().any(|diagnostic| {
                    matches!(
                        diagnostic,
                        DeviceSamplerDiagnostic::InvalidSample(_)
                            | DeviceSamplerDiagnostic::DroppedSamples(_)
                    )
                })
            {
                break drain;
            }
            assert!(
                Instant::now() < deadline,
                "sampler produced no bounded output"
            );
            thread::sleep(Duration::from_millis(10));
        };
        assert_eq!(drain.samples.len(), 1);
        assert_eq!(drain.samples[0].metrics.device_index, 3);
        let stopped = Instant::now();
        drop(sampler);
        assert!(stopped.elapsed() < Duration::from_secs(2));
    }

    #[cfg(unix)]
    fn executable(path: &Path, source: &str) {
        fs::write(path, source).unwrap();
        let mut permissions = fs::metadata(path).unwrap().permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(path, permissions).unwrap();
    }
}
