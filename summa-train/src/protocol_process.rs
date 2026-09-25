//! Deadline-bounded supervision for local JSONL worker processes.
//!
//! Protocol pipes are nonblocking and owned by the supervising thread. This
//! is intentional: a blocking reader thread cannot be joined safely when an
//! unauthorized descendant calls `setsid(2)` and retains a copied pipe after
//! the worker leader exits. The supervisor instead drains bytes that are
//! currently available, treats leader exit as the protocol boundary, closes
//! its pipe ends, and reaps only the child it actually owns.

#![cfg(unix)]

use std::io::{Read, Write};
use std::os::fd::{AsRawFd, RawFd};
use std::process::{Child, ChildStdin, ChildStdout, ExitStatus};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};

const READ_CHUNK_BYTES: usize = 8 * 1024;
const MAX_POLL_INTERVAL: Duration = Duration::from_millis(100);

pub(crate) enum ProtocolRead {
    Pending,
    Line(Vec<u8>),
    Eof,
}

/// One child leader plus its nonblocking protocol pipes.
///
/// The child must have been spawned as the leader of a fresh process group.
/// `Drop` kills that group, closes both host pipe ends, and reaps the leader.
/// Descendants that deliberately escape with `setsid(2)` cannot strand a host
/// thread because this type never performs a blocking pipe read or join.
pub(crate) struct SupervisedProcess {
    child: Child,
    input: Option<ChildStdin>,
    output: Option<ChildStdout>,
    output_buffer: Vec<u8>,
    output_eof: bool,
    status: Option<ExitStatus>,
    process_group_terminated: bool,
    label: &'static str,
    max_message_bytes: usize,
}

impl SupervisedProcess {
    pub(crate) fn new(
        mut child: Child,
        label: &'static str,
        max_message_bytes: usize,
    ) -> Result<Self> {
        ensure!(
            max_message_bytes > 0,
            "protocol message bound must be positive"
        );
        let Some(input) = child.stdin.take() else {
            terminate_child(&mut child);
            bail!("{label} stdin is unavailable");
        };
        let Some(output) = child.stdout.take() else {
            terminate_child(&mut child);
            bail!("{label} stdout is unavailable");
        };
        if let Err(error) = set_nonblocking(&input).and_then(|()| set_nonblocking(&output)) {
            terminate_child(&mut child);
            return Err(error).with_context(|| format!("configuring {label} protocol pipes"));
        }
        Ok(Self {
            child,
            input: Some(input),
            output: Some(output),
            output_buffer: Vec::new(),
            output_eof: false,
            status: None,
            process_group_terminated: false,
            label,
            max_message_bytes,
        })
    }

    /// Write as much of one already-bounded request as the pipe accepts.
    pub(crate) fn write_available(&mut self, request: &[u8], written: &mut usize) -> Result<()> {
        ensure!(
            *written <= request.len(),
            "{} request cursor exceeds its encoded length",
            self.label
        );
        while *written < request.len() {
            let input = self
                .input
                .as_mut()
                .with_context(|| format!("{} stdin is closed", self.label))?;
            match input.write(&request[*written..]) {
                Ok(0) => bail!("{} stopped accepting its request", self.label),
                Ok(count) => *written += count,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => return Ok(()),
                Err(error) => {
                    return Err(error).with_context(|| format!("writing {} request", self.label));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn close_input(&mut self) {
        self.input.take();
    }

    /// Whether bytes have been received that do not yet form a complete frame.
    /// Persistent protocols use this to prevent partial output from being
    /// silently carried across request boundaries.
    pub(crate) fn has_buffered_output(&self) -> bool {
        !self.output_buffer.is_empty()
    }

    /// Return one complete bounded JSONL frame without ever blocking.
    pub(crate) fn read_line(&mut self) -> Result<ProtocolRead> {
        loop {
            if let Some(newline) = self.output_buffer.iter().position(|byte| *byte == b'\n') {
                let line_len = newline + 1;
                ensure!(
                    line_len <= self.max_message_bytes,
                    "{} emitted a JSONL message larger than {} bytes",
                    self.label,
                    self.max_message_bytes
                );
                let trailing = self.output_buffer.split_off(line_len);
                let line = std::mem::replace(&mut self.output_buffer, trailing);
                return Ok(ProtocolRead::Line(line));
            }
            if self.output_eof {
                ensure!(
                    self.output_buffer.is_empty(),
                    "{} emitted an unterminated JSONL message",
                    self.label
                );
                return Ok(ProtocolRead::Eof);
            }

            let output = self
                .output
                .as_mut()
                .with_context(|| format!("{} stdout is closed", self.label))?;
            let mut chunk = [0_u8; READ_CHUNK_BYTES];
            match output.read(&mut chunk) {
                Ok(0) => self.output_eof = true,
                Ok(count) => {
                    let new_len = self
                        .output_buffer
                        .len()
                        .checked_add(count)
                        .with_context(|| format!("{} response length overflow", self.label))?;
                    // If this chunk contains a newline, the exact frame bound
                    // is checked at the top of the loop. Otherwise all bytes
                    // belong to one still-incomplete frame and are bounded now.
                    if !chunk[..count].contains(&b'\n') {
                        ensure!(
                            new_len <= self.max_message_bytes,
                            "{} emitted a JSONL message larger than {} bytes",
                            self.label,
                            self.max_message_bytes
                        );
                    }
                    self.output_buffer.extend_from_slice(&chunk[..count]);
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    return Ok(ProtocolRead::Pending);
                }
                Err(error) => {
                    return Err(error).with_context(|| format!("reading {} response", self.label));
                }
            }
        }
    }

    /// Validate the bytes left after every currently available frame has been
    /// drained. This is used when the leader exits but an escaped descendant
    /// keeps the pipe from reaching EOF.
    pub(crate) fn finish_output_at_leader_exit(&self) -> Result<()> {
        ensure!(
            self.output_buffer.is_empty(),
            "{} emitted an unterminated JSONL message",
            self.label
        );
        Ok(())
    }

    pub(crate) fn try_wait(&mut self) -> Result<Option<ExitStatus>> {
        if let Some(status) = self.status {
            return Ok(Some(status));
        }
        if !self.process_group_terminated {
            // Peek with WNOWAIT so an exited leader is detected while it is
            // still an unreaped zombie: the zombie keeps the pid/pgid
            // reserved, so the group SIGKILL below cannot land on a recycled
            // process group. Only afterwards is the leader actually reaped.
            if !leader_exited_unreaped(&self.child, self.label)? {
                return Ok(None);
            }
            self.terminate_process_group();
        }
        let status = self
            .child
            .try_wait()
            .with_context(|| format!("waiting for {}", self.label))?;
        if let Some(status) = status {
            self.status = Some(status);
        }
        Ok(status)
    }

    pub(crate) fn wait_for_activity(&self, want_write: bool, deadline: Instant) -> Result<()> {
        let output = (!self.output_eof)
            .then(|| self.output.as_ref().map(AsRawFd::as_raw_fd))
            .flatten();
        let input = want_write
            .then(|| self.input.as_ref().map(AsRawFd::as_raw_fd))
            .flatten();
        wait_for_pipe_activity(output, input, deadline, self.label)
    }

    /// Kill the worker's original process group, including ordinary helpers.
    /// A malicious descendant may have escaped that group with `setsid(2)`;
    /// closing our pipe ends still guarantees bounded host return in that case.
    pub(crate) fn terminate_process_group(&mut self) {
        if self.process_group_terminated {
            return;
        }
        kill_process_group(self.child.id());
        self.process_group_terminated = true;
    }

    pub(crate) fn terminate(&mut self) {
        self.close_input();
        self.output.take();
        self.terminate_process_group();
        if self.status.is_none() {
            let _ = self.child.kill();
            if let Ok(status) = self.child.wait() {
                self.status = Some(status);
            }
        }
    }
}

impl Drop for SupervisedProcess {
    fn drop(&mut self) {
        self.terminate();
    }
}

fn set_nonblocking(stream: &impl AsRawFd) -> Result<()> {
    let descriptor = stream.as_raw_fd();
    // SAFETY: `descriptor` is owned by `stream` for both fcntl calls.
    let flags = unsafe { libc::fcntl(descriptor, libc::F_GETFL) };
    ensure!(
        flags >= 0,
        "failed to inspect protocol pipe flags: {}",
        std::io::Error::last_os_error()
    );
    // SAFETY: the same live descriptor is updated without changing ownership.
    let changed = unsafe { libc::fcntl(descriptor, libc::F_SETFL, flags | libc::O_NONBLOCK) };
    ensure!(
        changed == 0,
        "failed to configure a nonblocking protocol pipe: {}",
        std::io::Error::last_os_error()
    );
    Ok(())
}

fn wait_for_pipe_activity(
    output: Option<RawFd>,
    input: Option<RawFd>,
    deadline: Instant,
    label: &str,
) -> Result<()> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    let wait = remaining.min(MAX_POLL_INTERVAL);
    let timeout_millis = wait.as_millis().clamp(1, i32::MAX as u128) as i32;
    let mut descriptors = [
        libc::pollfd {
            fd: output.unwrap_or(-1),
            events: libc::POLLIN | libc::POLLHUP | libc::POLLERR,
            revents: 0,
        },
        libc::pollfd {
            fd: input.unwrap_or(-1),
            events: libc::POLLOUT | libc::POLLHUP | libc::POLLERR,
            revents: 0,
        },
    ];
    loop {
        // SAFETY: `descriptors` is live and has exactly the supplied length.
        let result = unsafe {
            libc::poll(
                descriptors.as_mut_ptr(),
                descriptors.len() as libc::nfds_t,
                timeout_millis,
            )
        };
        if result >= 0 {
            return Ok(());
        }
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::Interrupted {
            return Err(error).with_context(|| format!("waiting for {label} protocol activity"));
        }
        ensure!(
            Instant::now() < deadline,
            "{label} exceeded its execution deadline"
        );
    }
}

/// Report whether the leader has exited without reaping it, so its zombie
/// keeps the pid/pgid reserved until after the group is killed.
fn leader_exited_unreaped(child: &Child, label: &str) -> Result<bool> {
    let pid = child.id() as libc::id_t;
    loop {
        // SAFETY: `info` is a plain-data out-parameter zeroed before every
        // attempt; WNOWAIT leaves the child owned by `child` for later reaping.
        let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
        let rc = unsafe {
            libc::waitid(
                libc::P_PID,
                pid,
                &mut info,
                libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
            )
        };
        if rc == 0 {
            return Ok(siginfo_pid(&info) != 0);
        }
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::Interrupted {
            return Err(error).with_context(|| format!("peeking exit of {label}"));
        }
    }
}

#[cfg(target_os = "linux")]
fn siginfo_pid(info: &libc::siginfo_t) -> libc::pid_t {
    // SAFETY: the caller obtained `info` from a successful waitid(WEXITED)
    // call, where the pid union member is valid.
    unsafe { info.si_pid() }
}

#[cfg(not(target_os = "linux"))]
fn siginfo_pid(info: &libc::siginfo_t) -> libc::pid_t {
    info.si_pid
}

fn kill_process_group(child_id: u32) {
    if let Ok(group) = i32::try_from(child_id) {
        // SAFETY: callers place `child_id` in a fresh process group at spawn.
        unsafe {
            libc::kill(-group, libc::SIGKILL);
        }
    }
}

fn terminate_child(child: &mut Child) {
    kill_process_group(child.id());
    let _ = child.kill();
    let _ = child.wait();
}
