//! Linux-only diagnostic, never a production directory backend.
#[cfg(not(panic = "abort"))]
compile_error!("I/O diagnostics require panic=abort while kernel buffer pointers are live");
mod measure;
use io_uring::{IoUring, opcode, types};
use measure::{cpu, read_bytes, residency, usage};
use std::{
    fs::File,
    io::{self, Write},
    os::fd::AsRawFd,
    os::unix::fs::FileExt,
    path::Path,
    time::Instant,
};

const SIZE: usize = 2 * 1024 * 1024 * 1024;
const PAGE: usize = 4096;
const MAX_READ: usize = 65536;
const DEPTH: usize = 32;
const READS: usize = 4096;

#[repr(align(4096))]
struct Buffer([u8; MAX_READ]);

fn word(offset: usize) -> u64 {
    (offset as u64)
        .wrapping_mul(0x9e3779b97f4a7c15)
        .rotate_left(17)
}

fn create(path: &Path) -> io::Result<()> {
    let mut file = File::options().write(true).create_new(true).open(path)?;
    let mut buffer = vec![0; 1024 * 1024];
    for base in (0..SIZE).step_by(buffer.len()) {
        for (i, bytes) in buffer.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            bytes.copy_from_slice(&word(base + i * 8).to_le_bytes());
        }
        file.write_all(&buffer)?;
    }
    file.sync_all()
}

fn verify(bytes: &[u8], offset: usize) {
    for (i, bytes) in bytes.as_chunks::<8>().0.iter().enumerate() {
        assert_eq!(u64::from_le_bytes(*bytes), word(offset + i * 8));
    }
}

struct Reader {
    // Field drop order closes the ring before any registered memory/file.
    ring: IoUring,
    buffers: Vec<Box<Buffer>>,
    file: File,
    registered: bool,
}

impl Reader {
    fn new(path: &Path, registered: bool) -> io::Result<Self> {
        let ring = IoUring::new(64)?;
        let file = File::open(path)?;
        let buffers: Vec<_> = (0..DEPTH)
            .map(|_| Box::new(Buffer([0; MAX_READ])))
            .collect();
        if registered {
            ring.submitter().register_files(&[file.as_raw_fd()])?;
            let iovecs: Vec<_> = buffers
                .iter()
                .map(|b| libc::iovec {
                    iov_base: b.0.as_ptr().cast_mut().cast(),
                    iov_len: MAX_READ,
                })
                .collect();
            // Owners never move their allocations, and all CQEs are drained
            // before reusing a slot. Ring destruction precedes buffer release.
            unsafe {
                ring.submitter().register_buffers(&iovecs)?;
            }
        }
        Ok(Self {
            ring,
            buffers,
            file,
            registered,
        })
    }

    fn entry(&mut self, slot: usize, offset: usize, size: usize) -> io_uring::squeue::Entry {
        let pointer = self.buffers[slot].0.as_mut_ptr();
        if self.registered {
            opcode::ReadFixed::new(types::Fixed(0), pointer, size as u32, slot as u16)
                .offset(offset as u64)
                .build()
        } else {
            opcode::Read::new(types::Fd(self.file.as_raw_fd()), pointer, size as u32)
                .offset(offset as u64)
                .build()
        }
        .user_data(slot as u64)
    }

    fn cancellation_probe(&mut self) -> io::Result<(usize, usize)> {
        let mut cancelled = 0;
        let mut completed = 0;
        for round in 0..32 {
            assert_eq!(
                unsafe {
                    libc::posix_fadvise(self.file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED)
                },
                0
            );
            for slot in 0..8 {
                let entry = self
                    .entry(slot, (round * 8 + slot) * MAX_READ, MAX_READ)
                    .flags(io_uring::squeue::Flags::ASYNC);
                unsafe {
                    self.ring.submission().push(&entry).unwrap();
                }
            }
            for slot in 0..8 {
                let entry = opcode::AsyncCancel::new(slot as u64)
                    .build()
                    .user_data(32 + slot as u64);
                unsafe {
                    self.ring.submission().push(&entry).unwrap();
                }
            }
            let mut read_mask = 0u8;
            let mut cancel_mask = 0u8;
            while read_mask != 255 || cancel_mask != 255 {
                if let Err(e) = self.ring.submit_and_wait(1) {
                    if e.kind() == io::ErrorKind::Interrupted {
                        continue;
                    }
                    eprintln!("fatal cancel submission: {e}");
                    std::process::abort();
                }
                for cqe in self.ring.completion() {
                    let id = cqe.user_data() as usize;
                    if id >= 32 {
                        assert!(id < 40 && cancel_mask & (1 << (id - 32)) == 0);
                        cancel_mask |= 1 << (id - 32);
                        assert!([0, -libc::ENOENT, -libc::EALREADY].contains(&cqe.result()));
                    } else {
                        assert!(id < 8 && read_mask & (1 << id) == 0);
                        read_mask |= 1 << id;
                        match cqe.result() {
                            n if n == MAX_READ as i32 => {
                                verify(&self.buffers[id].0, (round * 8 + id) * MAX_READ);
                                completed += 1;
                            }
                            n if n == -libc::ECANCELED => {
                                cancelled += 1;
                            }
                            n => panic!("unexpected read completion: {n}"),
                        }
                    }
                }
            }
            // The original read CQE, not its cancel CQE, releases each slot.
            self.batch("uring", &[0, MAX_READ], MAX_READ)?;
            verify(&self.buffers[0].0, 0);
            verify(&self.buffers[1].0, MAX_READ);
        }
        Ok((cancelled, completed))
    }

    fn batch(&mut self, method: &str, offsets: &[usize], size: usize) -> io::Result<()> {
        if method == "pread" {
            for (buffer, &offset) in self.buffers.iter_mut().zip(offsets) {
                self.file
                    .read_exact_at(&mut buffer.0[..size], offset as u64)?;
            }
            return Ok(());
        }
        for (i, &offset) in offsets.iter().enumerate() {
            let entry = self.entry(i, offset, size);
            // All owners are held by Reader. No buffer is accessed until its
            // read CQE; even an error drains the other admitted operations.
            unsafe {
                self.ring.submission().push(&entry).unwrap();
            }
        }
        let mut complete = 0;
        let mut seen = 0u64;
        let mut error = None;
        while complete < offsets.len() {
            if let Err(e) = self.ring.submit_and_wait(1) {
                if e.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                eprintln!("fatal submission error with owned buffers: {e}");
                std::process::abort();
            }
            for entry in self.ring.completion() {
                let id = entry.user_data() as usize;
                assert!(id < offsets.len() && seen & (1 << id) == 0);
                seen |= 1 << id;
                complete += 1;
                if entry.result() != size as i32 && error.is_none() {
                    error = Some(if entry.result() < 0 {
                        io::Error::from_raw_os_error(-entry.result())
                    } else {
                        io::Error::new(io::ErrorKind::UnexpectedEof, "short read")
                    });
                }
            }
        }
        error.map_or(Ok(()), Err)
    }
}

type Job = (usize, usize, usize, Box<Buffer>);
type Done = (usize, Box<Buffer>, io::Result<()>);

/// Persistent bounded positional-read control: one slot per worker, with
/// owned buffers transferred through channels and no per-batch thread spawn.
struct Pool {
    senders: Vec<std::sync::mpsc::SyncSender<Job>>,
    done: std::sync::mpsc::Receiver<Done>,
    buffers: Vec<Option<Box<Buffer>>>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl Pool {
    fn new(path: &Path) -> io::Result<Self> {
        let (complete, done) = std::sync::mpsc::sync_channel(8);
        let mut senders = Vec::new();
        let mut workers = Vec::new();
        for _ in 0..8 {
            let file = File::open(path)?;
            let (sender, receiver) = std::sync::mpsc::sync_channel::<Job>(1);
            let complete = complete.clone();
            workers.push(std::thread::spawn(move || {
                while let Ok((slot, offset, size, mut buffer)) = receiver.recv() {
                    let result = file.read_exact_at(&mut buffer.0[..size], offset as u64);
                    if complete.send((slot, buffer, result)).is_err() {
                        break;
                    }
                }
            }));
            senders.push(sender);
        }
        Ok(Self {
            senders,
            done,
            workers,
            buffers: (0..8)
                .map(|_| Some(Box::new(Buffer([0; MAX_READ]))))
                .collect(),
        })
    }
    fn batch(&mut self, offsets: &[usize], size: usize) -> io::Result<()> {
        assert!(offsets.len() <= 8);
        for (slot, &offset) in offsets.iter().enumerate() {
            self.senders[slot]
                .send((slot, offset, size, self.buffers[slot].take().unwrap()))
                .unwrap();
        }
        let mut error = None;
        for _ in offsets {
            let (slot, buffer, result) = self.done.recv().unwrap();
            self.buffers[slot] = Some(buffer);
            if let Err(e) = result {
                error.get_or_insert(e);
            }
        }
        error.map_or(Ok(()), Err)
    }
}
impl Drop for Pool {
    fn drop(&mut self) {
        self.senders.clear();
        for worker in self.workers.drain(..) {
            worker.join().unwrap();
        }
    }
}

fn run(path: &Path, followup: bool) -> io::Result<()> {
    let file = File::open(path)?;
    assert_eq!(file.metadata()?.len(), SIZE as u64);
    let map = unsafe { memmap2::Mmap::map(&file)? };
    map.advise(if followup {
        memmap2::Advice::Normal
    } else {
        memmap2::Advice::Random
    })?;
    assert_eq!(
        unsafe {
            libc::posix_fadvise(
                file.as_raw_fd(),
                0,
                0,
                if followup {
                    libc::POSIX_FADV_NORMAL
                } else {
                    libc::POSIX_FADV_RANDOM
                },
            )
        },
        0
    );
    // Distinct, deterministic page offsets. Fixed order across every method.
    let offsets: Vec<_> = (0..READS)
        .map(|i| ((i * 104729 + 29) % (SIZE / MAX_READ)) * MAX_READ)
        .collect();
    let mut ordinary = Reader::new(path, false)?;
    let mut registered = Reader::new(path, true)?;
    let mut pool = if followup {
        Some(Pool::new(path)?)
    } else {
        None
    };
    // Failure boundary: one EOF with valid neighboring reads must drain, and
    // the next use of every slot must still return the right immutable bytes.
    for reader in [&mut ordinary, &mut registered] {
        assert_eq!(
            reader
                .batch("uring", &[0, SIZE, MAX_READ], PAGE)
                .unwrap_err()
                .kind(),
            io::ErrorKind::UnexpectedEof
        );
        reader.batch("uring", &offsets[..DEPTH], MAX_READ)?;
        for (buffer, &offset) in reader.buffers.iter().zip(&offsets) {
            verify(&buffer.0, offset);
        }
    }
    for size in [PAGE, MAX_READ] {
        for depth in if followup { &[8][..] } else { &[1, 8, 32][..] } {
            let depth = *depth;
            for cold in [false, true] {
                for pass in 0..2 {
                    let mut methods = if followup {
                        vec!["mmap_default", "pread_pool", "uring", "registered"]
                    } else {
                        vec!["mmap", "pread", "uring", "registered"]
                    };
                    if pass == 1 {
                        methods.reverse();
                    }
                    for method in methods {
                        // Drop this mapping's PTEs before asking the kernel to
                        // evict this private fixture. Verify residency below.
                        if cold {
                            unsafe {
                                map.unchecked_advise(memmap2::UncheckedAdvice::DontNeed)?;
                            }
                            assert_eq!(
                                unsafe {
                                    libc::posix_fadvise(
                                        file.as_raw_fd(),
                                        0,
                                        0,
                                        libc::POSIX_FADV_DONTNEED,
                                    )
                                },
                                0
                            );
                        } else {
                            for &offset in &offsets {
                                ordinary.file.read_exact_at(
                                    &mut ordinary.buffers[0].0[..size],
                                    offset as u64,
                                )?;
                            }
                        }
                        let resident = residency(&map);
                        let reader = if method == "registered" {
                            &mut registered
                        } else {
                            &mut ordinary
                        };
                        let before = usage();
                        let io_before = read_bytes();
                        let mut samples = Vec::with_capacity(READS / depth);
                        let mut verified = 0;
                        for batch in offsets.chunks(depth) {
                            let start = Instant::now();
                            if method.starts_with("mmap") {
                                for (buffer, &offset) in reader.buffers.iter_mut().zip(batch) {
                                    buffer.0[..size].copy_from_slice(&map[offset..offset + size]);
                                }
                            } else if method == "pread_pool" {
                                pool.as_mut().unwrap().batch(batch, size)?;
                            } else {
                                reader.batch(method, batch, size)?;
                            }
                            samples.push(start.elapsed().as_nanos() as u64);
                            // Complete byte verification is outside I/O timing.
                            for (slot, &offset) in batch.iter().enumerate() {
                                let buffer = if method == "pread_pool" {
                                    pool.as_ref().unwrap().buffers[slot].as_ref().unwrap()
                                } else {
                                    &reader.buffers[slot]
                                };
                                verify(&buffer.0[..size], offset);
                                verified += 1;
                            }
                        }
                        let after = usage();
                        println!(
                            "{}",
                            serde_json::json!({"method":method,"cold_advised":cold,"pass":pass,
                            "size":size,"depth":depth,"reads":READS,"verified":verified,"resident_bytes_before":resident,
                            "physical_read_bytes":read_bytes()-io_before,"cpu_seconds_including_verification":cpu(&after)-cpu(&before),
                            "major_faults":after.ru_majflt-before.ru_majflt,"minor_faults":after.ru_minflt-before.ru_minflt,
                            "peak_rss_kib":after.ru_maxrss,"buffer_bytes_per_reader":DEPTH*MAX_READ,"samples_ns":samples})
                        );
                    }
                }
            }
        }
    }
    println!(
        "{}",
        serde_json::json!({"complete":true,"short_read_drain":true})
    );
    Ok(())
}

fn lifecycle(path: &Path) -> io::Result<()> {
    let alias = path.with_extension(format!("retired-{}", std::process::id()));
    std::fs::hard_link(path, &alias)?;
    let mut ordinary = Reader::new(&alias, false)?;
    let mut registered = Reader::new(&alias, true)?;
    std::fs::remove_file(&alias)?;
    let ordinary = ordinary.cancellation_probe()?;
    let registered = registered.cancellation_probe()?;
    println!(
        "{}",
        serde_json::json!({"complete":true,"unlinked_file_reads":true,
        "ordinary_cancelled":ordinary.0,"ordinary_completed":ordinary.1,
        "registered_cancelled":registered.0,"registered_completed":registered.1,
        "slot_reuse_verified":true})
    );
    Ok(())
}

fn main() -> io::Result<()> {
    let args: Vec<_> = std::env::args_os().collect();
    assert_eq!(args.len(), 3, "probe FIXTURE create|run");
    match args[2].to_str().unwrap() {
        "create" => create(Path::new(&args[1])),
        "run" => run(Path::new(&args[1]), false),
        "followup" => run(Path::new(&args[1]), true),
        "lifecycle" => lifecycle(Path::new(&args[1])),
        _ => panic!("expected create or run"),
    }
}
