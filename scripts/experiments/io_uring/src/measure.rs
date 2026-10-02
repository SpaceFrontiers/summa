//! Linux process and mapping measurements shared by both diagnostic binaries.
pub fn usage() -> libc::rusage {
    let mut result = unsafe { std::mem::zeroed() };
    assert_eq!(
        unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut result) },
        0
    );
    result
}

pub fn cpu(u: &libc::rusage) -> f64 {
    (u.ru_utime.tv_sec + u.ru_stime.tv_sec) as f64
        + (u.ru_utime.tv_usec + u.ru_stime.tv_usec) as f64 / 1e6
}

pub fn read_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/io")
        .unwrap()
        .lines()
        .find_map(|line| {
            line.strip_prefix("read_bytes: ")
                .map(|n| n.parse().unwrap())
        })
        .unwrap()
}

pub fn residency(map: &memmap2::Mmap) -> usize {
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    assert!(page > 0);
    let page = page as usize;
    let mut state = vec![0u8; map.len().div_ceil(page)];
    assert_eq!(
        unsafe {
            libc::mincore(
                map.as_ptr().cast_mut().cast(),
                map.len(),
                state.as_mut_ptr(),
            )
        },
        0
    );
    state.iter().filter(|&&value| value & 1 != 0).count() * page
}
