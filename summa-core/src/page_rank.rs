#[inline]
pub fn inverse_quantized_page_rank(val: u64) -> f64 {
    static VALS: [f64; 8] = [0.15495413, 0.16896642, 0.191641, 0.2273262, 0.28240761, 0.36859454, 0.53123659, 1.74778878];
    VALS[val as usize]
}

pub fn quantize_page_rank(val: f64) -> u64 {
    if val < 0.15990829 {
        0
    } else if val < 0.17802456 {
        1
    } else if val < 0.20525744 {
        2
    } else if val < 0.24939496 {
        3
    } else if val < 0.31542026 {
        4
    } else if val < 0.42176882 {
        5
    } else if val < 0.64070435 {
        6
    } else {
        7
    }
}
