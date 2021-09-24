use std::io;
use std::io::Write;
use tantivy::directory::{AntiCallToken, TerminatingWrite};

pub struct Noop {}
impl Write for Noop {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
impl TerminatingWrite for Noop {
    fn terminate_ref(&mut self, _: AntiCallToken) -> io::Result<()> {
        Ok(())
    }
}
