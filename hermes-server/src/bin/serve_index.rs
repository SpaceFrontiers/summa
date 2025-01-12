//! Simple HTTP server for serving test index files
//!
//! Run with: cargo run --bin serve_test_index

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;

fn main() -> std::io::Result<()> {
    let index_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("tests/fixtures/test_index"));

    let port: u16 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8765);

    println!(
        "Serving index from {:?} on http://localhost:{}",
        index_dir, port
    );

    let listener = TcpListener::bind(format!("127.0.0.1:{}", port))?;

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let dir = index_dir.clone();
                std::thread::spawn(move || {
                    if let Err(e) = handle_request(stream, &dir) {
                        eprintln!("Error handling request: {}", e);
                    }
                });
            }
            Err(e) => eprintln!("Connection failed: {}", e),
        }
    }

    Ok(())
}

fn handle_request(mut stream: TcpStream, index_dir: &PathBuf) -> std::io::Result<()> {
    let mut buffer = [0; 4096];
    let n = stream.read(&mut buffer)?;
    let request = String::from_utf8_lossy(&buffer[..n]);

    // Parse request line
    let request_line = request.lines().next().unwrap_or("");
    let parts: Vec<&str> = request_line.split_whitespace().collect();

    if parts.len() < 2 {
        return send_response(&mut stream, 400, "Bad Request", b"Bad Request");
    }

    let method = parts[0];
    let path = parts[1];

    // Parse Range header if present
    let range = parse_range_header(&request);

    if let Some((start, end)) = range {
        println!(
            "[{}] {} {} [Range: {}-{}]",
            chrono_lite(),
            method,
            path,
            start,
            end
        );
    } else {
        println!("[{}] {} {}", chrono_lite(), method, path);
    }

    if method != "GET" && method != "HEAD" {
        return send_response(
            &mut stream,
            405,
            "Method Not Allowed",
            b"Method Not Allowed",
        );
    }

    // Remove leading slash and decode
    let file_path = path.trim_start_matches('/');
    let full_path = index_dir.join(file_path);

    // Security: ensure path is within index_dir
    if !full_path.starts_with(index_dir) {
        return send_response(&mut stream, 403, "Forbidden", b"Forbidden");
    }

    // For HEAD requests, just return headers with Content-Length
    if method == "HEAD" {
        match std::fs::metadata(&full_path) {
            Ok(metadata) => {
                return send_head_response(&mut stream, metadata.len());
            }
            Err(_) => return send_response(&mut stream, 404, "Not Found", b"Not Found"),
        }
    }

    // Read and serve file (with optional range support)
    match std::fs::read(&full_path) {
        Ok(contents) => {
            let content_type = guess_content_type(&full_path);

            if let Some((start, end)) = range {
                // Serve partial content
                let start = start as usize;
                let end = (end as usize + 1).min(contents.len()); // HTTP range is inclusive
                if start >= contents.len() {
                    return send_response(
                        &mut stream,
                        416,
                        "Range Not Satisfiable",
                        b"Range Not Satisfiable",
                    );
                }
                let partial = &contents[start..end];
                send_partial_response(
                    &mut stream,
                    partial,
                    content_type,
                    start,
                    end - 1,
                    contents.len(),
                )
            } else {
                send_response_with_type(&mut stream, 200, "OK", &contents, content_type)
            }
        }
        Err(_) => send_response(&mut stream, 404, "Not Found", b"Not Found"),
    }
}

/// Parse Range header: "Range: bytes=start-end"
fn parse_range_header(request: &str) -> Option<(u64, u64)> {
    for line in request.lines() {
        let line_lower = line.to_lowercase();
        if line_lower.starts_with("range:") {
            // Format: "Range: bytes=start-end"
            let value = line[6..].trim();
            if let Some(bytes_part) = value.strip_prefix("bytes=") {
                let parts: Vec<&str> = bytes_part.split('-').collect();
                if parts.len() == 2 {
                    let start = parts[0].parse::<u64>().ok()?;
                    let end = parts[1].parse::<u64>().ok()?;
                    return Some((start, end));
                }
            }
        }
    }
    None
}

fn send_response(
    stream: &mut TcpStream,
    status: u16,
    status_text: &str,
    body: &[u8],
) -> std::io::Result<()> {
    send_response_with_type(stream, status, status_text, body, "text/plain")
}

fn send_response_with_type(
    stream: &mut TcpStream,
    status: u16,
    status_text: &str,
    body: &[u8],
    content_type: &str,
) -> std::io::Result<()> {
    let response = format!(
        "HTTP/1.1 {} {}\r\n\
         Content-Type: {}\r\n\
         Content-Length: {}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Headers: Range\r\n\
         Access-Control-Expose-Headers: Content-Range, Accept-Ranges\r\n\
         Accept-Ranges: bytes\r\n\
         Cache-Control: no-cache, no-store, must-revalidate\r\n\
         Pragma: no-cache\r\n\
         Expires: 0\r\n\
         Connection: close\r\n\
         \r\n",
        status,
        status_text,
        content_type,
        body.len()
    );

    stream.write_all(response.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

fn send_head_response(stream: &mut TcpStream, content_length: u64) -> std::io::Result<()> {
    let response = format!(
        "HTTP/1.1 200 OK\r\n\
         Content-Type: application/octet-stream\r\n\
         Content-Length: {}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Headers: Range\r\n\
         Access-Control-Expose-Headers: Content-Range, Accept-Ranges, Content-Length\r\n\
         Accept-Ranges: bytes\r\n\
         Connection: close\r\n\
         \r\n",
        content_length
    );

    stream.write_all(response.as_bytes())?;
    stream.flush()
}

fn send_partial_response(
    stream: &mut TcpStream,
    body: &[u8],
    content_type: &str,
    range_start: usize,
    range_end: usize,
    total_size: usize,
) -> std::io::Result<()> {
    let response = format!(
        "HTTP/1.1 206 Partial Content\r\n\
         Content-Type: {}\r\n\
         Content-Length: {}\r\n\
         Content-Range: bytes {}-{}/{}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Headers: Range\r\n\
         Access-Control-Expose-Headers: Content-Range, Accept-Ranges\r\n\
         Accept-Ranges: bytes\r\n\
         Cache-Control: no-cache, no-store, must-revalidate\r\n\
         Connection: close\r\n\
         \r\n",
        content_type,
        body.len(),
        range_start,
        range_end,
        total_size
    );

    stream.write_all(response.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

fn guess_content_type(path: &PathBuf) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("json") => "application/json",
        Some("bin") => "application/octet-stream",
        _ => "application/octet-stream",
    }
}

fn chrono_lite() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let secs = duration.as_secs();
    let hours = (secs / 3600) % 24;
    let mins = (secs / 60) % 60;
    let secs = secs % 60;
    format!("{:02}:{:02}:{:02}", hours, mins, secs)
}
