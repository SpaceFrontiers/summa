pub mod documents;
pub mod search;

use actix_web::http::header::HeaderValue;
use actix_web::HttpRequest;

pub(crate) fn get_header<'a>(req: &'a HttpRequest, header_name: &'a str) -> Option<&'a str> {
    req.headers()
        .get(header_name)
        .map(HeaderValue::to_str)
        .unwrap_or(Ok(""))
        .ok()
}
