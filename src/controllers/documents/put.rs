use actix_web::{http, web, HttpRequest, HttpResponse};

use crate::errors::{BadRequestError, Error};
use crate::search_engine::SearchEngine;


use super::super::get_header;

pub async fn put(
    req: HttpRequest,
    path: web::Path<(String,)>,
    body: web::Bytes,
    search_engine: web::Data<SearchEngine>,
) -> Result<HttpResponse, Error> {
    let schema_name = path.0.0.as_str();

    let document = match get_header(&req, "Content-Type") {
        Some("application/json") | Some("*/*") | None => {
            Ok(search_engine.get_schema(schema_name)?.parse_document(std::str::from_utf8(&body[..])?)?)
        },
        _ => Err(Error::BadRequestError(BadRequestError::UnknownContentTypeError))
    }?;

    search_engine.put_document(schema_name, document).await?;
    Ok(HttpResponse::Ok()
        .set_header(http::header::CONTENT_TYPE, "application/json")
        .body("{\"status\": \"ok\"}"))
}
