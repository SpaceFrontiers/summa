use actix_web::{web, HttpRequest, HttpResponse};
use serde::Deserialize;

use crate::errors::{BadRequestError, Error};
use crate::search_engine::SearchEngine;

use super::super::get_header;

#[derive(Deserialize)]
pub struct Search {
    page: Option<usize>,
    page_size: Option<usize>,
    query: String,
}

pub async fn search(
    req: HttpRequest,
    path: web::Path<(String,)>,
    query_params: web::Query<Search>,
    search_engine: web::Data<SearchEngine>,
) -> Result<HttpResponse, Error> {
    let schema_name = path.0.0.as_str();
    let query = query_params.query.clone();

    let page = query_params.page.unwrap_or(0);
    let page_size = match query_params.page_size.unwrap_or(0) {
        0 => search_engine.get_config().default_page_size,
        x => x,
    };

    let limit = page_size;
    let offset = page_size * page;
    let search_response = search_engine.search(schema_name, &query, limit, offset).await?;
    match get_header(&req, "Accept") {
        Some("application/json") | Some("*/*") | None => {
            Ok(HttpResponse::Ok()
                .content_type("application/json")
                .body(search_response.to_json()?))
        },
        _ => Err(Error::BadRequestError(BadRequestError::UnknownContentTypeError))?
    }
}
