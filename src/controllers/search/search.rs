use actix_web::{web, HttpRequest, HttpResponse};
use serde::Deserialize;

use crate::config::Config as ApplicationConfig;
use crate::errors::Error;
use crate::search_engine::SearchEngine;

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
    config: web::Data<ApplicationConfig>,
    search_engine: web::Data<SearchEngine>,
) -> Result<HttpResponse, Error> {
    let schema_name = path.0.0.as_str();
    let query = query_params.query.clone();

    let page = query_params.page.unwrap_or(0);
    let page_size = match query_params.page_size.unwrap_or(0) {
        0 => config.search_engine.default_page_size,
        x => x,
    };

    let limit = page_size;
    let offset = page_size * page;
    let results = search_engine.search(schema_name, &query, limit, offset)?;

    Ok(HttpResponse::Ok()
        .content_type("application/json")
        .body(serde_json::to_string(&results)?))
}
