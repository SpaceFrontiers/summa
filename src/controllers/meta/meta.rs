use actix_web::{web, HttpResponse};
use serde::Serialize;

use crate::errors::Error;
use crate::search_engine::SearchEngine;

#[derive(Serialize)]
pub struct MetaResponse {
    schemas: Vec<String>,
}

pub async fn meta(search_engine: web::Data<SearchEngine>) -> Result<HttpResponse, Error> {
    let meta_response = MetaResponse {
        schemas: search_engine.list_schemas(),
    };
    Ok(HttpResponse::Ok()
        .content_type("application/json")
        .body(serde_json::to_string(&meta_response)?))
}
