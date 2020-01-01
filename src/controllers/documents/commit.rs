use actix_web::{http, web, HttpResponse};

use crate::errors::Error;
use crate::search_engine::SearchEngine;


pub async fn commit(
    path: web::Path<(String,)>,
    search_engine: web::Data<SearchEngine>,
) -> Result<HttpResponse, Error> {
    let schema_name = path.0.0.as_str();
    search_engine.commit(schema_name).await?;
    Ok(HttpResponse::Ok()
        .set_header(http::header::CONTENT_TYPE, "application/json")
        .body("{\"status\": \"ok\"}"))
}
