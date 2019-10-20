use actix_service::Service;
use actix_web::{web, App, HttpServer};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::time::timeout;

use summa::config::Config as ApplicationConfig;
use summa::errors::Error;
use summa::logging::{create_logger, Logging};
use summa::request_id::RequestIDWrapper;
use summa::SearchEngine;

fn main() -> Result<(), Error> {
    let matches = clap::App::new("Summa")
        .version("0.1.0")
        .author("Pasha Podolsky")
        .about("High-efficient full-text search server with handy replication")
        .arg(
            clap::Arg::with_name("config")
                .short("c")
                .long("config")
                .value_name("FILE")
                .help("Sets a custom config file")
                .takes_value(true),
        )
        .arg(
            clap::Arg::with_name("v")
                .short("v")
                .multiple(true)
                .help("Sets the level of verbosity"),
        )
        .get_matches();

    let config_file = String::from(matches.value_of("config").unwrap_or("config.yaml"));

    actix_rt::System::new("<name>").block_on(async move {
        let config = ApplicationConfig::from_reader(File::open(config_file)?)?;
        println!("\n{}", config);

        let http_config = config.http.clone();

        let error_logger = create_logger(&Path::new(&config.log_path).join("error.log"))?;
        let request_logger = create_logger(&Path::new(&config.log_path).join("request.log"))?;

        let search_engine =
            SearchEngine::new(&config.search_engine, PathBuf::from(&config.log_path))?;

        Ok(HttpServer::new(move || {
            let search_engine_timeout_secs = Duration::from_secs(config.search_engine.timeout_secs);
            App::new()
                .wrap(Logging::new(request_logger.clone(), error_logger.clone()))
                .wrap(RequestIDWrapper)
                .wrap_fn(move |req, srv| {
                    let fut = srv.call(req);
                    let search_engine_timeout_secs = search_engine_timeout_secs.clone();
                    async move {
                        timeout(search_engine_timeout_secs, fut)
                            .await
                            .map_err(|_e| Error::TimeoutError)?
                    }
                })
                .app_data(web::PayloadConfig::new(
                    config.http.max_body_size_mb * 1024 * 1024,
                ))
                .data(config.clone())
                .data(search_engine.clone())
                .service(
                    web::resource("/v1/{schema_name}/search/")
                        .route(web::get().to(summa::controllers::search::search)),
                )
        })
        .keep_alive(http_config.keep_alive_secs)
        .workers(http_config.workers)
        .bind(&http_config.bind_addr).map_err(
            |e| std::io::Error::new(e.kind(), format!("cannot bind to {}", &http_config.bind_addr))
        )?
        .run()
        .await?)
    })
}
