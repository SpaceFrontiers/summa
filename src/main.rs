#[macro_use]
extern crate slog;

use actix_cors::Cors;
use actix_service::Service;
use actix_web::{web, App, HttpServer};
use std::path::Path;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;
use tokio::time::timeout;

use summa::config::{Config as ApplicationConfig, CorsConfig};
use summa::errors::Error;
use summa::logging::{create_logger, create_term_logger, Logging};
use summa::request_id::RequestIDWrapper;
use summa::SearchEngine;
use summa::ThreadHandler;

fn configure_cors(cors_config: &CorsConfig) -> actix_cors::Cors {
    let allowed_headers: Vec<actix_web::http::HeaderName> = cors_config
        .allowed_headers
        .iter()
        .map(|header| actix_web::http::HeaderName::from_bytes(header.as_bytes()).unwrap())
        .collect();
    let allowed_methods: Vec<actix_web::http::Method> = cors_config
        .allowed_methods
        .iter()
        .map(|method| actix_web::http::Method::from_bytes(method.as_bytes()).unwrap())
        .collect();
    let mut cors = Cors::default()
        .allowed_methods(allowed_methods)
        .allowed_headers(allowed_headers)
        .max_age(3600);
    for origin in cors_config.allowed_origins.iter() {
        if origin == "*" {
            cors = cors.allow_any_origin();
            cors = cors.send_wildcard();
            break;
        } else {
            cors = cors.allowed_origin(origin);
        }
    }
    cors
}

fn main() -> Result<(), Error> {
    env_logger::init();

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

    let config_filepath = String::from(matches.value_of("config").unwrap_or("/summa/summa.yaml"));
    let config = ApplicationConfig::from_file(&config_filepath)?;
    println!("{}", config);
    let http_config = config.http.clone();
    let search_engine_config = config.search_engine.clone();
    let payload_config = web::PayloadConfig::new(config.http.max_body_size_mb * 1024 * 1024);

    let error_logger = create_logger(&Path::new(&config.log_path).join("error.log"))?;
    let logger = match config.debug {
        true => create_term_logger(),
        false => create_logger(&Path::new(&config.log_path).join("statbox.log"))
    }?;
    let request_logger = match config.debug {
        true => create_term_logger(),
        false => create_logger(&Path::new(&config.log_path).join("request.log")),
    }?;

    let search_engine = SearchEngine::new(&config.search_engine, logger.clone())?;

    let master_auto_commit_thread_handler = if config.search_engine.auto_commit {
        let timeout = Duration::from_secs(60);
        let wakeup_timeout = Duration::from_secs(5);

        let auto_commit_handlers = search_engine.start_auto_commit_threads(timeout, wakeup_timeout);
        let running = Arc::new(AtomicBool::new(true));
        let running_in_thread = running.clone();

        let join_handle = Box::new(std::thread::spawn(
            move || -> Result<(), summa::errors::Error> {
                info!(logger, "start";
                    "action" => "start",
                    "mode" => "master_auto_commit_thread"
                );
                while running_in_thread.load(Ordering::Relaxed) {
                    std::thread::park_timeout(std::time::Duration::from_secs(1));
                }
                info!(logger, "sigterm";
                    "action" => "sigterm",
                    "mode" => "master_auto_commit_thread"
                );
                for auto_commit_handler in auto_commit_handlers {
                    auto_commit_handler.stop().unwrap().unwrap();
                }
                Ok(())
            },
        ));
        Some(ThreadHandler::new(join_handle, running))
    } else {
        None
    };

    let rval = actix_rt::System::new("<name>").block_on(async move {
        Ok(HttpServer::new(move || {
            let search_engine_timeout_secs = Duration::from_secs(search_engine_config.timeout_secs);
            let cors = configure_cors(&http_config.cors);
            App::new()
                .wrap(cors)
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
                .app_data(payload_config.clone())
                .data(search_engine.clone())
                .service(
                    web::resource("/v1/meta")
                        .route(web::get().to(summa::controllers::meta::meta)),
                )
                .service(
                    web::resource("/v1/{schema_name}/search/")
                        .route(web::get().to(summa::controllers::search::search)),
                )
                .service(
                    web::resource("/v1/{schema_name}/commit/")
                        .route(web::post().to(summa::controllers::documents::commit)),
                )
                .service(
                    web::resource("/v1/{schema_name}/")
                        .route(web::put().to(summa::controllers::documents::put)),
                )
        })
        .keep_alive(config.http.keep_alive_secs)
        .workers(config.http.workers)
        .bind(&format!("{}:{}", &config.http.address, config.http.port))
        .map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!(
                    "cannot bind to {}:{}",
                    &config.http.address, config.http.port
                ),
            )
        })?
        .run()
        .await?)
    });
    if let Some(master_auto_commit_thread_handler) = master_auto_commit_thread_handler {
        master_auto_commit_thread_handler.stop().unwrap().unwrap();
    };
    rval
}
