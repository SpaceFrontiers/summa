use actix_cors::Cors;
use actix_service::Service;
use actix_web::{web, App, HttpServer};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;
use tokio::time::timeout;

use summa::config::{Config as ApplicationConfig, CorsConfig};
use summa::errors::Error;
use summa::logging::{create_logger, Logging};
use summa::request_id::RequestIDWrapper;
use summa::SearchEngine;

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

    let config_file = String::from(matches.value_of("config").unwrap_or("summa.yaml"));

    actix_rt::System::new("<name>").block_on(async move {
        let config = ApplicationConfig::from_reader(
            File::open(config_file.clone()).map_err(|e| Error::FileError((e, config_file)))?,
        )?;
        println!("{}", config);

        let http_config = config.http.clone();
        let search_engine_config = config.search_engine.clone();
        let payload_config = web::PayloadConfig::new(
            config.http.max_body_size_mb * 1024 * 1024,
        );

        let error_logger = create_logger(&Path::new(&config.log_path).join("error.log"))?;
        let request_logger = create_logger(&Path::new(&config.log_path).join("request.log"))?;

        let search_engine =
            SearchEngine::new(&config.search_engine, PathBuf::from(&config.log_path))?;

        if config.search_engine.auto_commit {
            let auto_commit_handler = search_engine.start_auto_commit_thread();
            std::thread::spawn(move || {
                let term = Arc::new(AtomicBool::new(false));
                signal_hook::flag::register(
                    signal_hook::consts::signal::SIGTERM,
                    Arc::clone(&term),
                )
                .unwrap();
                while !term.load(Ordering::Relaxed) {
                    std::thread::park_timeout(std::time::Duration::from_secs(5));
                }
                auto_commit_handler.stop().unwrap().unwrap();
            });
        }

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
        .bind(&config.http.bind_addr)
        .map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("cannot bind to {}", &config.http.bind_addr),
            )
        })?
        .run()
        .await?)
    })
}
