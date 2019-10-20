extern crate libc;
use actix_service::{Service, Transform};
use actix_web::{dev::ServiceRequest, dev::ServiceResponse, http::header::HeaderValue, Error};
use futures::future::{ok, Ready};
use futures::Future;
use reopen::Reopen;
use slog::{error, info, Drain, PushFnValue, Record};
use std::fs::OpenOptions;
use std::fs::Permissions;
use std::os::unix::fs::PermissionsExt;
use std::pin::Pin;
use std::sync::Mutex;
use std::task::{Context, Poll};
use std::time::Instant;

pub struct Logging {
    logger: slog::Logger,
    error_logger: slog::Logger,
}

impl Logging {
    pub fn new(logger: slog::Logger, error_logger: slog::Logger) -> Logging {
        Logging {
            logger,
            error_logger,
        }
    }
}

impl<S, B> Transform<S> for Logging
where
    S: Service<Request = ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Request = ServiceRequest;
    type Response = ServiceResponse<B>;
    type Error = Error;
    type InitError = ();
    type Transform = LoggingMiddleware<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ok(LoggingMiddleware {
            service,
            logger: self.logger.clone(),
            error_logger: self.error_logger.clone(),
        })
    }
}

pub struct LoggingMiddleware<S> {
    service: S,
    logger: slog::Logger,
    error_logger: slog::Logger,
}

impl<S, B> Service for LoggingMiddleware<S>
where
    S: Service<Request = ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Request = ServiceRequest;
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>>>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.service.poll_ready(cx)
    }

    fn call(&mut self, req: ServiceRequest) -> Self::Future {
        let start_time = Instant::now();
        let logger = self.logger.clone();
        let error_logger = self.error_logger.clone();

        let fut = self.service.call(req);
        Box::pin(async move {
            let res = fut.await;
            let duration = start_time.elapsed().as_secs_f32();
            match res {
                Ok(r) => {
                    if r.request().method() == "GET" {
                        info!(logger, "handled request";
                            "method" => %r.request().method(),
                            "mode" => "request",
                            "response_time" => duration,
                            "request_id" => r.request().headers().get("request-id").map(HeaderValue::to_str).unwrap_or(Ok("")).unwrap(),
                            "route" => r.request().path(),
                            "status_code" => r.status().as_u16(),
                            "url" => %r.request().uri(),
                        );
                    }
                    Ok(r)
                }
                Err(e) => {
                    error!(error_logger, "error";
                        "error" => format!("{:?}", e),
                        "mode" => "unhandled_error",
                    );
                    Err(e)
                }
            }
        })
    }
}

pub fn create_logger(filepath: &std::path::Path) -> Result<slog::Logger, crate::errors::Error> {
    let cloned_filepath = filepath.to_owned();
    let log_file = Reopen::new(Box::new(move || {
        std::fs::create_dir_all(&cloned_filepath.parent().unwrap())?;
        let log_file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(&cloned_filepath);
        let permissions = Permissions::from_mode(0o644);

        std::fs::set_permissions(&cloned_filepath, permissions)?;
        log_file
    })).map_err(
        |e| std::io::Error::new(
            e.kind(),
            format!("cannot create logfile {}", &filepath.to_str().unwrap())
        )
    )?;

    log_file.handle().register_signal(libc::SIGHUP)?;

    let json_drain = Mutex::new(
        slog_json::Json::new(log_file)
            .add_key_value(o!(
                "unixtime" => PushFnValue(move |_ : &Record, ser| {
                    ser.emit(chrono::Local::now().timestamp())
                }),
                "datetime" => PushFnValue(move |_ : &Record, ser| {
                    ser.emit(format!("{}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S.%f")))
                })
            ))
            .set_flush(true)
            .build(),
    )
    .map(slog::Fuse);
    let async_drain = Mutex::new(slog_async::Async::new(json_drain).build()).map(slog::Fuse);
    Ok(slog::Logger::root(async_drain, o!()))
}
