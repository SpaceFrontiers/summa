use actix_service::{Service, Transform};
use actix_web::error::ErrorBadRequest;
use actix_web::http::header::{HeaderName, HeaderValue};
use actix_web::Result;
use actix_web::{dev, Error, FromRequest, HttpMessage, HttpRequest};
use actix_web::{dev::ServiceRequest, dev::ServiceResponse};
use futures::future::{err, ok, Ready};
use std::task::{Context, Poll};

/// The header set by the middleware
pub const REQUEST_ID_HEADER: &str = "request-id";

/// Request ID wrapper.
pub struct RequestIDWrapper;

impl<S, B> Transform<S> for RequestIDWrapper
where
    S: Service<Request = ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Request = ServiceRequest;
    type Response = ServiceResponse<B>;
    type Error = Error;
    type InitError = ();
    type Transform = RequestIDMiddleware<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ok(RequestIDMiddleware { service })
    }
}

/// Actual actix-web middleware
pub struct RequestIDMiddleware<S> {
    service: S,
}

impl<S, B> Service for RequestIDMiddleware<S>
where
    S: Service<Request = ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Request = ServiceRequest;
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Future = S::Future;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.service.poll_ready(cx)
    }

    fn call(&mut self, req: ServiceRequest) -> Self::Future {
        use rand::{distributions::Alphanumeric, thread_rng, Rng};
        let request_id: String = if req.headers().contains_key("request-id") {
            req.headers()
                .get("request-id")
                .unwrap()
                .to_str()
                .unwrap()
                .to_owned()
        } else {
            thread_rng().sample_iter(&Alphanumeric).take(12).collect()
        };
        let mut req = req;
        req.headers_mut().append(
            HeaderName::from_static(REQUEST_ID_HEADER),
            HeaderValue::from_str(&request_id).unwrap(),
        );
        req.extensions_mut().insert(RequestID(request_id));
        self.service.call(req)
    }
}

/// Request ID extractor
pub struct RequestID(pub String);

impl FromRequest for RequestID {
    type Error = Error;
    type Future = Ready<Result<Self, Self::Error>>;
    type Config = ();

    fn from_request(req: &HttpRequest, _payload: &mut dev::Payload) -> Self::Future {
        if let Some(RequestID(req_id)) = req.extensions().get::<RequestID>() {
            ok(RequestID(req_id.clone()))
        } else {
            err(ErrorBadRequest("request id is missing"))
        }
    }
}
