//! Loopback-only live server for the model visualization lab.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::mpsc::{SyncSender, TrySendError, sync_channel};

use anyhow::{Context, Result, ensure};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Redirect, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;
use tower_http::services::ServeDir;

use crate::generate::SamplingConfig;
use crate::trace::{TraceGeneration, TraceOptions, TraceRequest, VisualizationBundle};
use crate::{Device, TextGenerator, Tokenizer, Transformer, capture_bundle};

const DEFAULT_MAX_TOKENS: usize = 32;
const DEFAULT_TEMPERATURE: f64 = 0.9;
const DEFAULT_REPETITION_PENALTY: f64 = 1.05;

#[derive(Debug, Clone)]
pub struct LabServerConfig {
    pub bind: SocketAddr,
    pub allow_remote: bool,
    pub web_root: PathBuf,
    pub metrics_path: Option<PathBuf>,
    pub trace_options: TraceOptions,
    pub max_new_tokens: usize,
    pub max_prompt_bytes: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LiveTraceRequest {
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f64,
    #[serde(default)]
    pub seed: Option<u64>,
}

fn default_max_tokens() -> usize {
    DEFAULT_MAX_TOKENS
}

fn default_temperature() -> f64 {
    DEFAULT_TEMPERATURE
}

fn default_repetition_penalty() -> f64 {
    DEFAULT_REPETITION_PENALTY
}

#[derive(Clone, Serialize)]
struct LabStatus {
    status: &'static str,
    model: String,
    num_layers: usize,
    hidden_size: usize,
    vocab_size: usize,
    max_new_tokens: usize,
    max_prompt_bytes: usize,
}

#[derive(Serialize)]
struct ApiError {
    error: String,
}

enum TraceFailure {
    Invalid(String),
    Internal(String),
}

struct TraceJob {
    request: LiveTraceRequest,
    response: oneshot::Sender<std::result::Result<VisualizationBundle, TraceFailure>>,
}

#[derive(Clone)]
struct AppState {
    jobs: SyncSender<TraceJob>,
    status: LabStatus,
}

struct InferenceWorker {
    model: Transformer,
    device: Device,
    tokenizer: Tokenizer,
    metrics_path: Option<PathBuf>,
    trace_options: TraceOptions,
}

impl InferenceWorker {
    fn trace(
        &self,
        request: LiveTraceRequest,
    ) -> std::result::Result<VisualizationBundle, TraceFailure> {
        let prompt_tokens = self
            .tokenizer
            .encode(&request.prompt, false)
            .map_err(|error| {
                TraceFailure::Invalid(format!("prompt tokenization failed: {error}"))
            })?;
        if prompt_tokens.is_empty() {
            return Err(TraceFailure::Invalid(
                "prompt encodes to zero tokens".to_owned(),
            ));
        }
        let seed = request.seed.unwrap_or_else(rand::random);
        let sampling = SamplingConfig {
            max_new_tokens: request.max_tokens,
            temperature: request.temperature,
            top_k: request.top_k,
            repetition_penalty: request.repetition_penalty,
            eos_token: Some(self.tokenizer.eos_token_id()),
            seed: Some(seed),
        };
        let output_tokens = TextGenerator::new(&self.model, &self.device)
            .generate(&prompt_tokens, &sampling)
            .map_err(|error| TraceFailure::Invalid(format!("generation failed: {error}")))?;
        capture_bundle(
            &self.model,
            &self.tokenizer,
            TraceRequest {
                prompt: &request.prompt,
                prompt_token_count: prompt_tokens.len(),
                output_tokens: &output_tokens,
                generation: TraceGeneration {
                    max_new_tokens: request.max_tokens,
                    temperature: request.temperature,
                    top_k: request.top_k,
                    repetition_penalty: request.repetition_penalty,
                    seed,
                    stop_at_eos: true,
                },
                metrics_path: self.metrics_path.as_deref(),
            },
            &self.trace_options,
        )
        .map_err(|error| TraceFailure::Internal(format!("trace capture failed: {error:#}")))
    }
}

fn validate_request(request: &LiveTraceRequest, status: &LabStatus) -> Result<()> {
    ensure!(
        !request.prompt.trim().is_empty(),
        "prompt must not be blank"
    );
    ensure!(
        request.prompt.len() <= status.max_prompt_bytes,
        "prompt is {} bytes; server limit is {}",
        request.prompt.len(),
        status.max_prompt_bytes
    );
    ensure!(request.max_tokens > 0, "max_tokens must be positive");
    ensure!(
        request.max_tokens <= status.max_new_tokens,
        "max_tokens {} exceeds server limit {}",
        request.max_tokens,
        status.max_new_tokens
    );
    ensure!(
        request.temperature.is_finite() && request.temperature >= 0.0 && request.temperature <= 5.0,
        "temperature must be finite and between 0 and 5"
    );
    ensure!(
        request.repetition_penalty.is_finite()
            && request.repetition_penalty >= 1.0
            && request.repetition_penalty <= 5.0,
        "repetition_penalty must be finite and between 1 and 5"
    );
    ensure!(
        request
            .top_k
            .is_none_or(|top_k| top_k > 0 && top_k <= status.vocab_size),
        "top_k must be between 1 and model vocab_size {}",
        status.vocab_size
    );
    Ok(())
}

async fn get_status(State(state): State<AppState>) -> Json<LabStatus> {
    Json(state.status)
}

async fn post_trace(
    State(state): State<AppState>,
    Json(request): Json<LiveTraceRequest>,
) -> Response {
    if let Err(error) = validate_request(&request, &state.status) {
        return api_error(StatusCode::BAD_REQUEST, error.to_string());
    }
    let (response, receive) = oneshot::channel();
    match state.jobs.try_send(TraceJob { request, response }) {
        Ok(()) => {}
        Err(TrySendError::Full(_)) => {
            return api_error(
                StatusCode::TOO_MANY_REQUESTS,
                "the model is busy; wait for the active trace to finish".to_owned(),
            );
        }
        Err(TrySendError::Disconnected(_)) => {
            return api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "the inference worker is unavailable".to_owned(),
            );
        }
    }
    match receive.await {
        Ok(Ok(bundle)) => Json(bundle).into_response(),
        Ok(Err(TraceFailure::Invalid(error))) => api_error(StatusCode::UNPROCESSABLE_ENTITY, error),
        Ok(Err(TraceFailure::Internal(error))) => {
            api_error(StatusCode::INTERNAL_SERVER_ERROR, error)
        }
        Err(_) => api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "the inference worker stopped before returning a trace".to_owned(),
        ),
    }
}

fn api_error(status: StatusCode, error: String) -> Response {
    (status, Json(ApiError { error })).into_response()
}

async fn shutdown_signal() {
    if let Err(error) = tokio::signal::ctrl_c().await {
        tracing::warn!("failed to install Ctrl-C handler: {error}");
    }
}

/// Serve the live model lab and serialize all GPU work through one worker.
pub async fn serve_lab(
    model: Transformer,
    device: Device,
    tokenizer: Tokenizer,
    mut config: LabServerConfig,
) -> Result<()> {
    ensure!(
        config.bind.ip().is_loopback() || config.allow_remote,
        "refusing non-loopback bind {}; pass --allow-remote to expose prompts and model traces",
        config.bind
    );
    ensure!(config.max_new_tokens > 0, "max_new_tokens must be positive");
    ensure!(
        config.max_prompt_bytes > 0,
        "max_prompt_bytes must be positive"
    );
    ensure!(
        config.max_prompt_bytes <= 1024 * 1024,
        "max_prompt_bytes must not exceed 1048576"
    );
    config.trace_options.validate()?;
    config.web_root = config
        .web_root
        .canonicalize()
        .with_context(|| format!("web root {} does not exist", config.web_root.display()))?;
    ensure!(
        config.web_root.join("model-lab.html").is_file(),
        "web root {} has no model-lab.html",
        config.web_root.display()
    );
    if let Some(metrics) = &config.metrics_path {
        ensure!(
            metrics.is_file(),
            "metrics file {} does not exist",
            metrics.display()
        );
    }

    let status = LabStatus {
        status: "ready",
        model: model.config().name.clone(),
        num_layers: model.config().num_layers,
        hidden_size: model.config().hidden_size,
        vocab_size: model.config().vocab_size,
        max_new_tokens: config.max_new_tokens,
        max_prompt_bytes: config.max_prompt_bytes,
    };
    let worker = InferenceWorker {
        model,
        device,
        tokenizer,
        metrics_path: config.metrics_path,
        trace_options: config.trace_options,
    };
    let (send, receive) = sync_channel::<TraceJob>(1);
    let worker_thread = std::thread::Builder::new()
        .name("hermes-model-lab".to_owned())
        .spawn(move || {
            while let Ok(job) = receive.recv() {
                let _ = job.response.send(worker.trace(job.request));
            }
        })
        .context("failed to start model-lab inference worker")?;

    let state = AppState { jobs: send, status };
    let listener = tokio::net::TcpListener::bind(config.bind)
        .await
        .with_context(|| format!("failed to bind model lab to {}", config.bind))?;
    tracing::info!(
        "Model Lab ready at http://{}/model-lab.html (web root {})",
        config.bind,
        config.web_root.display()
    );
    let web_root = config.web_root;
    let body_limit = config
        .max_prompt_bytes
        .checked_add(4 * 1024)
        .context("model-lab request body limit overflows usize")?;
    let result = {
        let app = Router::new()
            .route(
                "/",
                get(|| async { Redirect::temporary("/model-lab.html") }),
            )
            .route("/api/status", get(get_status))
            .route("/api/trace", post(post_trace))
            .layer(DefaultBodyLimit::max(body_limit))
            .fallback_service(ServeDir::new(web_root))
            .with_state(state);
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
    };
    worker_thread
        .join()
        .map_err(|_| anyhow::anyhow!("model-lab inference worker panicked"))?;
    result.context("model-lab HTTP server failed")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn status() -> LabStatus {
        LabStatus {
            status: "ready",
            model: "test".to_owned(),
            num_layers: 2,
            hidden_size: 16,
            vocab_size: 32,
            max_new_tokens: 64,
            max_prompt_bytes: 128,
        }
    }

    #[test]
    fn live_request_limits_fail_before_inference() {
        let valid = LiveTraceRequest {
            prompt: "Explain attention".to_owned(),
            max_tokens: 8,
            temperature: 0.8,
            top_k: Some(10),
            repetition_penalty: 1.05,
            seed: Some(7),
        };
        validate_request(&valid, &status()).unwrap();

        let mut blank = valid;
        blank.prompt = "   ".to_owned();
        assert!(
            validate_request(&blank, &status())
                .unwrap_err()
                .to_string()
                .contains("blank")
        );
        blank.prompt = "valid".to_owned();
        blank.max_tokens = 65;
        assert!(
            validate_request(&blank, &status())
                .unwrap_err()
                .to_string()
                .contains("server limit")
        );
        blank.max_tokens = 8;
        blank.repetition_penalty = 0.99;
        assert!(
            validate_request(&blank, &status())
                .unwrap_err()
                .to_string()
                .contains("repetition_penalty")
        );
    }

    #[test]
    fn live_request_defaults_enable_moderate_sampling_controls() {
        let request = serde_json::from_str::<LiveTraceRequest>(r#"{"prompt":"hello"}"#).unwrap();
        assert_eq!(request.temperature, DEFAULT_TEMPERATURE);
        assert_eq!(request.repetition_penalty, DEFAULT_REPETITION_PENALTY);
    }

    #[test]
    fn live_request_rejects_unknown_fields() {
        let error = serde_json::from_str::<LiveTraceRequest>(
            r#"{"prompt":"hello","max_tokens":4,"unbounded":true}"#,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("unknown field `unbounded`"), "{error}");
    }
}
