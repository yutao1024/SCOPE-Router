//! SCOPE-Router integration for live proxy requests.
//!
//! This module mirrors the proxy-level ACRouter hook, but delegates the route
//! decision to a local SCOPE-Router HTTP service. It only rewrites `body.model`
//! when the service returns one of the provider's configured candidate models.

use super::provider_router::ProviderRouter;
use crate::{app_config::AppType, provider::Provider};
use serde::Serialize;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Clone, Serialize)]
struct Candidate {
    model: String,
    selector: String,
    estimated_cost_usd: Option<f64>,
}

#[derive(Debug, Clone)]
struct ScopeRouterConfig {
    enabled: bool,
    endpoint: String,
    timeout_ms: u64,
}

#[derive(Debug, Clone)]
pub struct ScopeRouterResult {
    pub body: Value,
    pub routed: bool,
}

const DEFAULT_ENDPOINT: &str = "http://127.0.0.1:8760/route";
const DEFAULT_TIMEOUT_MS: u64 = 2000;

pub async fn apply_scope_router(
    router: &Arc<ProviderRouter>,
    app_type: &AppType,
    provider: &Provider,
    body: Value,
) -> ScopeRouterResult {
    let config = scope_router_config(provider);
    if !config.enabled || matches!(app_type, AppType::ClaudeDesktop | AppType::Gemini) {
        return ScopeRouterResult {
            body,
            routed: false,
        };
    }

    let candidates = collect_candidates(router, provider, &body);
    if candidates.len() <= 1 {
        return ScopeRouterResult {
            body,
            routed: false,
        };
    }

    let original_body = body.clone();
    let fallback_model = body
        .get("model")
        .and_then(Value::as_str)
        .map(str::to_string);
    let request_body = json!({
        "body": body,
        "candidates": candidates,
        "fallback_model": fallback_model,
        "source": "cc-switch"
    });

    let timeout = Duration::from_millis(config.timeout_ms.max(1000));
    match call_scope_router(&config.endpoint, request_body, timeout).await {
        Ok(response) => {
            if response.get("routed").and_then(Value::as_bool) != Some(true) {
                return ScopeRouterResult {
                    body: original_body,
                    routed: false,
                };
            }
            if let Some(model) = parse_scope_model(&response, &candidates) {
                let mut routed = original_body;
                routed["model"] = Value::String(model.clone());
                log::info!(
                    "[SCOPE-Router] provider={} endpoint={} selected_model={}",
                    provider.id,
                    config.endpoint,
                    model
                );
                return ScopeRouterResult {
                    body: routed,
                    routed: true,
                };
            }
            log::warn!(
                "[SCOPE-Router] Ignoring response without a valid candidate model: {}",
                truncate(&compact_json(Some(&response)), 300)
            );
            ScopeRouterResult {
                body: original_body,
                routed: false,
            }
        }
        Err(error) => {
            log::warn!(
                "[SCOPE-Router] Route request failed for provider={}: {error}",
                provider.id
            );
            ScopeRouterResult {
                body: original_body,
                routed: false,
            }
        }
    }
}

fn scope_router_config(provider: &Provider) -> ScopeRouterConfig {
    let raw = provider
        .settings_config
        .get("scopeRouter")
        .or_else(|| provider.settings_config.get("scope_router"))
        .or_else(|| provider.settings_config.get("SCOPERouter"));
    let enabled = raw
        .and_then(|value| value.get("enabled"))
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let endpoint = raw
        .and_then(|value| value.get("endpoint").or_else(|| value.get("url")))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DEFAULT_ENDPOINT)
        .to_string();
    let timeout_ms = raw
        .and_then(|value| {
            value
                .get("timeoutMs")
                .or_else(|| value.get("timeout_ms"))
                .or_else(|| value.get("requestTimeoutMs"))
        })
        .and_then(Value::as_u64)
        .unwrap_or(DEFAULT_TIMEOUT_MS)
        .clamp(1000, 60_000);
    ScopeRouterConfig {
        enabled,
        endpoint,
        timeout_ms,
    }
}

fn collect_candidates(
    router: &Arc<ProviderRouter>,
    provider: &Provider,
    body: &Value,
) -> Vec<Candidate> {
    let mut models = Vec::new();
    push_model(
        &mut models,
        body.get("model")
            .and_then(Value::as_str)
            .unwrap_or_default(),
    );

    if let Some(env) = provider.settings_config.get("env") {
        for key in [
            "ANTHROPIC_MODEL",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL",
            "ANTHROPIC_DEFAULT_SONNET_MODEL",
            "ANTHROPIC_DEFAULT_OPUS_MODEL",
            "ANTHROPIC_DEFAULT_FABLE_MODEL",
        ] {
            push_model(
                &mut models,
                env.get(key).and_then(Value::as_str).unwrap_or_default(),
            );
        }
    }

    collect_models_from_value(&provider.settings_config, &mut models);

    models
        .into_iter()
        .map(|model| Candidate {
            estimated_cost_usd: router.estimate_model_cost_1k_in_1k_out(&model),
            selector: model.clone(),
            model,
        })
        .collect()
}

fn collect_models_from_value(value: &Value, output: &mut Vec<String>) {
    match value {
        Value::Array(items) => {
            for item in items {
                collect_models_from_value(item, output);
            }
        }
        Value::Object(map) => {
            for (key, value) in map {
                let normalized = key.to_ascii_lowercase();
                if matches!(normalized.as_str(), "acrouter" | "agentrouter" | "scoperouter" | "scope_router") {
                    continue;
                }
                if matches!(
                    normalized.as_str(),
                    "apikey"
                        | "api_key"
                        | "anthropic_api_key"
                        | "anthropic_auth_token"
                        | "openai_api_key"
                        | "authorization"
                        | "auth"
                        | "routermodel"
                        | "router_model"
                        | "endpoint"
                ) {
                    continue;
                }
                if normalized == "model"
                    || normalized.ends_with("model")
                    || normalized == "models"
                    || normalized == "modelid"
                    || normalized == "model_id"
                {
                    push_model_values(output, value);
                }
                collect_models_from_value(value, output);
            }
        }
        Value::String(text) => {
            for model in extract_toml_model_values(text) {
                push_model(output, &model);
            }
        }
        _ => {}
    }
}

fn push_model_values(output: &mut Vec<String>, value: &Value) {
    match value {
        Value::String(text) => push_model(output, text),
        Value::Array(items) => {
            for item in items {
                if let Some(text) = item.as_str() {
                    push_model(output, text);
                }
            }
        }
        _ => {}
    }
}

fn extract_toml_model_values(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if !(trimmed.starts_with("model ") || trimmed.starts_with("model=")) {
                return None;
            }
            let (_, value) = trimmed.split_once('=')?;
            let value = value.trim().trim_matches('"').trim_matches('\'').trim();
            (!value.is_empty()).then(|| value.to_string())
        })
        .collect()
}

fn push_model(output: &mut Vec<String>, value: &str) {
    let model = value.trim();
    if model.is_empty() || model == "unknown" || looks_secret_like(model) {
        return;
    }
    if !output.iter().any(|item| item.eq_ignore_ascii_case(model)) {
        output.push(model.to_string());
    }
}

fn looks_secret_like(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    lower.starts_with("sk-")
        || lower.starts_with("bearer ")
        || lower.contains("api_key")
        || lower.contains("auth_token")
}

async fn call_scope_router(endpoint: &str, body: Value, timeout: Duration) -> Result<Value, String> {
    let request = super::http_client::get()
        .post(endpoint)
        .timeout(timeout)
        .json(&body);
    let response = request.send().await.map_err(|error| error.to_string())?;
    let status = response.status();
    let text = response.text().await.map_err(|error| error.to_string())?;
    let payload = serde_json::from_str::<Value>(&text).unwrap_or(Value::String(text));
    if !status.is_success() {
        return Err(format!(
            "HTTP {status}: {}",
            truncate(&compact_json(Some(&payload)), 500)
        ));
    }
    Ok(payload)
}

fn parse_scope_model(response: &Value, candidates: &[Candidate]) -> Option<String> {
    let model = response
        .get("model")
        .or_else(|| response.get("selector"))
        .or_else(|| response.get("selected_model"))
        .and_then(Value::as_str)?
        .trim();
    candidates
        .iter()
        .find(|candidate| {
            candidate.model.eq_ignore_ascii_case(model)
                || candidate.selector.eq_ignore_ascii_case(model)
        })
        .map(|candidate| candidate.selector.clone())
}

fn compact_json(value: Option<&Value>) -> String {
    serde_json::to_string(value.unwrap_or(&Value::Null)).unwrap_or_else(|_| "null".to_string())
}

fn truncate(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let mut output = text.chars().take(limit).collect::<String>();
    output.push_str("...");
    output
}
