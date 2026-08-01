//! ACRouter integration for live proxy requests.
//!
//! This module is intentionally side-effect light: it only rewrites
//! `body.model` when a router call returns one of the provider's configured
//! candidate models. Any router error falls back to the original request path.

use super::{
    provider_router::ProviderRouter,
    providers::{
        codex_provider_uses_chat_completions, get_claude_api_format, AuthStrategy, ProviderAdapter,
    },
};
use crate::{app_config::AppType, provider::Provider};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Clone)]
struct Candidate {
    model: String,
    estimated_cost_usd: Option<f64>,
}

#[derive(Debug, Clone)]
struct RouterConfig {
    enabled: bool,
    router_model: Option<String>,
    timeout_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RouterWireApi {
    AnthropicMessages,
    OpenAIChat,
    OpenAIResponses,
}

#[derive(Debug, Clone)]
pub struct ACRouterResult {
    pub body: Value,
    pub routed: bool,
}

const DEFAULT_TIMEOUT_MS: u64 = 8000;
const PROMPT_PREVIEW_LIMIT: usize = 12_000;

pub async fn apply_acrouter(
    router: &Arc<ProviderRouter>,
    app_type: &AppType,
    provider: &Provider,
    body: Value,
    adapter: &dyn ProviderAdapter,
) -> ACRouterResult {
    let config = router_config(provider);
    if !config.enabled || matches!(app_type, AppType::ClaudeDesktop | AppType::Gemini) {
        return ACRouterResult {
            body,
            routed: false,
        };
    }

    let candidates = collect_candidates(router, provider, &body);
    if candidates.len() <= 1 {
        return ACRouterResult {
            body,
            routed: false,
        };
    }

    let Some(router_model) = select_router_model(&candidates, config.router_model.as_deref())
    else {
        return ACRouterResult {
            body,
            routed: false,
        };
    };

    let prompt = build_prompt(&body, &candidates);
    let timeout = Duration::from_millis(config.timeout_ms.max(1000));
    match call_router_model(app_type, provider, adapter, &router_model, &prompt, timeout).await {
        Ok(text) => {
            if let Some(model) = parse_router_model(&text, &candidates) {
                if body.get("model").and_then(Value::as_str) == Some(model.as_str()) {
                    return ACRouterResult { body, routed: true };
                }
                let mut routed = body;
                routed["model"] = Value::String(model.clone());
                log::info!(
                    "[ACRouter] provider={} router_model={} selected_model={}",
                    provider.id,
                    router_model,
                    model
                );
                ACRouterResult {
                    body: routed,
                    routed: true,
                }
            } else {
                log::warn!(
                    "[ACRouter] Ignoring router response without a valid model: {}",
                    truncate(&text, 300)
                );
                ACRouterResult {
                    body,
                    routed: false,
                }
            }
        }
        Err(error) => {
            log::warn!(
                "[ACRouter] Router call failed for provider={}: {error}",
                provider.id
            );
            ACRouterResult {
                body,
                routed: false,
            }
        }
    }
}

fn router_config(provider: &Provider) -> RouterConfig {
    let raw = provider
        .settings_config
        .get("acrouter")
        .or_else(|| provider.settings_config.get("ACRouter"));
    let enabled = raw
        .and_then(|value| value.get("enabled"))
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let router_model = raw
        .and_then(|value| {
            value
                .get("routerModel")
                .or_else(|| value.get("router_model"))
                .or_else(|| value.get("model"))
        })
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
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
    RouterConfig {
        enabled,
        router_model,
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
                if matches!(normalized.as_str(), "acrouter" | "agentrouter") {
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

fn select_router_model(candidates: &[Candidate], preferred: Option<&str>) -> Option<String> {
    if let Some(preferred) = preferred {
        if candidates
            .iter()
            .any(|candidate| candidate.model.eq_ignore_ascii_case(preferred))
        {
            return Some(preferred.to_string());
        }
    }

    candidates
        .iter()
        .min_by(|left, right| {
            let left_cost = left.estimated_cost_usd.unwrap_or(f64::INFINITY);
            let right_cost = right.estimated_cost_usd.unwrap_or(f64::INFINITY);
            left_cost
                .partial_cmp(&right_cost)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.model.cmp(&right.model))
        })
        .map(|candidate| candidate.model.clone())
}

fn build_prompt(body: &Value, candidates: &[Candidate]) -> String {
    let candidate_lines = candidates
        .iter()
        .map(|candidate| {
            let price = candidate
                .estimated_cost_usd
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "unknown".to_string());
            format!(
                "- {} | estimated_cost_1k_in_1k_out_usd={price}",
                candidate.model
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "You are ACRouter, a model router for coding-agent requests.\n\
Choose exactly one candidate model. Prefer the cheapest model likely to solve the request; use stronger models for complex code edits, debugging, architecture changes, long context, or tool-heavy requests.\n\
Return only compact JSON: {{\"model\":\"model-id\",\"reason\":\"short reason\"}}.\n\n\
Candidates:\n{candidate_lines}\n\n\
Request summary:\n{}",
        summarize_body(body)
    )
}

fn summarize_body(body: &Value) -> String {
    let summary = json!({
        "model": body.get("model"),
        "max_tokens": body.get("max_tokens").or_else(|| body.get("max_output_tokens")),
        "system": truncate(&compact_json(body.get("system")), 1600),
        "messages": truncate(&compact_json(body.get("messages")), PROMPT_PREVIEW_LIMIT),
        "thinking": body.get("thinking"),
        "tool_names": tool_names(body.get("tools")),
    });
    compact_json(Some(&summary))
}

async fn call_router_model(
    app_type: &AppType,
    provider: &Provider,
    adapter: &dyn ProviderAdapter,
    router_model: &str,
    prompt: &str,
    timeout: Duration,
) -> Result<String, String> {
    let base_url = adapter
        .extract_base_url(provider)
        .map_err(|error| error.to_string())?;
    let auth = adapter
        .extract_auth(provider)
        .ok_or_else(|| "provider has no direct auth for ACRouter".to_string())?;
    let client = super::http_client::get();
    let wire_api = router_wire_api(app_type, provider)?;

    if wire_api == RouterWireApi::AnthropicMessages {
        let mut request = client
            .post(build_url(&base_url, "/v1/messages"))
            .timeout(timeout)
            .json(&json!({
                "model": router_model,
                "max_tokens": 160,
                "temperature": 0,
                "system": "Route coding-agent tasks. Return only JSON.",
                "messages": [{"role": "user", "content": prompt}]
            }));
        request = match auth.strategy {
            AuthStrategy::Anthropic => request
                .header("x-api-key", &auth.api_key)
                .header("anthropic-version", "2023-06-01"),
            AuthStrategy::ClaudeAuth | AuthStrategy::Bearer => request.bearer_auth(&auth.api_key),
            _ => return Err("unsupported auth strategy for Claude ACRouter".to_string()),
        };
        let payload = send_json(request).await?;
        return extract_anthropic_text(&payload)
            .ok_or_else(|| "router response did not contain text".to_string());
    }

    if !matches!(
        auth.strategy,
        AuthStrategy::Bearer | AuthStrategy::ClaudeAuth
    ) {
        return Err("unsupported bearer auth strategy for ACRouter".to_string());
    }

    if wire_api == RouterWireApi::OpenAIResponses {
        let request = client
            .post(build_url(&base_url, "/v1/responses"))
            .timeout(timeout)
            .bearer_auth(&auth.api_key)
            .json(&json!({
                "model": router_model,
                "max_output_tokens": 160,
                "temperature": 0,
                "input": format!("Route coding-agent tasks. Return only JSON.\n\n{prompt}")
            }));
        let payload = send_json(request).await?;
        return extract_openai_responses_text(&payload)
            .ok_or_else(|| "router response did not contain text".to_string());
    }

    let request = client
        .post(build_url(&base_url, "/v1/chat/completions"))
        .timeout(timeout)
        .bearer_auth(&auth.api_key)
        .json(&json!({
            "model": router_model,
            "max_tokens": 160,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "Route coding-agent tasks. Return only JSON."},
                {"role": "user", "content": prompt}
            ]
        }));
    let payload = send_json(request).await?;
    payload
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| "router response did not contain text".to_string())
}

fn router_wire_api(app_type: &AppType, provider: &Provider) -> Result<RouterWireApi, String> {
    match app_type {
        AppType::Claude => match get_claude_api_format(provider) {
            "anthropic" => Ok(RouterWireApi::AnthropicMessages),
            "openai_chat" => Ok(RouterWireApi::OpenAIChat),
            "openai_responses" => Ok(RouterWireApi::OpenAIResponses),
            "gemini_native" => {
                Err("Gemini-native Claude providers are unsupported by ACRouter".to_string())
            }
            other => Err(format!(
                "unsupported Claude API format for ACRouter: {other}"
            )),
        },
        AppType::Codex => {
            if codex_provider_uses_chat_completions(provider) {
                Ok(RouterWireApi::OpenAIChat)
            } else {
                Ok(RouterWireApi::OpenAIResponses)
            }
        }
        AppType::OpenCode | AppType::OpenClaw | AppType::Hermes => Ok(RouterWireApi::OpenAIChat),
        AppType::ClaudeDesktop | AppType::Gemini => {
            Err("app type is unsupported by ACRouter".to_string())
        }
    }
}

async fn send_json(request: reqwest::RequestBuilder) -> Result<Value, String> {
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

fn parse_router_model(text: &str, candidates: &[Candidate]) -> Option<String> {
    let parsed = serde_json::from_str::<Value>(text)
        .ok()
        .or_else(|| extract_json_object(text).and_then(|json| serde_json::from_str(&json).ok()));
    if let Some(model) = parsed
        .as_ref()
        .and_then(|value| value.get("model"))
        .and_then(Value::as_str)
    {
        if let Some(candidate) = candidates
            .iter()
            .find(|candidate| candidate.model.eq_ignore_ascii_case(model.trim()))
        {
            return Some(candidate.model.clone());
        }
    }

    let lower = text.to_ascii_lowercase();
    candidates
        .iter()
        .find(|candidate| lower.contains(&candidate.model.to_ascii_lowercase()))
        .map(|candidate| candidate.model.clone())
}

fn extract_anthropic_text(payload: &Value) -> Option<String> {
    let parts = payload
        .get("content")?
        .as_array()?
        .iter()
        .filter_map(|item| {
            (item.get("type").and_then(Value::as_str) == Some("text"))
                .then(|| item.get("text").and_then(Value::as_str))
                .flatten()
        })
        .collect::<Vec<_>>();
    (!parts.is_empty()).then(|| parts.join("\n"))
}

fn extract_openai_responses_text(payload: &Value) -> Option<String> {
    if let Some(text) = payload.get("output_text").and_then(Value::as_str) {
        return Some(text.to_string());
    }
    let mut parts = Vec::new();
    if let Some(output) = payload.get("output").and_then(Value::as_array) {
        for item in output {
            if let Some(content) = item.get("content").and_then(Value::as_array) {
                for part in content {
                    if let Some(text) = part.get("text").and_then(Value::as_str) {
                        parts.push(text);
                    }
                }
            }
        }
    }
    (!parts.is_empty()).then(|| parts.join("\n"))
}

fn extract_json_object(text: &str) -> Option<String> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    (end > start).then(|| text[start..=end].to_string())
}

fn tool_names(tools: Option<&Value>) -> Vec<String> {
    tools
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.get("name").and_then(Value::as_str))
                .take(50)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn build_url(base_url: &str, path: &str) -> String {
    let base = base_url.trim_end_matches('/');
    if base.ends_with(path) {
        return base.to_string();
    }
    if base.ends_with("/v1") && path.starts_with("/v1/") {
        return format!("{base}{}", &path[3..]);
    }
    format!("{base}{path}")
}

fn compact_json(value: Option<&Value>) -> String {
    value
        .map(|value| serde_json::to_string(value).unwrap_or_else(|_| value.to_string()))
        .unwrap_or_else(|| "null".to_string())
}

fn truncate(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut output = value.chars().take(max_chars).collect::<String>();
    output.push_str("...");
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_json_router_response() {
        let candidates = vec![
            Candidate {
                model: "cheap-model".to_string(),
                estimated_cost_usd: Some(0.001),
            },
            Candidate {
                model: "strong-model".to_string(),
                estimated_cost_usd: Some(0.02),
            },
        ];
        assert_eq!(
            parse_router_model(r#"{"model":"strong-model","reason":"hard"}"#, &candidates),
            Some("strong-model".to_string())
        );
    }

    #[test]
    fn cheapest_router_model_wins_by_default() {
        let candidates = vec![
            Candidate {
                model: "strong-model".to_string(),
                estimated_cost_usd: Some(0.02),
            },
            Candidate {
                model: "cheap-model".to_string(),
                estimated_cost_usd: Some(0.001),
            },
        ];
        assert_eq!(
            select_router_model(&candidates, None),
            Some("cheap-model".to_string())
        );
    }

    #[test]
    fn collect_models_supports_model_arrays() {
        let mut models = Vec::new();
        collect_models_from_value(
            &json!({
                "models": ["model-a", "model-b"],
                "env": {
                    "ANTHROPIC_MODEL": "model-c"
                }
            }),
            &mut models,
        );
        assert_eq!(models.len(), 3);
        assert!(models.contains(&"model-a".to_string()));
        assert!(models.contains(&"model-b".to_string()));
        assert!(models.contains(&"model-c".to_string()));
    }

    #[test]
    fn collect_models_skips_acrouter_config() {
        let mut models = Vec::new();
        collect_models_from_value(
            &json!({
                "model": "target-model",
                "acrouter": {
                    "routerModel": "router-only-model"
                }
            }),
            &mut models,
        );
        assert_eq!(models, vec!["target-model"]);
    }
}
