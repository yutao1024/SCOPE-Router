# cc-switch Integration

SCOPE-Router should be wired into cc-switch as a proxy-level route-decision
hook, not as an external agent wrapper.

The insertion point is the same place used by ACRouter in cc-switch:

```text
forwarder receives request body
-> collect provider candidate models
-> POST body + candidates to SCOPE-Router /route
-> if routed, set body["model"] = response["model"]
-> continue existing provider mapping / adapter / streaming path
```

Minimal provider config:

```json
{
  "settings_config": {
    "scopeRouter": {
      "enabled": true,
      "endpoint": "http://127.0.0.1:8760/route",
      "timeoutMs": 2000
    }
  }
}
```

The hook request should include the unmodified request body and all candidate
models visible to the provider:

```json
{
  "body": {
    "model": "claude-sonnet-4-5",
    "messages": [
      {"role": "user", "content": "Fix the failing parser test."}
    ]
  },
  "candidates": [
    {"model": "claude-haiku-4-5"},
    {"model": "claude-sonnet-4-5"},
    {"model": "gpt-5.1-codex"}
  ],
  "fallback_model": "claude-sonnet-4-5"
}
```

On success, cc-switch should only rewrite the model:

```rust
if let Some(selected) = scope_router_response.get("model").and_then(Value::as_str) {
    body["model"] = Value::String(selected.to_string());
}
```

If the service errors, times out, or returns a model that is not in the
candidate list, keep cc-switch's original static mapping path.
