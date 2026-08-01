# Claude Code Router Integration

SCOPE-Router should be called from Claude Code Router's gateway model-selection
path. It replaces the router-model call used by ACRouter, but keeps the rest of
the gateway behavior unchanged.

Expected flow:

```text
gateway receives Claude/Codex-compatible request
-> collect configured provider/model candidates
-> POST body + candidates to SCOPE-Router /route
-> set body.model to response.model
-> continue normal provider resolution and forwarding
```

Candidate entries should include both the provider selector and the upstream
model name:

```json
[
  {
    "selector": "OpenRouter/openai/gpt-5.1",
    "provider": "OpenRouter",
    "model": "openai/gpt-5.1",
    "estimated_cost_usd": 0.01
  },
  {
    "selector": "Anthropic/claude-sonnet-4-5",
    "provider": "Anthropic",
    "model": "claude-sonnet-4-5",
    "estimated_cost_usd": 0.02
  }
]
```

The SCOPE service matches checkpoint `model_names` against `model`,
`selector`, `provider/model`, and optional `aliases`. The returned `model` field
is the selector that Claude Code Router should write into `body.model`.

Fallback rule: if SCOPE-Router cannot route, keep the existing route decision
or the request's original `body.model`.

