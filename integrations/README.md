# Runtime Integrations

These integrations use SCOPE-Router as a model selector inside existing agent
gateways. The agent framework stays fixed; SCOPE-Router only chooses the
backend model from the gateway's configured candidate pool.

This repository also ships two ready-to-adapt gateway forks:

- `../cc-switch/`: cc-switch with a proxy-level `scope_router` hook.
- `../claude-code-router/`: Claude Code Router with a gateway-level
  `resolveScopeRouterRouteDecision` hook.

This matches the deployment pattern used by tools such as cc-switch and Claude
Code Router:

```text
incoming agent request
-> gateway collects configured candidate models
-> POST request summary + candidates to SCOPE-Router
-> SCOPE-Router returns selected model
-> gateway rewrites body.model
-> normal provider/model mapping and upstream request continue
```

## Start The Router Service

```bash
python integrations/scope_router_service.py \
  --router /path/to/scope_router.pkl \
  --host 127.0.0.1 \
  --port 8760 \
  --text-encoder BAAI/bge-m3 \
  --vision-encoder facebook/dinov2-large
```

Health check:

```bash
curl http://127.0.0.1:8760/health
```

Route request contract:

```json
{
  "body": {
    "model": "current-static-model",
    "messages": [
      {"role": "user", "content": "Fix the failing parser test."}
    ],
    "tools": []
  },
  "candidates": [
    {"model": "cheap-model"},
    {"model": "strong-model"}
  ],
  "fallback_model": "cheap-model"
}
```

Response:

```json
{
  "model": "strong-model",
  "selected_model": "strong-model",
  "selector": "strong-model",
  "score": 0.82,
  "routed": true,
  "reason": "scope-router"
}
```

For Claude Code Router style candidates, pass both `selector` and `model`:

```json
{"selector": "OpenRouter/openai/gpt-5.4", "provider": "OpenRouter", "model": "openai/gpt-5.4"}
```

The service matches SCOPE checkpoint `model_names` against candidate `model`,
`selector`, `name`, `id`, `provider/model`, or `aliases`. It returns `model` as
the value the gateway should write into `body.model`.

## cc-switch

The included `../cc-switch/` fork calls SCOPE-Router in the same place its
ACRouter hook runs: after candidate models are collected and before static
model mapping. The hook:

1. Build a request with the current JSON body and provider candidate models.
2. POST it to `http://127.0.0.1:8760/route`.
3. If the response contains a valid candidate, set `body["model"]` to
   `response["model"]`.
4. If the service errors or returns a non-candidate, keep the original static
   mapping path.

A provider-level config can mirror the existing ACRouter config:

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

Conceptually, the cc-switch forwarding path becomes:

```text
raw request body
-> SCOPE-Router model rewrite, if enabled
-> existing static model mapping, if SCOPE did not route
-> existing adapter/transform/provider forwarding
```

This keeps cc-switch's normal provider support, auth handling, streaming,
fallback, and model mapping behavior intact.

In the included cc-switch fork, SCOPE-Router is enabled by default. Set
`"enabled": false` under `scopeRouter` to disable it for a provider.

## Claude Code Router

The included `../claude-code-router/` fork calls the same `/route` service from
its gateway router path, where ACRouter currently resolves a model from
configured `provider/model` candidates.

The candidate shape should include selectors:

```json
[
  {
    "selector": "OpenRouter/openai/gpt-5.4",
    "provider": "OpenRouter",
    "model": "openai/gpt-5.4",
    "estimated_cost_usd": 0.01
  }
]
```

If SCOPE-Router returns `{"model": "OpenRouter/openai/gpt-5.4"}`, CCR can set
`body.model` to that selector and continue through its existing provider
resolution path.

## Training Requirement

For runtime routing to be meaningful, train the SCOPE checkpoint with the same
candidate model ids used by the gateway. For cc-switch this is usually upstream
model names. For Claude Code Router this may be provider/model selectors.
