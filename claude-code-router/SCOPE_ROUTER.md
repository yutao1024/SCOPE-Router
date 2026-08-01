# SCOPE-Router Support

This fork adds gateway-level SCOPE-Router support to Claude Code Router.

## Runtime Flow

```text
incoming gateway request
-> collect configured provider/model candidates
-> POST body + candidates to SCOPE-Router /route
-> if SCOPE returns a valid provider/model selector, set body.model
-> continue Claude Code Router's normal provider resolution and forwarding path
```

The integration lives in:

- `src/server/gateway/scope-router.ts`
- `src/server/gateway/claude-code-router-plugin.ts`
- `src/shared/app.ts`
- `src/main/config.ts`

SCOPE-Router is enabled by default and tried before ACRouter. If SCOPE is
disabled, times out, or returns a non-candidate selector, the original
ACRouter/static routing path is kept.

## Config

Add `ScopeRouter` to the app config:

```json
{
  "ScopeRouter": {
    "enabled": true,
    "endpoint": "http://127.0.0.1:8760/route",
    "timeoutMs": 2000
  }
}
```

Set `"enabled": false` to disable SCOPE-Router.

Start the SCOPE-Router service from the repository root:

```bash
python integrations/scope_router_service.py \
  --router /path/to/scope_router.pkl \
  --host 127.0.0.1 \
  --port 8760
```
