# SCOPE-Router Support

This fork adds proxy-level SCOPE-Router support to cc-switch.

## Runtime Flow

```text
incoming request
-> collect candidate models from provider config
-> POST body + candidates to SCOPE-Router /route
-> if SCOPE returns a valid candidate, rewrite body["model"]
-> continue cc-switch's existing provider mapping and forwarding path
```

The integration lives in:

- `src-tauri/src/proxy/scope_router.rs`
- `src-tauri/src/proxy/forwarder.rs`
- `src-tauri/src/proxy/mod.rs`

SCOPE-Router is enabled by default and runs before static model mapping. If
SCOPE is disabled, times out, or returns a non-candidate model, the request
falls back to the original ACRouter/static mapping path.

## Config

Add `scopeRouter` to the provider `settings_config`:

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

Set `"enabled": false` to disable SCOPE-Router for that provider.

Start the SCOPE-Router service from the repository root:

```bash
python integrations/scope_router_service.py \
  --router /path/to/scope_router.pkl \
  --host 127.0.0.1 \
  --port 8760
```
