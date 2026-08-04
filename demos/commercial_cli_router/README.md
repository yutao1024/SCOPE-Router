# Commercial CLI Router MVP

This MVP routes a user prompt into one of three local coding-agent CLIs:
Codex, Claude Code, or opencode. It is a lightweight wrapper around installed
command-line tools, so users keep their normal product login, billing,
workspace trust, and permission settings.

Dry-run the routing decision:

```bash
python demos/commercial_cli_router/router_mvp.py \
  --prompt "Patch this repository so pytest passes" \
  --dry-run
```

Force a backend:

```bash
python demos/commercial_cli_router/router_mvp.py \
  --tool codex \
  --workdir /path/to/project \
  --prompt "Run the tests and fix the failing parser case"
```

The default templates are in `tools.example.json`:

- `codex`: `codex exec --sandbox workspace-write --cd {workdir} <prompt>`
- `claude-code`: `claude --print --permission-mode acceptEdits <prompt>`
- `opencode`: `opencode run <prompt>`

Edit those templates if your installed CLI uses a different non-interactive
form. `router_mvp.py --dry-run` writes the exact rendered command to
`demos/commercial_cli_router/runs/`.

## cc-switch Or Other Wrappers

Each tool supports an optional `command_prefix` list and a `command_prefix_env`.
This lets you wrap the selected product CLI without changing router code:

```bash
SCOPE_ROUTER_CODEX_PREFIX="ccswitch" \
python demos/commercial_cli_router/router_mvp.py \
  --tool codex \
  --prompt "Patch this repository so pytest passes" \
  --dry-run
```

The rendered command becomes `ccswitch codex exec ... <prompt>`. If your local
wrapper uses a different syntax, either set a longer prefix such as
`SCOPE_ROUTER_CODEX_PREFIX="ccswitch run --"` or copy
`tools.ccswitch.example.json` and edit `command_prefix` directly.

For compatibility with agent-as-a-router examples, the script also accepts
`ACROUTER_CODEX_PREFIX`, `ACROUTER_CLAUDE_PREFIX`, and
`ACROUTER_OPENCODE_PREFIX` when the corresponding `SCOPE_ROUTER_*` variable is
not set.

## Local npm Wrapper

The `npm/` folder is a ready-to-link local package. It is private by default so
that publishing requires an intentional maintainer step.

```bash
cd demos/commercial_cli_router/npm
npm link

scope-cli-router --prompt "Review this repo for failing tests" --dry-run
```
