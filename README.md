# SCOPE-Router

SCOPE-Router is a cost-aware, open-set router for vision-language models. It
routes each query to a candidate VLM by matching the query embedding against
reusable model profiles built from a small calibration set, allowing new
candidate models to be added by profiling them without retraining the router.

The repository contains the SCOPE-Router implementation, feature extraction and
calibration utilities, evaluation scripts, and the VLM-ExecRouterBench data
generation/evaluation pipeline.

## Citation

```bibtex
@article{scope-router,
  title  = {SCOPE-Router: Cost-Aware Open-Set Routing for Execution-Oriented Vision-Language Models},
  author = {Anonymous Authors},
  journal = {arXiv preprint},
  year   = {2026}
}
```

## Method

SCOPE-Router has three main stages:

1. Select a compact calibration set from the training split.
2. Build one reusable profile vector per candidate model from calibration
   correctness, cost, and optional query-aware statistics.
3. Train a profile-aware router that scores query-model compatibility.

The main training objective uses:

- Cost-aware Relevance Matching (CRM): dense query-model relevance targets that
  reward correct low-cost models.
- Routing-Consistency Contrastive Regularization (RCCR): a query-space
  regularizer that keeps queries with similar routing targets close.

## Repository Layout

| Path | Purpose |
|---|---|
| `routers/scope_router/` | Main frozen-feature SCOPE-Router implementation. |
| `routers/scope_router_online/` | Optional online variant with trainable query encoders. |
| `routers/features/` | Text and vision embedding extraction utilities. |
| `routers/utils/` | Shared evaluation, ranking, target-construction, and training helpers. |
| `tools/select_calibration_set.py` | Random, diagnostic, diversity, and hybrid calibration selection. |
| `tools/build_calibration_profile.py` | Calibration profile construction. |
| `scripts/train_scope_router.sh` | End-to-end SCOPE-Router training script. |
| `integrations/scope_router_service.py` | Runtime HTTP routing service for agent gateways such as cc-switch and Claude Code Router. |
| `cc-switch/` | Ready-to-adapt cc-switch fork with proxy-level SCOPE-Router support. |
| `claude-code-router/` | Ready-to-adapt Claude Code Router fork with gateway-level SCOPE-Router support. |
| `VLM-ExecRouterBench/` | Execution-oriented benchmark construction and SFT/evaluation utilities. |

## Installation

```bash
git clone https://github.com/yutao1024/SCOPE-Router.git
cd SCOPE-Router

conda create -n scope-router python=3.10 -y
conda activate scope-router

pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset Format

The router-side pipeline expects a prepared dataset directory with files such as:

```text
BENCHMARKS/*.tsv
SPLITS/*.jsonl
data/matrices/Y.npz
data/matrices/C.npy
data/matrices/sample_ids.pkl
data/matrices/model_names.pkl
data/registry/meta.parquet
config/models.yaml
config/pricing.yaml
```

Precomputed text and vision embeddings are stored under:

```text
EMBEDDINGS/text/<encoder>.parquet
EMBEDDINGS/vision/<encoder>.parquet
```

## Feature Extraction

The default paper setting uses BAAI/bge-m3 for text and facebook/dinov2-large
for images:

```bash
python routers/features/extract_cli.py \
  --dataset_dir /path/to/dataset \
  --text_encoder BAAI/bge-m3 \
  --vision_encoder facebook/dinov2-large \
  --batch_size 64
```

## Train SCOPE-Router

The convenience script selects a 1024-sample hybrid calibration set, builds a
query-aware profile, and trains SCOPE-Router with CRM + RCCR:

```bash
bash scripts/train_scope_router.sh /path/to/dataset
```

Equivalent core training flags:

```bash
python routers/scope_router/train_and_eval.py \
  --dataset_dir /path/to/dataset \
  --profile_path /path/to/profile.npz \
  --output_dir outputs/scope_router \
  --text_encoder BAAI/bge-m3 \
  --vision_encoder facebook/dinov2-large \
  --fusion_method normalize_concat \
  --embedding_dim 64 \
  --query_hidden_dim 128 \
  --profile_hidden_dim 128 \
  --loss_type crm \
  --crm_target relevance \
  --crm_bias none \
  --rccr_weight 1.0 \
  --rccr_temperature 0.1 \
  --learn_rccr_temperature \
  --train_lambda 10 \
  --cost_scale 100 \
  --monitor_metric rank_score
```

## Runtime Gateway Integration

SCOPE-Router can be used as a live model selector inside an existing agent
gateway. The agent framework stays fixed; SCOPE-Router only chooses the backend
model from the gateway's configured candidate pool:

```text
Codex / Claude Code / OpenClaw request
-> gateway collects candidate models
-> SCOPE-Router selects one candidate
-> gateway rewrites body.model
-> normal provider mapping and upstream forwarding continue
```

Start the service:

```bash
python integrations/scope_router_service.py \
  --router /path/to/scope_router.pkl \
  --host 127.0.0.1 \
  --port 8760 \
  --text-encoder BAAI/bge-m3 \
  --vision-encoder facebook/dinov2-large
```

A gateway sends the original request body and candidate models:

```bash
curl -s http://127.0.0.1:8760/route \
  -H 'Content-Type: application/json' \
  -d '{
    "body": {
      "model": "current-static-model",
      "messages": [{"role": "user", "content": "Fix the failing parser test."}]
    },
    "candidates": [
      {"model": "cheap-model"},
      {"model": "strong-model"}
    ],
    "fallback_model": "cheap-model"
  }'
```

The response contains the selector that should be written into `body.model`:

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

For cc-switch, this repository includes a ready-to-adapt fork under
`cc-switch/`. It adds a proxy-level `scope_router` hook before static model
mapping. For Claude Code Router, `claude-code-router/` adds a gateway-level
`resolveScopeRouterRouteDecision` hook before provider resolution. See
`integrations/README.md` for the request contract and config snippets.

## VLM-ExecRouterBench

`VLM-ExecRouterBench/` contains the execution-oriented benchmark generation and
evaluation utilities used by the project. See
`VLM-ExecRouterBench/README.md` and the documents under
`VLM-ExecRouterBench/docs/` for the benchmark-side workflow.
