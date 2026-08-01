# Local VLM Serving

Scripts for downloading and launching the local OOD candidate models on a 4xA100
machine. vLLM endpoints are OpenAI-compatible and can be called at
`http://HOST:PORT/v1/chat/completions`.

## Install

```bash
cd /path/to/VL-RouterBench
bash scripts/local_vlm_serving/install_deps.sh
```

If Kimi or other long-context models hit attention-related OOMs, install
FlashAttention:

```bash
INSTALL_FLASH_ATTN=1 bash scripts/local_vlm_serving/install_deps.sh
```

## Download

Set a large model directory. On the server, prefer a data disk rather than the
repo directory.

```bash
export MODEL_ROOT=/data/yutao/qyf/VLM-Router/local_models/vlm
export HF_HOME=/data/yutao/qyf/VLM-Router/hf_cache

# Download all configured local VLMs.
bash scripts/local_vlm_serving/download_models.sh

# Or download selected models.
bash scripts/local_vlm_serving/download_models.sh Qwen2.5-VL-32B-Instruct Pixtral-12B
```

If a model is gated, login first:

```bash
huggingface-cli login
```

## Serve One Model

```bash
# 32B model, default config uses GPU 0,1 and tensor parallel 2.
bash scripts/local_vlm_serving/serve_model.sh Qwen2.5-VL-32B-Instruct

# 78B model, default config uses all 4 A100s.
bash scripts/local_vlm_serving/serve_model.sh InternVL2_5-78B --port 8001

# Small model on one chosen GPU.
bash scripts/local_vlm_serving/serve_model.sh Pixtral-12B --gpus 2 --port 8002
```

Use `--dry-run` to print the exact command without starting the server:

```bash
bash scripts/local_vlm_serving/serve_model.sh Qwen2.5-VL-32B-Instruct --dry-run
```

## 4xA100 Scheduling

Recommended sequential serving for clean benchmark outputs:

- `InternVL2_5-78B`: GPUs `0,1,2,3`, tensor parallel 4.
- `Qwen2.5-VL-32B-Instruct`: GPUs `0,1`, tensor parallel 2.
- Other vLLM models: one A100 each.
- `Janus-Pro-1B` and `Janus-Pro-7B`: served through the lightweight
  Transformers fallback endpoint because the default vLLM launcher is not
  enabled for them here. Override `--backend vllm` only after validating the
  endpoint.

The local model list is in `config/local_vlm_serving.yaml`.
