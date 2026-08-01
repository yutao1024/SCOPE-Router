# RouterSFT Conda Environments

This note records the native setup commands for the auxiliary environments used
by the RouterSFT pipeline. The helper script
`scripts/setup_router_sft_conda_envs.sh` wraps these commands, but the raw
commands below are easier to debug when a new machine has package-index,
CUDA, or path differences.

Default server paths used in this repo:

```bash
ROUTER_ROOT=<PROJECT_ROOT>
EXTERNAL_ROOT=<EXTERNAL_ROOT>
LIVECODEBENCH_DIR=<EXTERNAL_ROOT>/LiveCodeBench
SWEBENCH_DIR=<EXTERNAL_ROOT>/SWE-bench
MINI_SWE_AGENT_DIR=<EXTERNAL_ROOT>/mini-swe-agent
```

Change those paths on a new machine. If the CUDA runtime is not 12.6, also
change the Paddle wheel index in the OCR section, for example from `cu126` to
`cu118`.

## One-Command Installer

```bash
cd <PROJECT_ROOT>
bash scripts/setup_router_sft_conda_envs.sh all
```

Install selected environments only:

```bash
bash scripts/setup_router_sft_conda_envs.sh code ocr
bash scripts/setup_router_sft_conda_envs.sh swebench browsecomp swe-agent
```

Override paths:

```bash
EXTERNAL_ROOT=<EXTERNAL_ROOT> \
bash scripts/setup_router_sft_conda_envs.sh all
```

## Start Local Services

After installing the `ocr` and `browsecomp-plus` environments, start the local
OCR and BrowseComp-Plus retriever services with:

```bash
cd <PROJECT_ROOT>
bash scripts/start_router_sft_services.sh start
```

The default service layout is:

```text
BrowseComp-Plus retriever: 8765 8766 8767 8768 on GPU 0 1 2 3
PaddleOCR:                  8775 8776 8777 8778 on GPU 4 5 6 7
```

Override ports, GPUs, or environment names when needed:

```bash
BROWSECOMP_ENV=browsecomp-plus \
OCR_ENV=ocr \
RETRIEVER_PORTS="8765 8766" \
RETRIEVER_GPUS="0 1" \
OCR_PORTS="8775 8776" \
OCR_GPUS="4 5" \
bash scripts/start_router_sft_services.sh start
```

Useful commands:

```bash
bash scripts/start_router_sft_services.sh status
bash scripts/start_router_sft_services.sh check
bash scripts/start_router_sft_services.sh restart
bash scripts/start_router_sft_services.sh stop
```

Runtime exports for generation:

```bash
export MINI_AGENT_OCR_BACKEND=paddle_http
export MINI_AGENT_PADDLE_OCR_TIMEOUT=300
export BROWSECOMP_PLUS_RETRIEVER=qwen3-embedding-8b
export BROWSECOMP_PLUS_RETRIEVER_SERVER_TIMEOUT_MS=600000
```

## Code Eval

Used by BigCodeBench and LiveCodeBench official wrappers.

```bash
conda create -y -n router-code-eval python=3.10 pip
conda activate router-code-eval

python -m pip install -U pip setuptools wheel
python -m pip install numpy==1.24.4 pandas==1.5.3 matplotlib==3.8.2 pytest requests tqdm
python -m pip install bigcodebench

cd <EXTERNAL_ROOT>/LiveCodeBench
python -m pip install -e .
```

Verify:

```bash
python -c "from bigcodebench.data import get_bigcodebench; print('bigcodebench.data OK')"
python -c "from bigcodebench.eval import PASS; print('bigcodebench.eval OK')"
python -c "from bigcodebench.evaluate import check_correctness; print('bigcodebench.evaluate OK')"
python -c "import importlib.util; print('lcb_runner', importlib.util.find_spec('lcb_runner') is not None)"
```

Runtime exports:

```bash
export BIGCODEBENCH_EVAL_COMMAND='conda run -n router-code-eval python scripts/run_bigcodebench_eval.py --predictions {predictions} --result {result}'
export LIVECODEBENCH_EVAL_COMMAND='conda run -n router-code-eval python scripts/run_livecodebench_eval.py --predictions {predictions} --result {result}'
```

If BigCodeBench fails with `numpy.dtype size changed`, reinstall the ABI-bound
packages together:

```bash
conda activate router-code-eval
python -m pip install --force-reinstall --no-cache-dir \
  'numpy==1.24.4' \
  'pandas==1.5.3' \
  'datasets<3' \
  'pyarrow<16'
```

## OCR

Used by `scripts/ocr_server.py` for PaddleOCR-backed multimodal OCR.

```bash
conda create -y -n ocr python=3.10 pip
conda activate ocr

python -m pip install -U pip setuptools wheel

python -m pip install paddlepaddle-gpu==3.3.1 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
  --trusted-host www.paddlepaddle.org.cn

python -m pip install \
  paddleocr==3.7.0 \
  paddlex==3.7.2 \
  opencv-contrib-python==4.10.0.84 \
  pillow==12.3.0 \
  numpy==2.2.6 \
  pandas==2.3.3 \
  requests==2.34.2 \
  pydantic==2.13.4 \
  modelscope==1.38.1 \
  huggingface_hub==1.22.0
```

Verify:

```bash
python - <<'PY'
import paddle
from paddleocr import PaddleOCR
print("paddle", paddle.__version__)
print("cuda compiled", paddle.device.is_compiled_with_cuda())
print("paddleocr import OK")
PY
```

Start service:

```bash
cd <PROJECT_ROOT>
CUDA_VISIBLE_DEVICES=0 conda run -n ocr python scripts/ocr_server.py \
  --host 127.0.0.1 \
  --port 8766 \
  --lang ch \
  --device gpu:0
```

Runtime exports:

```bash
export MINI_AGENT_OCR_BACKEND=paddle_http
export MINI_AGENT_PADDLE_OCR_URL=http://127.0.0.1:8766/ocr
export MINI_AGENT_PADDLE_OCR_TIMEOUT=60
```

## SWE-bench

Used by official SWE-bench verification.

```bash
conda create -y -n swebench python=3.10 pip
conda activate swebench

python -m pip install -U pip setuptools wheel

python -m pip install \
  numpy==2.2.6 \
  pandas==2.3.3 \
  datasets==5.0.0 \
  pyarrow==24.0.0 \
  docker==7.1.0 \
  GitPython==3.1.50 \
  unidiff==0.7.5 \
  modal==1.5.1 \
  requests==2.34.2 \
  tqdm==4.68.4 \
  beautifulsoup4==4.15.0 \
  python-dotenv==1.2.2 \
  rich==15.0.0 \
  ghapi==2.0.0 \
  pre-commit==4.6.0

cd <EXTERNAL_ROOT>/SWE-bench
python -m pip install -e .
```

Verify:

```bash
python - <<'PY'
import docker
import swebench
import datasets
import pandas
import numpy
print("swebench OK", getattr(swebench, "__version__", "editable"))
print("datasets", datasets.__version__)
print("pandas", pandas.__version__)
print("numpy", numpy.__version__)
print("docker import OK")
PY
docker ps
```

Runtime export:

```bash
export SWEBENCH_CONDA_ENV=swebench
```

## BrowseComp-Plus Search

Used by `scripts/browsecomp_plus_retriever.py`.

```bash
conda create -y -n browsecomp-plus python=3.10 pip
conda activate browsecomp-plus

python -m pip install -U pip setuptools wheel

python -m pip install \
  torch==2.13.0 \
  transformers==5.13.0 \
  sentence-transformers==5.6.0 \
  tokenizers==0.22.2 \
  safetensors==0.8.0 \
  sentencepiece==0.2.1

python -m pip install \
  pyserini==1.2.0 \
  onnxruntime==1.23.2 \
  openai==2.44.0 \
  tiktoken==0.13.0 \
  fastapi==0.139.0 \
  uvicorn==0.51.0 \
  Flask==3.1.3 \
  mcp==1.28.1 \
  pydantic==2.13.4 \
  pydantic-settings==2.14.2 \
  pandas==2.3.3 \
  numpy==2.2.6 \
  scipy==1.15.3 \
  scikit-learn==1.7.2 \
  requests==2.34.2 \
  tqdm==4.68.4 \
  huggingface_hub==1.22.0

conda install -c conda-forge openjdk=21 -y
```

Verify:

```bash
python - <<'PY'
import torch
import transformers
import sentence_transformers
import pyserini
import sklearn
import openai
import fastapi
import numpy
import pandas
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("sentence-transformers OK")
print("pyserini OK")
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("browsecomp-plus deps OK")
PY
java -version
```

Start service:

```bash
cd <PROJECT_ROOT>
conda run --no-capture-output -n browsecomp-plus python scripts/browsecomp_plus_retriever.py \
  --serve \
  --warmup \
  --host 127.0.0.1 \
  --port 8765 \
  --retriever qwen3-embedding-8b
```

Health check:

```bash
curl -s http://127.0.0.1:8765/health | python -m json.tool
```

Runtime exports:

```bash
export BROWSECOMP_PLUS_RETRIEVER_SERVER_URL=http://127.0.0.1:8765
export BROWSECOMP_PLUS_MAX_AGENT_LOOP_TURNS=4
export BROWSECOMP_PLUS_MAX_SEARCH_CALLS_PER_ROUND=2
```

## Mini SWE Agent

Used by SWE-bench real-repo mini-agent execution.

```bash
conda create -y -n swe-agent python=3.10 pip
conda activate swe-agent

python -m pip install -U pip setuptools wheel

python -m pip install \
  datasets==5.0.0 \
  numpy==2.2.6 \
  pandas==2.3.3 \
  pyarrow==24.0.0 \
  requests==2.34.2 \
  openai==2.44.0 \
  litellm==1.91.1 \
  tiktoken==0.13.0 \
  pillow==12.3.0 \
  python-dotenv==1.2.2 \
  rich==15.0.0 \
  textual==8.2.8 \
  typer==0.26.8 \
  prompt_toolkit==3.0.52 \
  pydantic==2.13.4 \
  jsonschema==4.26.0 \
  tqdm==4.68.4 \
  PyYAML==6.0.3

cd <EXTERNAL_ROOT>/mini-swe-agent
python -m pip install -e .
```

Verify:

```bash
python - <<'PY'
import minisweagent
import litellm
import openai
import datasets
import pandas
import numpy
print("mini-swe-agent OK", getattr(minisweagent, "__version__", "editable"))
print("litellm", getattr(litellm, "__version__", "OK"))
print("openai", openai.__version__)
print("datasets", datasets.__version__)
print("pandas", pandas.__version__)
print("numpy", numpy.__version__)
PY
```

Runtime exports:

```bash
export MINI_AGENT_CONDA_ENV=swe-agent
export SWEBENCH_MINI_AGENT_COMMAND='conda run -n swe-agent python scripts/run_swebench_mini_agent_executor.py --input {input} --output {output} --model {executor_model_ref} --model-ref {executor_model_ref}'
```

## Common Smoke Commands

Code official evaluator smoke:

```bash
mkdir -p runs/code_official_eval_smoke
python scripts/generate_router_sft.py \
  --tasks openclaw_tasks/code_debug_edit/tasks.jsonl \
  --results-out runs/code_official_eval_smoke/results.jsonl \
  --sft-out runs/code_official_eval_smoke/sft.jsonl \
  --summary-out runs/code_official_eval_smoke/summary.json \
  --category code_debug_edit \
  --executor-backend raw_api \
  --source-dataset bigcode/bigcodebench \
  --source-dataset livecodebench/code_generation \
  --limit-calls 10
```

Single task:

```bash
python scripts/generate_router_sft.py \
  --tasks openclaw_tasks/code_debug_edit/tasks.jsonl \
  --results-out runs/one_task/results.jsonl \
  --sft-out runs/one_task/sft.jsonl \
  --summary-out runs/one_task/summary.json \
  --category code_debug_edit \
  --executor-backend raw_api \
  --task-id code-a9e31ae6a354cfb6 \
  --rerun-failed
```
