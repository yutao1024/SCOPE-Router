# RouterSFT Runbook

This is the operational runbook for running RouterSFT generation on the server.
Default repo path:

```bash
cd <PROJECT_ROOT>
```

## 1. Install Environments

Install all auxiliary conda environments:

```bash
bash scripts/setup_router_sft_conda_envs.sh all
```

Install selected environments only:

```bash
bash scripts/setup_router_sft_conda_envs.sh code ocr browsecomp swebench swe-agent
```

If external repo paths differ, override them:

```bash
EXTERNAL_ROOT=<EXTERNAL_ROOT> \
bash scripts/setup_router_sft_conda_envs.sh all
```

Default environments:

```text
router-code-eval  BigCodeBench / LiveCodeBench
ocr               PaddleOCR HTTP server
browsecomp-plus   BrowseComp-Plus retriever
swebench          SWE-bench official verifier
swe-agent         mini-swe-agent executor
```

## 2. Start Services

Start BrowseComp-Plus retrievers and OCR servers:

```bash
cd <PROJECT_ROOT>
bash scripts/start_router_sft_services.sh start
```

Default service layout:

```text
BrowseComp retrievers: 8765 8766 8767 8768 on GPU 0 1 2 3
OCR servers:           8775 8776 8777 8778 on GPU 4 5 6 7
```

Override GPU or port layout if needed:

```bash
RETRIEVER_PORTS="8765 8766" \
RETRIEVER_GPUS="0 1" \
OCR_PORTS="8775 8776" \
OCR_GPUS="4 5" \
bash scripts/start_router_sft_services.sh start
```

Check services:

```bash
bash scripts/start_router_sft_services.sh check
```

Show listening status:

```bash
bash scripts/start_router_sft_services.sh status
```

Restart services:

```bash
bash scripts/start_router_sft_services.sh restart
```

Stop services:

```bash
bash scripts/start_router_sft_services.sh stop
```

## 3. Generate Data

Start generation with `nohup` so SSH disconnects do not stop the run:

```bash
cd <PROJECT_ROOT>
conda activate browsecomp-plus

nohup bash scripts/run_router_sft_full_shards.sh > logs/router_sft_full_shards.log 2>&1 &
echo $! > logs/router_sft_full_shards.pid
```

Default shard layout:

```text
code_no_swebench  8 shards, max-workers 1
vqa               8 shards, max-workers 1, OCR ports 8775-8778
browsecomp_plus   4 shards, max-workers 1, retriever ports 8765-8768
```

Override shard counts:

```bash
N_CODE=4 N_VQA=4 N_BROWSE=2 \
nohup bash scripts/run_router_sft_full_shards.sh > logs/router_sft_full_shards.log 2>&1 &
echo $! > logs/router_sft_full_shards.pid
```

Override output directory:

```bash
OUT=outputs/my_router_sft_run_$(date +%Y%m%d_%H%M%S) \
nohup bash scripts/run_router_sft_full_shards.sh > logs/router_sft_full_shards.log 2>&1 &
echo $! > logs/router_sft_full_shards.pid
```

## 4. Monitor

Main generation log:

```bash
tail -f logs/router_sft_full_shards.log
```

Find the output directory:

```bash
grep '^\[run-shards\] OUT=' logs/router_sft_full_shards.log
```

Shard logs, after setting `OUT` to the printed output path:

```bash
tail -f "$OUT"/code_no_swebench/shards/run_00.log
tail -f "$OUT"/vqa/shards/run_00.log
tail -f "$OUT"/browsecomp_plus/shards/run_00.log
```

Service logs:

```bash
tail -f logs/browsecomp_plus_retriever_8765.log
tail -f logs/ocr_server_8775.log
```

Check output row counts:

```bash
wc -l "$OUT"/code_no_swebench/executor_results.jsonl
wc -l "$OUT"/vqa/executor_results.jsonl
wc -l "$OUT"/browsecomp_plus/executor_results.jsonl
```

## 5. Stop Generation

Stop only the data generation run:

```bash
cd <PROJECT_ROOT>

if [ -f logs/router_sft_full_shards.pid ]; then
  kill "$(cat logs/router_sft_full_shards.pid)" 2>/dev/null || true
fi

pkill -f 'scripts/generate_router_sft.py'
pkill -f 'scripts/run_mini_agent_executor.py'
```

Check for leftovers:

```bash
pgrep -af 'scripts/generate_router_sft.py|scripts/run_mini_agent_executor.py'
```

If anything remains and you really want to stop it:

```bash
pkill -9 -f 'scripts/generate_router_sft.py'
pkill -9 -f 'scripts/run_mini_agent_executor.py'
```

## 6. Stop Services

Stop retriever and OCR services:

```bash
cd <PROJECT_ROOT>
bash scripts/start_router_sft_services.sh stop
```

Confirm ports are free:

```bash
bash scripts/start_router_sft_services.sh status
```

## Notes

Do not put `conda activate browsecomp-plus` inside
`configs/router_sft_env.example.sh`. Activate the environment in the shell or
use `conda run` around the top-level command. The env file is sourced by scripts
and should only set environment variables.

If service startup reports an invalid `JAVA_HOME` or `JVM_PATH`, the service
script will fall back to the Java inside the `browsecomp-plus` conda env when
available.
