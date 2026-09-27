# crusoe-demo

[![CI](https://github.com/juliensimon/crusoe-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/juliensimon/crusoe-demo/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Fine-tune, deploy and evaluate a model on **Crusoe Intelligence Foundry**, end to end, from one
Python file.

This is the code from my YouTube video on Crusoe Intelligence Foundry. The video is sponsored by
Crusoe. The code, the measurements and the opinions are mine.

The script fine-tunes **Qwen3.5-9B** with a LoRA adapter on 600 real banking customer-support
messages (15 intents from Banking77). It serves the adapter two ways, on serverless inference and
on a dedicated NVIDIA H100 deployment. It then scores both against the base model and a larger
model prompted zero-shot, on 180 held-out messages.

## Results

Measured on 2026-09-27, 180 held-out messages, 15 intents, temperature 0, thinking disabled:

| Model | Served on | Correct | Accuracy |
|---|---|---:|---:|
| Qwen3.5-9B, base | dedicated NVIDIA H100 | 127/180 | 70.6% |
| Gemma 4 31B, zero-shot | serverless | 140/180 | 77.8% |
| **Qwen3.5-9B + LoRA adapter** | serverless | 162/180 | **90.0%** |
| **Qwen3.5-9B + LoRA adapter** | dedicated NVIDIA H100 | 162/180 | **90.0%** |

The serverless adapter and the dedicated deployment gave the same answer on 179 of 180 messages.

| Step | Measured |
|---|---|
| Price estimate for the fine-tuning job | $0.12 |
| Job submitted → training finished | 5 min 39 s |
| Adapter loaded on serverless | 26 s |
| Dedicated deployment ready | 5.6 min |
| GPU time, two deployments, ~7 min each | $0.58 each |

Your numbers will differ. Zero-shot scores moved by ±1 between runs; the fine-tuned scores did not.

## How it works

`demo.py` is a resumable stage machine. Each stage records what it did in `state.json`, so a
failed run resumes where it stopped, and a stage that already ran is skipped (use `--force` to
re-run it).

| Stage | What it does |
|---|---|
| `probe` | Lists fine-tunable models and deployment flavors (GPU, profile, $/h); picks the model |
| `prepare` | Downloads Banking77, builds train/val/test splits (seed 42) in OpenAI chat format |
| `upload` | `client.files.create(purpose="fine-tune")` |
| `estimate` | Asks the API what the job will cost before creating it |
| `train` | `client.fine_tuning.jobs.create(...)`: 3 epochs, LoRA rank 16 |
| `watch` | Polls job events and training/validation loss every 30 s until the job ends |
| `checkpoints` | Lists checkpoints and picks the one with the lowest validation loss |
| `lora` | Loads the chosen checkpoint onto serverless inference (see the note below) |
| `deploy` | Deploys the same checkpoint on the cheapest LoRA-capable flavor for the model; `--with-base` also deploys the base model |
| `wait-ready` | Polls until the deployments are ready |
| `chat` | One chat completion against the deployment |
| `eval` | Scores base, zero-shot reference(s), serverless adapter and deployment |
| `cleanup` | Deletes the deployments and the serverless adapter, waits until they are gone, and flags any deployment with this script's names still in the project. **Run it.** |

The script uses three Crusoe endpoints:

| Host | What for | Client |
|---|---|---|
| `api.intelligence.crusoecloud.com/v1` | Files, fine-tuning jobs, checkpoints, model catalog (OpenAI-compatible) | OpenAI SDK + `httpx` for estimate and metrics |
| `api.inference.crusoecloud.com/v1` | Chat completions: serverless models, loaded adapters, deployments (OpenAI-compatible) | OpenAI SDK |
| `api.crusoecloud.com/v1/projects/{id}/foundry` | Dedicated deployments, flavors, serverless adapter loading | `httpx` |

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env        # add your Intelligence API key and project ID
uv sync --group dev
uv run pytest               # pure-function tests; the API stages are not mocked

uv run demo.py probe
uv run demo.py prepare
uv run demo.py upload
uv run demo.py estimate
uv run demo.py train
uv run demo.py watch
uv run demo.py checkpoints
uv run demo.py lora
uv run demo.py deploy --with-base
uv run demo.py wait-ready
uv run demo.py chat
uv run demo.py eval
uv run demo.py cleanup      # deployments bill by the hour until you run this
```

`uv run demo.py all --with-base` runs `probe` through `eval` in one go.

> **Cost warning.** `deploy` picks the cheapest LoRA-capable flavor for the model (for Qwen3.5-9B on
> 2026-09-27: 1× NVIDIA H100 at $5.50/hour), billed per second until deleted, and refuses to go
> above `--max-hourly` (default $12/h for everything it creates). `cleanup` deletes only what `state.json` says this script
> created, so don't delete `state.json` before running it.

## Things worth knowing

These were true on 2026-09-27.

- **Use an Intelligence API key, created with the right project selected.** Crusoe has two kinds of
  keys. Cloud API keys are for infrastructure and don't work here. An Intelligence API key is bound
  to the project that is selected in the console when you create it. With the wrong project, reads
  work but every write fails with a permission error.
- **Turn thinking off for Qwen3.5.** It reasons by default, even after fine-tuning on data with no
  reasoning in it. Every call in the script sends
  `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`. Without it, on the serverless
  adapter, the model used all 512 tokens reasoning and never returned a label. On the dedicated
  deployment it returned the label but used 85 tokens instead of 4.
- **Serverless adapter loading is undocumented.** The `lora` stage uses `POST /foundry/loras`, which
  is in the API spec but not in the docs. The docs describe serverless inference as base models
  only. Call the adapter by its `ftmodel-…` ID, not its `endpoint_alias`. Loaded adapters expire
  after a while; the documented way to serve a fine-tuned model is a dedicated deployment.
- **Deploying the base model** (`--with-base`) sends `"fine_tuned_model": ""`, an empty string.
- **Deployment status strings are lowercase** on the wire (`creating`, `ready`).
- **The fine-tuning `model` parameter is the catalog ID** (`model-qwen-qwen3-5-9b-…`), not the
  Hugging Face name. `--model` accepts either; `probe` resolves it.

## Data

Banking77 by PolyAI (Casanueva et al., 2020), licensed CC-BY-4.0:
<https://huggingface.co/datasets/PolyAI/banking77>. The script downloads it from the
`mteb/banking77` copy on the Hugging Face Hub.

## Links

- Crusoe Serverless Fine-Tuning: <https://docs.crusoecloud.com/serverless-fine-tuning/how-it-works/>
- Crusoe Self-Serve Deployments: <https://docs.crusoecloud.com/self-serve-deployments/quickstart/>

## License

MIT
