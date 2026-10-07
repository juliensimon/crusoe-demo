#!/usr/bin/env python3
"""Crusoe Intelligence Foundry, end to end, from one terminal.

    uv run demo.py all            # probe → prepare → upload → estimate → train → watch
                                  #       → checkpoints → lora → deploy → wait-ready → eval
    uv run demo.py <stage>        # any single stage; completed stages are skipped
    uv run demo.py <stage> --force

Every stage records what it did in state.json, so a killed run resumes where it stopped.
Nothing is ever deleted unless you run `cleanup`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

import httpx
from dotenv import load_dotenv
from openai import NotFoundError, OpenAI, PermissionDeniedError

Args = argparse.Namespace

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
STATE_FILE = ROOT / "state.json"

INTEL_URL = "https://api.intelligence.crusoecloud.com/v1"
INFER_URL = "https://api.inference.crusoecloud.com/v1"
CLOUD_HOSTS = ["https://api.crusoecloud.com", "https://api.cloud.crusoe.ai"]
HF_BASE = "https://huggingface.co/datasets/mteb/banking77/resolve/main"

DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEPLOYMENT_NAME = "banking-intents"
BASE_DEPLOYMENT_NAME = "qwen-base"
N_INTENTS, N_TRAIN, N_VAL, N_TEST = 15, 40, 6, 12
POLL_SECONDS = 30
TERMINAL_JOB_STATES = ("succeeded", "failed", "cancelled")
LORA_TIMEOUT_S = 10 * 60
DEPLOY_TIMEOUT_S = 45 * 60  # docs: provisioning "can take up to 40 minutes"

# ───────────────────────────── config / state ─────────────────────────────


def env(name: str, required: bool = True) -> str:
    value = os.environ.get(name, "").strip()
    if required and not value:
        sys.exit(f"missing {name} — copy .env.example to .env and fill it in")
    return value


def load_state() -> dict:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}


def save_state(state: dict) -> None:
    """Atomic write: state.json holds the only record of what is billing, so never truncate it."""
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(STATE_FILE)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def elapsed(since_iso: str) -> float:
    return (datetime.now(UTC) - datetime.fromisoformat(since_iso)).total_seconds()


def header(stage: str) -> None:
    print(f"\n── {stage} ──")


def die(msg: str, code: int = 1) -> NoReturn:
    print(f"✗ {msg}")
    sys.exit(code)


# ───────────────────────────── clients ─────────────────────────────


def intel() -> OpenAI:
    return OpenAI(api_key=env("CRUSOE_API_KEY"), base_url=INTEL_URL)


def infer() -> OpenAI:
    return OpenAI(api_key=env("CRUSOE_API_KEY"), base_url=INFER_URL)


def intel_get(path: str) -> dict:
    r = httpx.get(INTEL_URL + path, headers=auth(), timeout=60)
    r.raise_for_status()
    return r.json()


def intel_post(path: str, body: dict) -> dict:
    r = httpx.post(INTEL_URL + path, headers=auth(), json=body, timeout=60)
    r.raise_for_status()
    return r.json()


def auth() -> dict:
    return {"Authorization": f"Bearer {env('CRUSOE_API_KEY')}"}


def cloud(method: str, path: str, body: dict | None = None, host: str | None = None,
          fatal_auth: bool = True) -> httpx.Response:
    """Call the Crusoe Cloud API under /v1/projects/{project}/foundry/…

    The host comes from CRUSOE_CLOUD_HOST, which `main` fills from the one `probe` validated,
    so every call for a resource goes to the same host and project.
    """
    host = host or env("CRUSOE_CLOUD_HOST", required=False) or CLOUD_HOSTS[0]
    url = f"{host}/v1/projects/{env('CRUSOE_PROJECT_ID')}/foundry{path}"
    r = httpx.request(method, url, headers=auth(), json=body, timeout=60)
    if fatal_auth and r.status_code in (401, 403):
        die(f"{method} {path} → {r.status_code}. The Intelligence key was rejected by the "
            f"deployment API on {host}. Check CRUSOE_PROJECT_ID / CRUSOE_CLOUD_HOST.", 2)
    return r


# ───────────────────────────── stages ─────────────────────────────


def resolve_model(models: list[dict], wanted: str) -> dict | None:
    """Accept either the opaque id (model-qwen-…) or the HF-style model_name, case-insensitive."""
    w = wanted.lower()
    return next((m for m in models if m.get("id", "").lower() == w
                 or (m.get("model_name") or "").lower() == w), None)


def stage_probe(state: dict, args: Args) -> None:
    header("probe")
    models = intel_get("/models").get("data", [])
    base_models = [m for m in models if m.get("model_type") == "base_model"]
    print(f"{len(models)} models visible on {INTEL_URL} ({len(base_models)} base models)")
    print(f"{'model_name':44} {'fine-tune':>9} {'inference':>9}  id")
    wanted = args.model.lower()
    for m in sorted(base_models, key=lambda m: m.get("model_name") or ""):
        mark = "  ◀" if wanted in ((m.get("model_name") or "").lower(), (m.get("id") or "").lower()) else ""
        print(f"{m.get('model_name') or '?':44} {str(m.get('fine_tuning_available')):>9} "
              f"{str(m.get('inference_available')):>9}  {m.get('id')}{mark}")
    chosen = resolve_model(base_models, args.model)
    if not chosen:
        die(f"--model {args.model} not in the catalog above")
    if not chosen.get("fine_tuning_available"):
        die(f"{chosen['model_name']} is not fine-tunable")

    host_ok = None
    configured = env("CRUSOE_CLOUD_HOST", required=False)
    for host in dict.fromkeys([configured, *CLOUD_HOSTS] if configured else CLOUD_HOSTS):
        r = cloud("GET", "/selfserve/flavors", host=host, fatal_auth=False)
        if r.status_code == 200:
            host_ok = host
            break
        print(f"{host}: HTTP {r.status_code}")
    if not host_ok:
        die("no cloud host accepted the key for /selfserve/flavors", 2)
    flavors = r.json().get("flavors", [])
    lora = [f for f in flavors if f.get("lora_suitable")]
    print(f"\n{len(flavors)} self-serve flavors on {host_ok}, {len(lora)} LoRA-capable:")
    print(f"{'model':44} {'gpu':>6} {'x':>2} {'$/h':>7}  profile")
    chosen_name = (chosen.get("model_name") or "").lower()
    for f in sorted(lora, key=lambda f: float(f.get("price_per_hour") or 0)):
        mark = "  ◀" if (f.get("model_name") or "").lower() == chosen_name else ""
        print(f"{f.get('model_name', '?'):44} {f.get('gpu_type', '?'):>6} "
              f"{f.get('gpus_per_instance', '?'):>2} {float(f.get('price_per_hour') or 0):>7.2f}  "
              f"{f.get('flavor_type', '')}{mark}")

    r = cloud("GET", "/loras", host=host_ok)
    body = r.json() if r.status_code == 200 else {}
    loras = (body.get("lora_adapters") or []) if isinstance(body, dict) else body
    ready = sum(1 for a in loras if a.get("status") == "ready")
    print(f"\nserverless LoRA endpoint (/foundry/loras): HTTP {r.status_code}"
          + (f", {len(loras)} adapters in this project, {ready} ready" if r.status_code == 200
             else f" {r.text[:120]}"))

    print(f"\nusing {chosen['model_name']}  →  {chosen['id']}")
    state["model"] = chosen["model_name"]
    state["model_id"] = chosen["id"]
    state["cloud_host"] = host_ok
    save_state(state)


def download(name: str) -> Path:
    dst = DATA / f"raw_{name}.jsonl"
    if not dst.exists():
        tmp = dst.with_suffix(".part")
        with httpx.stream("GET", f"{HF_BASE}/{name}.jsonl", follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            tmp.write_bytes(b"".join(r.iter_bytes()))
        read_jsonl(tmp)  # a truncated download fails here, not later
        tmp.replace(dst)
    return dst


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def system_prompt(labels: list[str]) -> str:
    return ("You are a banking customer-support intent classifier. Reply with exactly one label "
            "from this list and nothing else: " + ", ".join(labels))


def build_splits(train_raw: list[dict], test_raw: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    """Pure: pick N_INTENTS labels, stratified N_TRAIN/N_VAL from train, N_TEST from test."""
    rng = random.Random(seed)
    labels = sorted({r["label_text"] for r in train_raw})
    rng.shuffle(labels)
    labels = sorted(labels[:N_INTENTS])
    system = system_prompt(labels)

    def by_label(rows):
        out = {label: [] for label in labels}
        for r in rows:
            if r["label_text"] in out:
                out[r["label_text"]].append(r["text"])
        return out

    def to_example(text: str, label: str) -> dict:
        return {"messages": [{"role": "system", "content": system},
                             {"role": "user", "content": text},
                             {"role": "assistant", "content": label}]}

    splits = {"train": [], "val": [], "test": []}
    tr, te = by_label(train_raw), by_label(test_raw)
    for label in labels:
        pool = tr[label][:]
        rng.shuffle(pool)
        if len(pool) < N_TRAIN + N_VAL or len(te[label]) < N_TEST:
            raise ValueError(f"not enough rows for {label}")
        splits["train"] += [to_example(t, label) for t in pool[:N_TRAIN]]
        splits["val"] += [to_example(t, label) for t in pool[N_TRAIN:N_TRAIN + N_VAL]]
        tpool = te[label][:]
        rng.shuffle(tpool)
        splits["test"] += [to_example(t, label) for t in tpool[:N_TEST]]
    for rows in splits.values():
        rng.shuffle(rows)
    seen = {normalize_text(ex["messages"][1]["content"]) for k in ("train", "val") for ex in splits[k]}
    leaked = [ex for ex in splits["test"] if normalize_text(ex["messages"][1]["content"]) in seen]
    if leaked:  # Banking77 has a few texts in both splits; never let one reach our test set
        raise ValueError(f"{len(leaked)} test messages also appear in train/val")
    splits["labels"] = labels
    return splits


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def stage_prepare(state: dict, args: Args) -> None:
    header("prepare")
    if state.get("data") and not args.force:
        print(f"skip — data already prepared ({state['data']})")
        return
    DATA.mkdir(exist_ok=True)
    splits = build_splits(read_jsonl(download("train")), read_jsonl(download("test")))
    for name in ("train", "val", "test"):
        write_jsonl(DATA / f"{name}.jsonl", splits[name])
    print(f"{len(splits['labels'])} intents: {', '.join(splits['labels'])}")
    print(f"train {len(splits['train'])} · val {len(splits['val'])} · test {len(splits['test'])} rows")
    sample = splits["train"][0]["messages"]
    print(f"\nsample row:\n  user:      {sample[1]['content']}\n  assistant: {sample[2]['content']}")
    state["data"] = {k: len(splits[k]) for k in ("train", "val", "test")}
    state["labels"] = splits["labels"]
    save_state(state)


def stage_upload(state: dict, args: Args) -> None:
    header("upload")
    if state.get("files") and not args.force:
        print(f"skip — files already uploaded: {state['files']}")
        return
    client = intel()
    files = {}
    for name in ("train", "val"):
        with open(DATA / f"{name}.jsonl", "rb") as fh:
            try:
                f = client.files.create(file=fh, purpose="fine-tune")
            except PermissionDeniedError as e:
                die(f"upload refused: {e.body}\n  Your user needs edit rights on project "
                    f"{env('CRUSOE_PROJECT_ID')} (or use a project you own).", 2)
        files[name] = f.id
        print(f"{name}.jsonl → {f.id}  ({f.bytes} bytes)")
    state["files"] = files
    save_state(state)


def job_body(state: dict) -> dict:
    if not state.get("files"):
        die("no uploaded files in state — run `upload` first")
    return {
        "model": state["model_id"],
        "training_file": state["files"]["train"],
        "validation_file": state["files"]["val"],
        "suffix": DEPLOYMENT_NAME,
        "method": {"type": "supervised", "supervised": {"hyperparameters": {
            "n_epochs": 3, "batch_size": "auto", "learning_rate_multiplier": "auto",
            "lora_rank": 16, "overlong_row_behavior": "drop"}}},
    }


def stage_estimate(state: dict, args: Args) -> None:
    header("estimate")
    est = intel_post("/fine_tuning/jobs/estimate_price", job_body(state))
    price = next((v for k, v in est.items() if "price" in k or "cost" in k), None)
    print(f"estimated price: {price if price is not None else json.dumps(est)}")
    state["estimate"] = est
    save_state(state)


def stage_train(state: dict, args: Args) -> None:
    header("train")
    if state.get("job_id"):
        if not args.force:
            print(f"skip — job already created: {state['job_id']} ({state.get('job_status')})")
            return
        if state.get("job_status") not in TERMINAL_JOB_STATES:
            die(f"job {state['job_id']} is {state.get('job_status')} — wait for it (`watch`) "
                "before starting another")
    job = intel().fine_tuning.jobs.create(**job_body(state))
    print(f"job {job.id}  model {job.model}  status {job.status}")
    state.update(job_id=job.id, job_status=job.status, job_created_at=now(), seen_events=[])
    save_state(state)


def metrics_line(metrics: dict) -> str:
    """Latest value of each series: {"metrics":[{"metric_name":"ev_loss","data":[{...,"value":..}]}]}"""
    latest = {}
    for series in metrics.get("metrics", []):
        data = series.get("data") or []
        if data:
            last = data[-1]
            value = last.get("value", last.get("y", last)) if isinstance(last, dict) else last
            latest[series["metric_name"]] = value
    if not latest:
        return ""
    step = latest.get("ev_current_step")
    total = latest.get("ev_total_steps")
    parts = [f"step {int(step)}/{int(total)}"] if step is not None and total is not None else []
    for key, label in (("ev_loss", "train_loss"), ("ev_eval_loss", "valid_loss")):
        if key in latest and isinstance(latest[key], (int, float)):
            parts.append(f"{label} {latest[key]:.4f}")
    return "  ".join(parts)


def event_order(ev) -> tuple[int, int]:
    """Events are ftevent-<job>-<n>; n breaks ties between events in the same second."""
    tail = ev.id.rsplit("-", 1)[-1]
    return ev.created_at, int(tail) if tail.isdigit() else 0


def event_text(message: str) -> str:
    """Drop the server's ' state transitioned at <US-local time>' suffix; we print our own clock."""
    return re.sub(r"\s+state transitioned at .*$", "", message or "")


def stage_watch(state: dict, args: Args) -> None:
    header("watch")
    if state.get("job_status") == "succeeded" and not args.force:
        print(f"skip — job {state['job_id']} already succeeded")
        return
    client = intel()
    seen = set(state.get("seen_events", []))
    last_metric = None
    while True:
        job = client.fine_tuning.jobs.retrieve(state["job_id"])
        # The API does not guarantee page order (it has returned both newest- and oldest-first),
        # so sort on (timestamp, event index) and the log always reads top to bottom.
        events = client.fine_tuning.jobs.list_events(state["job_id"], limit=50).data
        for ev in sorted(events, key=event_order):
            if ev.id not in seen:
                seen.add(ev.id)
                clock = datetime.fromtimestamp(ev.created_at).strftime("%H:%M:%S")
                print(f"  {clock}  {event_text(ev.message)}")
        try:
            line = metrics_line(intel_get(f"/fine_tuning/jobs/{state['job_id']}/metrics"))
            if line and line != last_metric:
                last_metric = line
                print(f"  {line}")
        except httpx.HTTPStatusError:
            pass
        state.update(job_status=job.status, seen_events=sorted(seen),
                     fine_tuned_model=getattr(job, "fine_tuned_model", None))
        save_state(state)
        if job.status in TERMINAL_JOB_STATES:
            break
        time.sleep(POLL_SECONDS)
    mins = elapsed(state["job_created_at"]) / 60
    print(f"job {job.status} after {mins:.1f} min")
    if job.status != "succeeded":
        die(f"fine-tuning job {job.status}: {getattr(job, 'error', '')}")
    state["job_finished_at"] = now()
    save_state(state)


def stage_checkpoints(state: dict, args: Args) -> None:
    header("checkpoints")
    if state.get("job_status") != "succeeded":
        die(f"job {state.get('job_id')} is {state.get('job_status')} — run `watch` first")
    cps = intel().fine_tuning.jobs.checkpoints.list(state["job_id"]).data
    rows = []
    for cp in cps:
        m = cp.metrics
        # Two ids per checkpoint: `fine_tuned_model_id` (ftmodel-…, deployable) and
        # `fine_tuned_model_checkpoint` (a display name). Prefer the deployable one.
        name = getattr(cp, "fine_tuned_model_checkpoint", None)
        rows.append({"step": cp.step_number,
                     "id": getattr(cp, "fine_tuned_model_id", None) or name,
                     "name": name,
                     "train_loss": getattr(m, "train_loss", None),
                     "valid_loss": getattr(m, "valid_loss", None)})
    rows = [r for r in rows if (r["id"] or "").startswith("ftmodel-")]
    if not rows:
        die(f"no deployable checkpoint (ftmodel-…) returned for job {state['job_id']}")
    rows.sort(key=lambda r: r["step"])
    scored = [r for r in rows if r["valid_loss"] is not None]
    best = min(scored, key=lambda r: r["valid_loss"]) if scored else rows[-1]
    print(f"{'step':>6} {'train_loss':>11} {'valid_loss':>11}  model id")
    for r in rows:
        tl = f"{r['train_loss']:.4f}" if r["train_loss"] is not None else "-"
        vl = f"{r['valid_loss']:.4f}" if r["valid_loss"] is not None else "-"
        print(f"{r['step']:>6} {tl:>11} {vl:>11}  {r['id']}{'  ◀ best' if r is best else ''}")
    state["best_checkpoint"] = best
    save_state(state)


def chosen_adapter(state: dict) -> str:
    """The checkpoint `checkpoints` picked. Serverless and dedicated both serve this one."""
    best = (state.get("best_checkpoint") or {}).get("id")
    return best or die("no checkpoint chosen — run `checkpoints` first")


def stage_lora(state: dict, args: Args) -> None:
    """Load the adapter onto serverless inference via /foundry/loras.

    In the API reference (docs.crusoecloud.com/api/); no guide covers it yet.
    Loaded adapters expire after a while; for a long-lived endpoint, use a deployment.
    """
    header("lora")
    if state.get("lora"):
        if args.force:
            die(f"adapter {state['lora']['id']} is already loaded — run `cleanup` before re-loading")
        print(f"skip — adapter already loaded: {state['lora']}")
        return
    adapter = chosen_adapter(state)
    r = cloud("POST", "/loras", {"model_id": adapter})
    if r.status_code >= 300:
        die(f"load adapter → HTTP {r.status_code}: {r.text}")
    d = r.json()
    lora_id = d.get("id") or die(f"no adapter id in response: {d}")
    state["lora"] = {"id": lora_id, "model": adapter, "created_at": now()}  # record before waiting
    save_state(state)
    print(f"POST /foundry/loras {{model_id: {adapter}}} → {r.status_code}  "
          f"id {lora_id}  status {d.get('status')}")
    t0 = time.time()
    while d.get("status") in ("pending", "downloading", "loading", None):
        if time.time() - t0 > LORA_TIMEOUT_S:
            die(f"adapter still {d.get('status')} after {LORA_TIMEOUT_S // 60} min — "
                "`cleanup` removes it")
        time.sleep(5)
        d = cloud("GET", f"/loras/{lora_id}").json()
    if d.get("status") != "ready":
        die(f"adapter load ended in {d.get('status')}: {d.get('last_error')}")
    print(f"ready after {time.time() - t0:.0f}s — callable on {INFER_URL} as model={adapter}")


def flavor_price(flavor: dict) -> float:
    price = float(flavor.get("price_per_hour") or "nan")
    if not math.isfinite(price) or price <= 0:
        die(f"flavor {flavor.get('id')} has no usable price ({flavor.get('price_per_hour')!r})")
    return price


def remote_deployments() -> dict[str, dict]:
    """Deployments in the project that carry this script's names, whatever state.json says."""
    r = cloud("GET", "/selfserve/deployments")
    body = r.json() if r.is_success else {}
    items = body.get("deployments", []) if isinstance(body, dict) else body
    ours = (DEPLOYMENT_NAME, BASE_DEPLOYMENT_NAME)
    return {d.get("deployment_name"): d for d in items or [] if d.get("deployment_name") in ours}


def create_deployment(state: dict, key: str, name: str, adapter: str, flavor: dict) -> None:
    r = cloud("POST", "/selfserve/deployments", {
        "deployment_name": name, "fine_tuned_model": adapter,
        "flavor_id": flavor["id"], "replicas": 1})
    if r.status_code >= 300:
        die(f"create {name} → HTTP {r.status_code}: {r.text}")
    d = r.json()
    dep_id = d.get("id") or die(f"create {name} returned no id: {d} — check the console, it may be "
                                "billing")
    state[key] = {"id": dep_id, "name": name, "adapter": adapter or None, "created_at": now(),
                  "price_per_hour": flavor_price(flavor)}
    save_state(state)
    extra = f"  (${flavor_price(flavor):.2f}/h more)" if key == "base_deployment" else ""
    label = "base deployment" if key == "base_deployment" else "deployment"
    print(f"{label} {dep_id}  status {d.get('status')}  alias {name}{extra}")


def stage_deploy(state: dict, args: Args) -> None:
    header("deploy")
    wanted = [("deployment", DEPLOYMENT_NAME)]
    if args.with_base:
        wanted.append(("base_deployment", BASE_DEPLOYMENT_NAME))
    if args.force and any(state.get(k) for k, _ in wanted):
        die("a deployment is still recorded (and may be billing) — run `cleanup` before deploying again")
    todo = [(k, n) for k, n in wanted if not state.get(k)]
    for k, _ in wanted:
        if state.get(k):
            print(f"skip — {k.replace('_', ' ')} exists: {state[k]['id']}")
    if not todo:
        return

    # Refuse to create a duplicate of something this script already started but lost track of.
    strays = [n for n in remote_deployments() if n in dict(todo).values()]
    if strays:
        die(f"{', '.join(strays)} already exist(s) in the project but not in state.json — "
            "delete in the console (or `cleanup`) before deploying")

    adapter = chosen_adapter(state)
    flavors = cloud("GET", "/selfserve/flavors").json().get("flavors", [])
    base = state["model"].lower()
    candidates = [f for f in flavors if f.get("lora_suitable")
                  and base in (f.get("model_name") or "").lower()]
    if not candidates:
        print("no LoRA-capable flavor matches", state["model"], "— available:")
        for f in flavors:
            print(f"  {f.get('model_name')}  lora={f.get('lora_suitable')}  ${f.get('price_per_hour')}/h")
        die("pick a model that has a self-serve flavor")
    flavor = min(candidates, key=flavor_price)
    price = flavor_price(flavor)
    total = price * len(todo)
    if not math.isfinite(args.max_hourly) or total > args.max_hourly:
        die(f"{len(todo)} deployment(s) × ${price:.2f}/h = ${total:.2f}/h > --max-hourly {args.max_hourly}")
    print(f"flavor {flavor['id']}: {flavor.get('model_name')} on {flavor.get('gpus_per_instance')}×"
          f"{flavor.get('gpu_type')} ({flavor.get('flavor_type')})  ${price:.2f}/h")
    state["flavor"] = flavor
    for key, name in todo:
        # Undocumented for the API (the console has "Deploy a Base Model"): an empty
        # fine_tuned_model deploys the flavor's base model.
        create_deployment(state, key, name, "" if key == "base_deployment" else adapter, flavor)


def stage_wait_ready(state: dict, args: Args) -> None:
    header("wait-ready")
    dep = state.get("deployment") or die("no deployment in state — run `deploy` first")
    for d in (dep, state.get("base_deployment")):
        if d:
            wait_for_ready(state, d, args)


def wait_for_ready(state: dict, dep: dict, args: Args) -> None:
    if dep.get("ready_at") and not args.force:
        print(f"skip — {dep['name']} ({dep['id']}) was Ready at {dep['ready_at']}")
        return
    print(f"{dep['name']} ({dep['id']}):")
    last = None
    t0 = time.time()
    while True:
        r = cloud("GET", f"/selfserve/deployments/{dep['id']}")
        if r.status_code == 404:
            die(f"deployment {dep['id']} not found — deleted outside this script?")
        r.raise_for_status()
        d = r.json()
        status = (d.get("status") or "").lower()  # the API sends lowercase; the docs capitalise
        line = (f"{status}  replicas {d.get('available_replicas')}/{d.get('target_replicas')}"
                f"  ${d.get('hourly_cost', dep['price_per_hour'])}/h")
        if line != last:
            print(f"  {datetime.now().strftime('%H:%M:%S')}  {line}")
            last = line
        if status == "ready":
            break
        if status in ("failed", "deleted", "deleting"):
            activity = cloud("GET", f"/selfserve/deployments/{dep['id']}/activity").json()
            for ev in activity.get("events", [])[-5:]:
                print(f"    {ev.get('timestamp')}  {ev.get('title')}: {ev.get('message')}")
            die(f"deployment {status}")
        if time.time() - t0 > DEPLOY_TIMEOUT_S:
            die(f"still {status} after {DEPLOY_TIMEOUT_S // 60} min and still billing — "
                "re-run `wait-ready`, or `cleanup`")
        time.sleep(POLL_SECONDS)
    dep["ready_at"] = now()
    print(f"{dep['name']} ready after {elapsed(dep['created_at']) / 60:.1f} min")
    save_state(state)


def normalize_label(text: str) -> str:
    """First line, lowercased, punctuation stripped, spaces → underscores."""
    first = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    first = re.sub(r"[^a-z0-9_ ]", "", first.lower()).strip()
    return re.sub(r"\s+", "_", first)


# Qwen3.5 reasons by default, fine-tuned or not. Measured 2026-09-27 without this flag: on the
# serverless adapter, "Thinking Process: …" in `content` until max_tokens (512 tokens, 6.6 s, no
# label); on a dedicated deployment, the label plus a separate `reasoning` field (85 tokens, 0.9 s).
# With it: 4 tokens, 0.3 s, on both.
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


def classify(client: OpenAI, model: str, messages: list[dict]) -> str:
    try:
        r = client.chat.completions.create(model=model, messages=messages[:2],
                                           temperature=0, max_tokens=64, extra_body=NO_THINKING)
        return r.choices[0].message.content or ""
    except Exception as e:  # noqa: BLE001 — a failed call is a wrong answer, shown as such
        return f"<error: {type(e).__name__}: {str(e)[:80]}>"


def servable(client: OpenAI, model: str, messages: list[dict]) -> bool:
    """One probe call; False on 404 (not served) so eval can skip a candidate cleanly."""
    try:
        client.chat.completions.create(model=model, messages=messages[:2], max_tokens=1,
                                       extra_body=NO_THINKING)
        return True
    except NotFoundError:
        return False


def stage_eval(state: dict, args: Args) -> None:
    header("eval")
    test = read_jsonl(DATA / "test.jsonl")
    client = infer()
    dep = state.get("deployment") or {}
    base_dep = state.get("base_deployment") or {}
    candidates = [("base", base_dep.get("name") or state.get("model"))]
    candidates += [(f"zero-shot {m.split('/')[-1]}", m) for m in args.reference]
    candidates += [("adapter (serverless)", state.get("fine_tuned_model")),
                   ("deployment", dep.get("name"))]
    candidates = [(label, m) for label, m in candidates if m]
    gold = [ex["messages"][2]["content"] for ex in test]
    n = len(test)
    print(f"{n} held-out queries, {len(state['labels'])} intents, thinking disabled\n")
    lw = max(len(label) for label, _ in candidates)
    mw = max(len(m) for _, m in candidates)
    print(f"{'':{lw}}  {'model':{mw}}  {'correct':>8}  {'accuracy':>8}  {'time':>6}")

    results = {}
    for label, model in candidates:
        if not servable(client, model, test[0]["messages"]):
            print(f"{label:{lw}}  {model:{mw}}  not served (404), skipped")
            continue
        t0 = time.time()
        with ThreadPoolExecutor(args.workers) as pool:
            outs = list(pool.map(lambda ex, m=model: classify(client, m, ex["messages"]), test))
        secs = time.time() - t0
        correct = sum(normalize_label(o) == g for o, g in zip(outs, gold, strict=True))
        errors = sum(o.startswith("<error") for o in outs)
        results[label] = {"model": model, "correct": correct, "total": n, "errors": errors,
                          "seconds": round(secs, 1), "outputs": outs}
        flag = f"  ⚠ {errors} failed calls counted as wrong" if errors else ""
        print(f"{label:{lw}}  {model:{mw}}  {f'{correct}/{n}':>8}  {100 * correct / n:7.1f}%  "
              f"{secs:5.1f}s{flag}")

    if "deployment" in results and "adapter (serverless)" in results:
        a, d = results["adapter (serverless)"]["outputs"], results["deployment"]["outputs"]
        same = sum(x == y for x, y in zip(a, d, strict=True))
        print(f"\nserverless adapter vs deployment: identical answers on {same}/{n}")
    if "base" in results:
        b = results["base"]["outputs"]
        tuned = results.get("deployment") or results.get("adapter (serverless)")
        if tuned:
            print("\nexamples (base wrong, fine-tuned right):")
            shown = 0
            for ex, bo, to, g in zip(test, b, tuned["outputs"], gold, strict=True):
                if normalize_label(bo) != g and normalize_label(to) == g and shown < 3:
                    shown += 1
                    print(f"  Q: {ex['messages'][1]['content']}")
                    print(f"     gold:  {g}\n     base:  {bo.strip()[:120]!r}\n     tuned: {to.strip()!r}")
    else:
        tuned = results.get("deployment") or results.get("adapter (serverless)")
        outs = tuned["outputs"] if tuned else []
        wrong = [(ex, o, g) for ex, o, g in zip(test, outs, gold, strict=False)
                 if normalize_label(o) != g][:3]
        if wrong:
            print("\nexamples the fine-tuned model got wrong:")
            for ex, o, g in wrong:
                print(f"  Q: {ex['messages'][1]['content']}\n     gold: {g}   got: {o.strip()!r}")
    state["eval"] = {k: {kk: vv for kk, vv in v.items() if kk != "outputs"} for k, v in results.items()}
    state["eval_outputs"] = {k: v["outputs"] for k, v in results.items()}
    save_state(state)


def stage_chat(state: dict, args: Args) -> None:
    header("chat")
    dep = state.get("deployment") or die("no deployment")
    r = infer().chat.completions.create(
        model=dep["name"], temperature=0, max_tokens=64, extra_body=NO_THINKING,
        messages=[{"role": "system", "content": system_prompt(state["labels"])},
                  {"role": "user", "content": args.prompt}])
    print(f"model: {r.model}\n{r.choices[0].message.content}")


def gone(r: httpx.Response) -> bool:
    """Deleted now (2xx) or already gone (404). Anything else: keep it in state and retry."""
    return r.is_success or r.status_code == 404


def delete_deployment(dep_id: str, attempts: int = 6) -> httpx.Response:
    """DELETE, retrying 429s: right after a create the API answers "please wait a few seconds
    between changes to a deployment" (measured 2026-09-27)."""
    for _ in range(attempts - 1):
        r = cloud("DELETE", f"/selfserve/deployments/{dep_id}", fatal_auth=False)
        if r.status_code != 429:
            return r
        time.sleep(10)
    return cloud("DELETE", f"/selfserve/deployments/{dep_id}", fatal_auth=False)


def confirm_deleted(dep_id: str, timeout_s: int = 90) -> bool:
    """A 202 only means "accepted"; wait until the deployment is really gone (404)."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        r = cloud("GET", f"/selfserve/deployments/{dep_id}", fatal_auth=False)
        if r.status_code == 404 or (r.is_success and (r.json().get("status") or "").lower() == "deleted"):
            return True
        time.sleep(5)
    return False


def stage_cleanup(state: dict, args: Args) -> None:
    header("cleanup")
    failed = False
    if not (state.get("deployment") or state.get("base_deployment")):
        print("no deployment to delete")
    for key, label in (("deployment", "deployment"), ("base_deployment", "base deployment")):
        dep = state.get(key)
        if not dep:
            continue
        r = delete_deployment(dep["id"])
        print(f"DELETE {label} {dep['id']} → HTTP {r.status_code}")
        if not gone(r):
            print(f"  ✗ not deleted, still billing: {r.text[:200]}")
            failed = True
            continue
        if r.status_code == 404:
            print("  already gone")
        elif not confirm_deleted(dep["id"]):
            print("  ✗ deletion accepted but not confirmed after 90 s — kept in state.json")
            failed = True
            continue
        else:
            hours = elapsed(dep["created_at"]) / 3600
            rate = dep["price_per_hour"]
            print(f"  billing stopped — ${rate:.2f}/h × {hours:.2f} h ≈ ${rate * hours:.2f}")
        state["deleted_deployments"] = [*state.get("deleted_deployments", []), dep]
        del state[key]
        save_state(state)
    lora = state.get("lora")
    if lora:
        r = cloud("DELETE", f"/loras/{lora['id']}", fatal_auth=False)
        print(f"DELETE serverless adapter {lora['id']} → HTTP {r.status_code}")
        if gone(r):
            del state["lora"]
        else:
            failed = True
    if args.files and state.get("files"):
        client = intel()
        for name, fid in state["files"].items():
            client.files.delete(fid)
            print(f"deleted file {name} {fid}")
        del state["files"]
    if args.adapter and state.get("fine_tuned_model"):
        r = httpx.delete(f"{INTEL_URL}/models/{state['fine_tuned_model']}", headers=auth(), timeout=60)
        print(f"DELETE model {state['fine_tuned_model']} → HTTP {r.status_code}")
    save_state(state)
    strays = remote_deployments()
    for name, d in strays.items():
        print(f"  ✗ {name} ({d.get('id')}) still exists in the project, status {d.get('status')}")
    if failed or strays:
        die("some resources were not deleted — run `cleanup` again, or delete them in the console")


ALL = ["probe", "prepare", "upload", "estimate", "train", "watch", "checkpoints",
       "lora", "deploy", "wait-ready", "eval"]
STAGES = {"probe": stage_probe, "prepare": stage_prepare, "upload": stage_upload,
          "estimate": stage_estimate, "train": stage_train, "watch": stage_watch,
          "checkpoints": stage_checkpoints, "lora": stage_lora, "deploy": stage_deploy,
          "wait-ready": stage_wait_ready, "eval": stage_eval, "chat": stage_chat,
          "cleanup": stage_cleanup}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=ALL + ["chat", "cleanup", "all"])
    p.add_argument("--model", default=DEFAULT_MODEL, help="base model to fine-tune (probe shows options)")
    p.add_argument("--force", action="store_true", help="re-run the stage even if state says it's done")
    p.add_argument("--max-hourly", type=float, default=12.0, help="refuse flavors above this $/h")
    p.add_argument("--with-base", action="store_true",
                   help="deploy: also deploy the un-tuned base model (same flavor) for the before/after eval")
    p.add_argument("--workers", type=int, default=8, help="eval concurrency")
    p.add_argument("--reference", nargs="*", default=["google/gemma-4-31b-it"],
                   help="serverless models to score zero-shot alongside (`--reference` alone: none)")
    p.add_argument("--prompt", default="Why did my top-up by bank transfer come with a fee?",
                   help="for `chat`")
    p.add_argument("--files", action="store_true", help="cleanup: also delete uploaded files")
    p.add_argument("--adapter", action="store_true", help="cleanup: also delete the fine-tuned adapter")
    args = p.parse_args()

    load_dotenv(ROOT / ".env")
    sys.stdout.reconfigure(line_buffering=True)  # stream output when piped or backgrounded
    state = load_state()
    if state.get("cloud_host") and not os.environ.get("CRUSOE_CLOUD_HOST"):
        os.environ["CRUSOE_CLOUD_HOST"] = state["cloud_host"]
    if state.get("model") and args.model != state["model"] and args.stage != "probe":
        args.model = state["model"]
    stages = ALL if args.stage == "all" else [args.stage]
    try:
        for name in stages:
            STAGES[name](state, args)
    except KeyboardInterrupt:
        print(f"\ninterrupted — state saved. Resume with:  uv run demo.py {args.stage}")
        warn_billing(state)
        sys.exit(130)
    except BaseException:
        if args.stage != "cleanup":  # cleanup prints its own list
            warn_billing(state)
        raise
    if args.stage == "all":
        print("\ndone.")
        warn_billing(state)


def warn_billing(state: dict) -> None:
    running = [state[k] for k in ("deployment", "base_deployment") if state.get(k)]
    if running:
        rate = sum(d["price_per_hour"] for d in running)
        print(f"{len(running)} deployment(s) still billing (${rate:.2f}/h):  uv run demo.py cleanup")


if __name__ == "__main__":
    main()
