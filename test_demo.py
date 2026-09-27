"""Pure-function checks. The API stages are verified against the real service, not mocked."""
import json

import demo


def fake_rows(n_labels=77, per_label=130, split="train"):
    rows = []
    for i in range(n_labels):
        for j in range(per_label):
            rows.append({"text": f"{split} query {i}-{j}", "label": i, "label_text": f"intent_{i:02d}"})
    return rows


def test_build_splits_sizes_and_schema():
    splits = demo.build_splits(fake_rows(), fake_rows(per_label=40, split="test"))
    assert len(splits["labels"]) == demo.N_INTENTS
    assert len(splits["train"]) == demo.N_INTENTS * demo.N_TRAIN
    assert len(splits["val"]) == demo.N_INTENTS * demo.N_VAL
    assert len(splits["test"]) == demo.N_INTENTS * demo.N_TEST
    for name in ("train", "val", "test"):
        for ex in splits[name]:
            roles = [m["role"] for m in ex["messages"]]
            assert roles == ["system", "user", "assistant"]
            assert ex["messages"][2]["content"] in splits["labels"]
            json.dumps(ex)  # serialisable


def test_build_splits_train_val_disjoint_and_deterministic():
    a = demo.build_splits(fake_rows(), fake_rows(per_label=40, split="test"))
    b = demo.build_splits(fake_rows(), fake_rows(per_label=40, split="test"))
    assert a == b
    train_q = {ex["messages"][1]["content"] for ex in a["train"]}
    val_q = {ex["messages"][1]["content"] for ex in a["val"]}
    assert not train_q & val_q


def test_build_splits_fails_loud_on_short_label():
    rows = fake_rows(per_label=10)
    try:
        demo.build_splits(rows, fake_rows(per_label=40, split="test"))
    except ValueError as e:
        assert "not enough rows" in str(e)
    else:
        raise AssertionError("short label must raise")


def test_normalize_label_handles_rambling_base_model():
    assert demo.normalize_label("card_arrival") == "card_arrival"
    assert demo.normalize_label("Card arrival") == "card_arrival"
    assert demo.normalize_label("  card_arrival.\n\nBecause the user asks…") == "card_arrival"
    assert demo.normalize_label("The intent is card_arrival") != "card_arrival"  # prose is wrong
    assert demo.normalize_label("") == ""
    assert demo.normalize_label("<error: X>") == "error_x"


def test_events_sort_oldest_first_whatever_the_api_order():
    from types import SimpleNamespace as E
    # Order and message format as returned on 2026-09-27.
    page = [E(id="ftevent-ftjob-x-2", created_at=48, message="[State: VALIDATING_FILES]"),
            E(id="ftevent-ftjob-x-1", created_at=48, message="[State: QUEUED_VALIDATING_FILES]"),
            E(id="ftevent-ftjob-x-0", created_at=46, message="[State: CREATING]")]
    assert [e.id[-1] for e in sorted(page, key=demo.event_order)] == ["0", "1", "2"]
    assert [e.id[-1] for e in sorted(reversed(page), key=demo.event_order)] == ["0", "1", "2"]


def test_event_text_drops_server_timestamp():
    msg = "[State: CREATING] state transitioned at 2026-09-27 04:41:46 AM PDT"
    assert demo.event_text(msg) == "[State: CREATING]"
    assert demo.event_text("step 16/30") == "step 16/30"


def test_save_state_is_atomic_and_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(demo, "STATE_FILE", tmp_path / "state.json")
    demo.save_state({"deployment": {"id": "d-1", "price_per_hour": 5.5}})
    assert demo.load_state() == {"deployment": {"id": "d-1", "price_per_hour": 5.5}}
    assert not list(tmp_path.glob("*.tmp"))  # temp file replaced, not left behind


def test_cleanup_only_forgets_what_is_really_gone():
    import httpx
    assert demo.gone(httpx.Response(202))
    assert demo.gone(httpx.Response(204))
    assert demo.gone(httpx.Response(404))  # already deleted
    assert not demo.gone(httpx.Response(500))  # still billing: must stay in state.json
    assert not demo.gone(httpx.Response(409))


def test_metrics_line_reads_latest_values():
    metrics = {"metrics": [
        {"metric_name": "ev_current_step", "data": [{"x": 1, "value": 8}, {"x": 2, "value": 16}]},
        {"metric_name": "ev_total_steps", "data": [{"x": 2, "value": 30}]},
        {"metric_name": "ev_loss", "data": [{"x": 2, "value": 0.03941}]},
        {"metric_name": "ev_eval_loss", "data": []},
    ]}
    assert demo.metrics_line(metrics) == "step 16/30  train_loss 0.0394"
    assert demo.metrics_line({"metrics": []}) == ""


def test_resolve_model_accepts_id_or_name():
    models = [{"id": "model-qwen-qwen3-5-9b-2ce94bd8", "model_name": "Qwen/Qwen3.5-9B"}]
    assert demo.resolve_model(models, "qwen/qwen3.5-9b") is models[0]
    assert demo.resolve_model(models, "model-qwen-qwen3-5-9b-2ce94bd8") is models[0]
    assert demo.resolve_model(models, "Qwen/Qwen3.5-2B") is None


def test_build_splits_refuses_test_rows_seen_in_training():
    train, test = fake_rows(), fake_rows(per_label=40, split="test")
    clean = demo.build_splits(train, test)
    planted = clean["train"][0]["messages"][1]["content"]
    label = clean["train"][0]["messages"][2]["content"]
    # Same message, different case and spacing, in every test row of that intent.
    test = [{**r, "text": f"  {planted.upper()} "} if r["label_text"] == label else r for r in test]
    try:
        demo.build_splits(train, test)
    except ValueError as e:
        assert "also appear in train/val" in str(e)
    else:
        raise AssertionError("a leaked test message must raise")
