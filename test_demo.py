"""Pure-function checks. The API stages are verified against the real service, not mocked."""
import json
import random

import demo


def fake_rows(n_labels=77, per_label=130):
    rows = []
    for i in range(n_labels):
        for j in range(per_label):
            rows.append({"text": f"query {i}-{j}", "label": i, "label_text": f"intent_{i:02d}"})
    return rows


def test_build_splits_sizes_and_schema():
    splits = demo.build_splits(fake_rows(), fake_rows(per_label=40))
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
    a = demo.build_splits(fake_rows(), fake_rows(per_label=40))
    b = demo.build_splits(fake_rows(), fake_rows(per_label=40))
    assert a == b
    train_q = {ex["messages"][1]["content"] for ex in a["train"]}
    val_q = {ex["messages"][1]["content"] for ex in a["val"]}
    assert not train_q & val_q


def test_build_splits_fails_loud_on_short_label():
    rows = fake_rows(per_label=10)
    try:
        demo.build_splits(rows, fake_rows(per_label=40))
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
