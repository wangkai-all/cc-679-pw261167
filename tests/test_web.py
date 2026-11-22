"""End-to-end web flows: upload, search, decide, validate."""
from __future__ import annotations

import io
import json
import re
import sqlite3
import time
from urllib.parse import quote

import pytest

from refgap.web import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(str(tmp_path / "db.sqlite3"), str(tmp_path / "uploads"))
    app.config.update(TESTING=True)
    return app.test_client()


@pytest.fixture
def dataset(client, tmp_path):
    client.post("/datasets", data={"name": "demo"})
    db = sqlite3.connect(str(tmp_path / "db.sqlite3"))
    ds = db.execute("select id from datasets").fetchone()[0]
    db.close()
    return ds


def upload(client, ds, files):
    return client.post(
        f"/datasets/{ds}/upload",
        data={"files": [(io.BytesIO(content), name) for content, name in files]},
        content_type="multipart/form-data",
    )


def wait_for(client, ds, run_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"/datasets/{ds}/runs/{run_id}/status").get_json()
        if not payload["running"]:
            return payload
        time.sleep(0.05)
    raise AssertionError("search did not finish")


def start(client, ds, **overrides):
    data = {
        "width": "2",
        "null_policy": "skip_child_nulls",
        "mapping": "auto",
        "max_verified": "100000",
        "max_seconds": "60",
    }
    data.update(overrides)
    response = client.post(f"/datasets/{ds}/search", data=data, follow_redirects=True)
    match = re.search(r"/datasets/" + ds + r"/runs/([0-9a-f]+)", response.get_data(as_text=True))
    assert match, "run page not found"
    return match.group(1)


# S9 keeps the child's lexicographic range inside the parent's, so the
# missing S5 is a real counterexample rather than an out-of-range prune.
PARENT = b"store_id,city\nS1,Shanghai\nS2,Hangzhou\nS3,Chengdu\nS9,Wuhan\n"
CHILD = b"order_id,store_id\nO1,S1\nO2,S2\nO3,S5\n"


def test_index_and_dataset_creation(client, dataset):
    assert client.get("/").status_code == 200
    page = client.get(f"/datasets/{dataset}").get_data(as_text=True)
    assert "上传文件" in page


def test_upload_rejects_unsupported_type(client, dataset):
    response = upload(client, dataset, [(b"a,b\n1,2\n", "notes.txt")])
    assert response.status_code == 400
    assert "不支持的文件类型" in response.get_data(as_text=True)


def test_upload_reports_unreadable_file(client, dataset):
    response = upload(client, dataset, [(b"", "empty.csv")])
    assert response.status_code == 400
    assert "file is empty" in response.get_data(as_text=True)


def test_search_requires_two_tables(client, dataset):
    upload(client, dataset, [(PARENT, "stores.csv")])
    page = client.get(f"/datasets/{dataset}").get_data(as_text=True)
    assert "至少需要两张表" in page


def test_full_flow_confirm_validate(client, dataset, tmp_path):
    upload(client, dataset, [(PARENT, "stores.csv"), (CHILD, "orders.csv")])
    run_id = start(client, dataset)
    status = wait_for(client, dataset, run_id)
    assert status["status"] == "completed"
    assert status["counts"].get("partial", 0) >= 1

    key = "orders[store_id] -> stores[store_id] @auto"
    candidate = client.get(
        f"/datasets/{dataset}/runs/{run_id}/candidates/{quote(key, safe='')}"
    )
    assert candidate.status_code == 200
    body = candidate.get_data(as_text=True)
    assert "S5" in body  # the counterexample is visible
    assert "不能匹配的反例" in body

    response = client.post(
        f"/datasets/{dataset}/runs/{run_id}/candidates/{quote(key, safe='')}/decision",
        data={"action": "confirmed", "allow_partial": "on", "note": "门店主数据"},
    )
    assert response.status_code == 302

    manifest = client.get(f"/datasets/{dataset}/manifest.json")
    payload = json.loads(manifest.get_data(as_text=True))
    assert payload["relations"][0]["candidate_key"] == key

    # Next batch with an orphan: publication must be refused.
    response = client.post(
        f"/datasets/{dataset}/validate",
        data={
            "label": "batch-2",
            "files": [
                (io.BytesIO(b"order_id,store_id\nX1,S1\nX2,S777\n"), "orders.csv"),
                (io.BytesIO(PARENT), "stores.csv"),
            ],
        },
        content_type="multipart/form-data",
    )
    page = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "拒绝发布" in page
    assert "S777" in page

    # A clean batch is allowed.
    response = client.post(
        f"/datasets/{dataset}/validate",
        data={
            "label": "batch-3",
            "files": [
                (io.BytesIO(b"order_id,store_id\nX1,S1\n"), "orders.csv"),
                (io.BytesIO(PARENT), "stores.csv"),
            ],
        },
        content_type="multipart/form-data",
    )
    page = response.get_data(as_text=True)
    assert "允许发布" in page
    assert "已解决缺口键" in page or "新增缺口键" in page


def test_confirming_partial_requires_acknowledgement(client, dataset):
    upload(client, dataset, [(PARENT, "stores.csv"), (CHILD, "orders.csv")])
    run_id = start(client, dataset)
    wait_for(client, dataset, run_id)
    key = "orders[store_id] -> stores[store_id] @auto"
    response = client.post(
        f"/datasets/{dataset}/runs/{run_id}/candidates/{quote(key, safe='')}/decision",
        data={"action": "confirmed"},
    )
    assert response.status_code == 400
    assert "存在不能匹配的记录" in response.get_data(as_text=True)

    response = client.post(
        f"/datasets/{dataset}/runs/{run_id}/candidates/{quote(key, safe='')}/decision",
        data={"action": "confirmed", "allow_partial": "on"},
    )
    assert response.status_code == 302


def test_excluded_candidate_never_becomes_a_constraint(client, dataset):
    upload(client, dataset, [(PARENT, "stores.csv"), (CHILD, "orders.csv")])
    run_id = start(client, dataset)
    wait_for(client, dataset, run_id)
    key = "orders[store_id] -> stores[store_id] @auto"
    client.post(
        f"/datasets/{dataset}/runs/{run_id}/candidates/{quote(key, safe='')}/decision",
        data={"action": "excluded"},
    )
    page = client.get(f"/datasets/{dataset}").get_data(as_text=True)
    assert "已确认关系：<strong>0</strong>" in page
    response = client.post(
        f"/datasets/{dataset}/validate",
        data={"files": [(io.BytesIO(CHILD), "orders.csv")]},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "没有已确认的引用关系" in response.get_data(as_text=True)


def test_validate_without_uploaded_child_marks_not_applicable(client, dataset):
    upload(client, dataset, [(PARENT, "stores.csv"), (CHILD, "orders.csv")])
    run_id = start(client, dataset)
    wait_for(client, dataset, run_id)
    key = "orders[store_id] -> stores[store_id] @auto"
    client.post(
        f"/datasets/{dataset}/runs/{run_id}/candidates/{quote(key, safe='')}/decision",
        data={"action": "confirmed", "allow_partial": "on"},
    )
    response = client.post(
        f"/datasets/{dataset}/validate",
        data={"files": [(io.BytesIO(PARENT), "stores.csv")]},
        content_type="multipart/form-data",
    )
    page = response.get_data(as_text=True)
    assert "本批不适用" in page
    assert "不会静默通过" in page


def test_column_limit_is_validated(client, dataset):
    upload(client, dataset, [(PARENT, "stores.csv"), (CHILD, "orders.csv")])
    response = client.post(
        f"/datasets/{dataset}/search",
        data={
            "width": "2",
            "null_policy": "skip_child_nulls",
            "mapping": "auto",
            "max_verified": "10",
            "max_seconds": "5",
            "cols_orders": "store_id,nope",
        },
    )
    assert response.status_code == 400
    assert "不存在列" in response.get_data(as_text=True)


def test_counterexample_download_is_csv(client, dataset):
    upload(client, dataset, [(PARENT, "stores.csv"), (CHILD, "orders.csv")])
    run_id = start(client, dataset)
    wait_for(client, dataset, run_id)
    key = "orders[store_id] -> stores[store_id] @auto"
    response = client.get(
        f"/datasets/{dataset}/runs/{run_id}/candidates/{quote(key, safe='')}/counterexamples.csv"
    )
    assert response.status_code == 200
    assert "attachment" in response.headers["Content-Disposition"]
    text = response.get_data(as_text=True)
    assert "S5" in text
    assert "occurrences" in text
