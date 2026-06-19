import pickle

import pytest


def _make_sklearn_bytes():
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.fit([[0, 0], [1, 1]], [0, 1])
    return pickle.dumps(model)


class TestRootAndHealth:
    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "name" in data
        assert "version" in data
        assert "architecture" in data

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


class TestExperimentAPI:
    def test_create_experiment(self, client):
        resp = client.post(
            "/experiments",
            json={
                "name": "api-exp",
                "project": "api-proj",
                "parameters": {"lr": "0.1"},
                "tags": ["tag1", "tag2"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "api-exp"
        assert data["status"] == "running"
        assert len(data["parameters"]) == 1
        assert sorted(data["tags"]) == ["tag1", "tag2"]

    def test_create_experiment_validation_error(self, client):
        resp = client.post("/experiments", json={"name": "", "project": "p"})
        assert resp.status_code == 422

    def test_list_experiments_filter(self, client):
        client.post("/experiments", json={"name": "e1", "project": "projA"})
        client.post("/experiments", json={"name": "e2", "project": "projB"})
        resp = client.get("/experiments", params={"project": "projA"})
        assert resp.status_code == 200
        assert len(resp.json()) == 1
        assert resp.json()[0]["project"] == "projA"

    def test_get_experiment_not_found(self, client):
        resp = client.get("/experiments/999999")
        assert resp.status_code == 404

    def test_get_experiment_ok(self, client, sample_experiment_id):
        resp = client.get(f"/experiments/{sample_experiment_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == sample_experiment_id

    def test_update_status_running_to_success(self, client, sample_experiment_id):
        resp = client.patch(
            f"/experiments/{sample_experiment_id}", json={"status": "success"}
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"
        assert resp.json()["end_time"] is not None

    def test_update_status_failed_cannot_revert_to_running(self, client, sample_experiment_id):
        client.patch(f"/experiments/{sample_experiment_id}", json={"status": "failed"})
        resp = client.patch(
            f"/experiments/{sample_experiment_id}", json={"status": "running"}
        )
        assert resp.status_code == 400

    def test_update_status_invalid(self, client, sample_experiment_id):
        resp = client.patch(
            f"/experiments/{sample_experiment_id}", json={"status": "unknown"}
        )
        assert resp.status_code == 400

    def test_delete_experiment(self, client, sample_experiment_id):
        resp = client.delete(f"/experiments/{sample_experiment_id}")
        assert resp.status_code == 200
        resp2 = client.get(f"/experiments/{sample_experiment_id}")
        assert resp2.status_code == 404

    def test_experiment_tags_crud(self, client, sample_experiment_id):
        resp = client.get(f"/experiments/{sample_experiment_id}/tags")
        assert resp.status_code == 200
        assert resp.json() == []

        resp = client.post(
            f"/experiments/{sample_experiment_id}/tags", json={"tags": ["a", "b"]}
        )
        assert resp.status_code == 200
        assert sorted(resp.json()) == ["a", "b"]

        resp = client.request(
            "DELETE",
            f"/experiments/{sample_experiment_id}/tags",
            json={"tags": ["a"]},
        )
        assert resp.status_code == 200
        assert resp.json() == ["b"]

    def test_list_all_tags(self, client):
        client.post(
            "/experiments",
            json={"name": "t1", "project": "p1", "tags": ["x", "y"]},
        )
        client.post(
            "/experiments",
            json={"name": "t2", "project": "p1", "tags": ["y", "z"]},
        )
        resp = client.get("/experiments/tags/all")
        assert resp.status_code == 200
        assert sorted(resp.json()) == ["x", "y", "z"]


class TestMetricAPI:
    def test_log_metric(self, client, sample_experiment_id):
        resp = client.post(
            f"/metrics/{sample_experiment_id}",
            json={"name": "loss", "value": 0.5},
        )
        assert resp.status_code == 200
        assert resp.json()["step"] == 0

    def test_log_metric_experiment_not_found(self, client):
        resp = client.post("/metrics/999999", json={"name": "loss", "value": 0.5})
        assert resp.status_code == 404

    def test_log_metrics_batch_step_auto_increment(self, client, sample_experiment_id):
        resp = client.post(
            f"/metrics/{sample_experiment_id}/batch",
            json=[
                {"name": "loss", "value": 1.0},
                {"name": "loss", "value": 0.8},
                {"name": "loss", "value": 0.6},
            ],
        )
        assert resp.status_code == 200
        steps = sorted([m["step"] for m in resp.json()])
        assert steps == [0, 1, 2]

    def test_get_metrics_ordered_by_step(self, client, sample_experiment_id):
        for v, s in [(0.5, 2), (0.1, 0), (0.3, 1)]:
            client.post(
                f"/metrics/{sample_experiment_id}",
                json={"name": "loss", "value": v, "step": s},
            )
        resp = client.get(
            f"/metrics/{sample_experiment_id}", params={"metric_name": "loss"}
        )
        assert resp.status_code == 200
        values = [m["value"] for m in resp.json()]
        assert values == [0.1, 0.3, 0.5]

    def test_get_metric_names(self, client, sample_experiment_id):
        client.post(
            f"/metrics/{sample_experiment_id}", json={"name": "acc", "value": 0.9}
        )
        client.post(
            f"/metrics/{sample_experiment_id}", json={"name": "loss", "value": 0.1}
        )
        resp = client.get(f"/metrics/{sample_experiment_id}/names")
        assert resp.status_code == 200
        assert sorted(resp.json()) == ["acc", "loss"]

    def test_metric_ranking_desc(self, client):
        for i, val in enumerate([0.9, 0.5, 0.7]):
            exp_resp = client.post(
                "/experiments", json={"name": f"rank{i}", "project": "p"}
            )
            eid = exp_resp.json()["id"]
            client.post(f"/metrics/{eid}", json={"name": "acc", "value": val})

        resp = client.get("/metrics/ranking/acc", params={"order": "desc"})
        assert resp.status_code == 200
        values = [r["latest_value"] for r in resp.json()]
        assert values == sorted(values, reverse=True)


class TestModelAPI:
    def test_register_sklearn_model(self, client, sample_experiment_id):
        content = _make_sklearn_bytes()
        resp = client.post(
            f"/models/register-sklearn/{sample_experiment_id}",
            data={"name": "mymodel"},
            files={"file": ("model.pkl", content, "application/octet-stream")},
        )
        assert resp.status_code == 200
        assert resp.json()["version"] == 1
        assert resp.json()["name"] == "mymodel"

    def test_register_invalid_model_file(self, client, sample_experiment_id):
        resp = client.post(
            f"/models/register-sklearn/{sample_experiment_id}",
            data={"name": "bad"},
            files={"file": ("model.pkl", b"not a real model", "application/octet-stream")},
        )
        assert resp.status_code == 400

    def test_register_model_experiment_not_found(self, client):
        content = _make_sklearn_bytes()
        resp = client.post(
            "/models/register-sklearn/999999",
            data={"name": "m"},
            files={"file": ("model.pkl", content, "application/octet-stream")},
        )
        assert resp.status_code == 404

    def test_list_models(self, client, production_model_id):
        resp = client.get("/models")
        assert resp.status_code == 200
        assert len(resp.json()) >= 1
        ids = [m["id"] for m in resp.json()]
        assert production_model_id in ids

    def test_get_model_not_found(self, client):
        resp = client.get("/models/999999")
        assert resp.status_code == 404

    def test_set_and_get_production(self, client, sample_experiment_id):
        content = _make_sklearn_bytes()
        resp = client.post(
            f"/models/register-sklearn/{sample_experiment_id}",
            data={"name": "prodmodel"},
            files={"file": ("model.pkl", content, "application/octet-stream")},
        )
        model_id = resp.json()["id"]

        resp = client.post(f"/models/{model_id}/production")
        assert resp.status_code == 200
        assert resp.json()["model_id"] == model_id

        resp = client.get(f"/models/{model_id}")
        assert resp.json()["is_production"] is True

        resp = client.get("/models/production/prodmodel")
        assert resp.status_code == 200
        assert resp.json()["id"] == model_id
        assert resp.json()["is_production"] is True

    def test_predict_with_production_model(self, client, production_model_id):
        resp = client.post(
            "/models/predict",
            params={"model_id": production_model_id},
            json={"features": [[0, 0], [1, 1], [0.5, 0.5]]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 3
        assert data["model_id"] == production_model_id

    def test_predict_model_not_found(self, client):
        resp = client.post(
            "/models/predict",
            params={"model_id": 999999},
            json={"features": [[0, 0]]},
        )
        assert resp.status_code == 404


class TestComparatorAPI:
    def test_compare_experiments(self, client):
        ids = []
        for i in range(3):
            resp = client.post(
                "/experiments", json={"name": f"c{i}", "project": "p"}
            )
            eid = resp.json()["id"]
            client.post(f"/metrics/{eid}", json={"name": "acc", "value": 0.5 + i * 0.1})
            ids.append(eid)

        resp = client.post("/comparator/compare", params={"experiment_ids": ids})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["experiments"]) == 3
        acc_summary = [s for s in data["metric_summaries"] if s["metric_name"] == "acc"][0]
        assert len(acc_summary["values"]) == 3

    def test_compare_missing_experiment(self, client, sample_experiment_id):
        resp = client.post(
            "/comparator/compare",
            params={"experiment_ids": [sample_experiment_id, 999999]},
        )
        assert resp.status_code == 404

    def test_find_best_experiment(self, client):
        for i, val in enumerate([0.9, 0.5, 0.7]):
            resp = client.post(
                "/experiments", json={"name": f"best{i}", "project": "p"}
            )
            eid = resp.json()["id"]
            client.post(f"/metrics/{eid}", json={"name": "loss", "value": val})
        resp = client.get(
            "/comparator/best", params={"metric_name": "loss", "order": "asc"}
        )
        assert resp.status_code == 200
        assert resp.json()["metric_value"] == 0.5


class TestEvaluationAPI:
    def test_create_evaluation_report(self, client, production_model_id):
        import io, csv

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["f1", "f2", "label"])
        writer.writerow([0, 0, 0])
        writer.writerow([1, 1, 1])
        writer.writerow([0.5, 0.5, 0])
        csv_content = buf.getvalue()

        resp = client.post(
            f"/evaluation/{production_model_id}",
            data={
                "label_column": "label",
                "prediction_type": "classification",
            },
            files={"file": ("eval.csv", csv_content.encode(), "text/csv")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_id"] == production_model_id
        assert "metrics" in data

    def test_list_evaluation_reports(self, client, production_model_id):
        resp = client.get("/evaluation")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
