import pickle
import sys

import pytest

from loaders import ModelLoaderRegistry, SklearnModelLoader, TorchModelLoader
import models


def _make_sklearn_bytes():
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.fit([[0, 0], [1, 1]], [0, 1])
    return pickle.dumps(model)


def _predict(client, production_model_id, features=None):
    if features is None:
        features = [[0, 0], [1, 1]]
    return client.post(
        "/models/predict",
        params={"model_id": production_model_id},
        json={"features": features},
    )


def _clear_cache(client, production_model_id):
    return client.post(
        "/models/cache/clear",
        params={"model_id": production_model_id},
    )


class TestLoaderSelection:
    def test_pkl_selects_sklearn_loader(self):
        loader = ModelLoaderRegistry.get_loader("model.pkl")
        assert isinstance(loader, SklearnModelLoader)

    def test_pickle_selects_sklearn_loader(self):
        loader = ModelLoaderRegistry.get_loader("model.pickle")
        assert isinstance(loader, SklearnModelLoader)

    def test_joblib_selects_sklearn_loader(self):
        loader = ModelLoaderRegistry.get_loader("model.joblib")
        assert isinstance(loader, SklearnModelLoader)

    def test_pt_selects_torch_loader(self):
        loader = ModelLoaderRegistry.get_loader("model.pt")
        assert isinstance(loader, TorchModelLoader)

    def test_pth_selects_torch_loader(self):
        loader = ModelLoaderRegistry.get_loader("model.pth")
        assert isinstance(loader, TorchModelLoader)

    def test_unknown_extension_falls_back_to_default(self):
        loader = ModelLoaderRegistry.get_loader("model.unknownext")
        assert isinstance(loader, SklearnModelLoader)

    def test_list_supported_extensions_returns_sorted_set(self):
        exts = ModelLoaderRegistry.list_supported_extensions()
        assert exts == sorted(exts)
        for expected in [".pkl", ".pickle", ".joblib", ".pt", ".pth"]:
            assert expected in exts


class TestSklearnModelLoader:
    def test_load_real_sklearn_model(self):
        loader = SklearnModelLoader()
        model = loader.load(_make_sklearn_bytes())
        assert hasattr(model, "predict")
        assert hasattr(model, "fit")

    def test_validate_sklearn_true_for_valid_model(self):
        loader = SklearnModelLoader()
        assert loader.validate_sklearn(_make_sklearn_bytes()) is True

    def test_validate_sklearn_false_for_non_sklearn_pickle(self):
        loader = SklearnModelLoader()
        content = pickle.dumps({"not": "a model"})
        assert loader.validate_sklearn(content) is False

    def test_validate_sklearn_false_for_invalid_bytes(self):
        loader = SklearnModelLoader()
        assert loader.validate_sklearn(b"not a pickle at all") is False

    def test_get_classes_returns_model_classes(self):
        loader = SklearnModelLoader()
        model = loader.load(_make_sklearn_bytes())
        classes = loader.get_classes(model)
        assert classes is not None
        assert [int(c) for c in classes] == [0, 1]

    def test_has_predict_proba_true_for_classifier(self):
        loader = SklearnModelLoader()
        model = loader.load(_make_sklearn_bytes())
        assert loader.has_predict_proba(model) is True

    def test_detect_prediction_type_returns_classification(self):
        loader = SklearnModelLoader()
        model = loader.load(_make_sklearn_bytes())
        assert loader.detect_prediction_type(model) == "classification"


class TestTorchModelLoader:
    def test_supported_extensions(self):
        loader = TorchModelLoader()
        assert loader.supported_extensions == [".pt", ".pth"]

    def test_load_raises_value_error_without_torch(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        loader = TorchModelLoader()
        with pytest.raises(ValueError):
            loader.load(b"fake torch bytes")


class TestModelCache:
    def test_first_prediction_loads_model_cache_miss(self, client, production_model_id):
        _clear_cache(client, production_model_id)
        before = client.get("/models/cache/stats").json()
        assert production_model_id not in before["cached_models"]

        resp = _predict(client, production_model_id)
        assert resp.status_code == 200

        after = client.get("/models/cache/stats").json()
        assert production_model_id in after["cached_models"]
        assert after["count"] >= 1

    def test_second_prediction_uses_cache_hit(self, client, production_model_id, monkeypatch):
        _clear_cache(client, production_model_id)

        load_calls = {"count": 0}
        real_load = models._load_model_instance

        def counting_load(file_path, checksum):
            load_calls["count"] += 1
            return real_load(file_path, checksum)

        monkeypatch.setattr(models, "_load_model_instance", counting_load)

        first = _predict(client, production_model_id)
        assert first.status_code == 200
        assert load_calls["count"] == 1

        second = _predict(client, production_model_id)
        assert second.status_code == 200
        assert load_calls["count"] == 1

    def test_cache_stats_endpoint(self, client, production_model_id):
        _clear_cache(client, production_model_id)
        _predict(client, production_model_id)

        resp = client.get("/models/cache/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "count" in data
        assert "cached_models" in data
        assert production_model_id in data["cached_models"]

    def test_cache_clear_endpoint(self, client, production_model_id):
        _clear_cache(client, production_model_id)
        _predict(client, production_model_id)
        assert production_model_id in client.get("/models/cache/stats").json()["cached_models"]

        resp = _clear_cache(client, production_model_id)
        assert resp.status_code == 200

        stats = client.get("/models/cache/stats").json()
        assert production_model_id not in stats["cached_models"]

    def test_cache_invalidated_after_registering_new_version(self, client, production_model_id):
        _clear_cache(client, production_model_id)
        _predict(client, production_model_id)
        assert production_model_id in client.get("/models/cache/stats").json()["cached_models"]

        model_info = client.get(f"/models/{production_model_id}").json()
        experiment_id = model_info["experiment_id"]
        content = _make_sklearn_bytes()

        resp = client.post(
            f"/models/register-sklearn/{experiment_id}",
            data={"name": "test-model"},
            files={"file": ("model.pkl", content, "application/octet-stream")},
        )
        assert resp.status_code == 200
        assert resp.json()["version"] == 2

        stats = client.get("/models/cache/stats").json()
        assert production_model_id not in stats["cached_models"]


class TestBatchPrediction:
    def _create_job(self, client, production_model_id, csv_content=None):
        if csv_content is None:
            csv_content = "f1,f2\n0,0\n1,1\n0.5,0.5"
        resp = client.post(
            "/batch-predict",
            data={"model_id": production_model_id},
            files={"file": ("data.csv", csv_content.encode(), "text/csv")},
        )
        assert resp.status_code == 200
        return resp.json()

    def test_create_batch_job(self, client, production_model_id):
        data = self._create_job(client, production_model_id)
        assert "id" in data
        assert data["model_id"] == production_model_id
        assert data["status"] == "pending"
        assert data["total_rows"] == 0

    def test_get_batch_job_status(self, client, production_model_id):
        created = self._create_job(client, production_model_id)
        job_id = created["id"]

        resp = client.get(f"/batch-predict/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == job_id
        assert data["status"] == "pending"
        assert data["model_id"] == production_model_id

    def test_list_batch_jobs(self, client, production_model_id):
        created = self._create_job(client, production_model_id)
        job_id = created["id"]

        resp = client.get("/batch-predict")
        assert resp.status_code == 200
        jobs = resp.json()
        assert isinstance(jobs, list)
        assert job_id in [j["id"] for j in jobs]

    def test_get_batch_job_not_found(self, client):
        resp = client.get("/batch-predict/999999")
        assert resp.status_code == 404
