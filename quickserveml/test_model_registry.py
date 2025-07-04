import os
import tempfile
from model import ModelRegistry, ModelMetadata
from datetime import datetime

def test_model_registry():
    # Use a temporary database file
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "models.db")
        registry = ModelRegistry(db_path=db_path)

        # Create a model metadata object
        model = ModelMetadata(
            name="test-model",
            version="v1.0.0",
            description="A test model",
            tags=["test", "demo"],
            author="UnitTester",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            status="prod",
            artifact_path="/tmp/model.onnx",
            hash="abc123",
            file_size=12345,
            framework="onnx",
            opset_version="13",
            dependencies={"onnxruntime": "1.15.0"},
            training_params={"epochs": 10, "lr": 0.001},
            dataset_version="v1.0.0",
            experiment_id="exp-001",
            metrics={"accuracy": 0.98, "f1": 0.97, "latency": 12.3},
            explainability={"feature_importances": [0.5, 0.3, 0.2]},
            approval_history=[{"user": "admin", "action": "approved"}],
            changelog=["Initial version"]
        )

        # Add model
        registry.add_model(model)

        # Retrieve model
        retrieved = registry.get_model("test-model", "v1.0.0")
        assert retrieved is not None
        assert retrieved.name == model.name
        assert retrieved.metrics["accuracy"] == 0.98
        assert retrieved.explainability["feature_importances"] == [0.5, 0.3, 0.2]
        assert retrieved.status == "prod"

        # List models
        models = registry.list_models()
        assert len(models) == 1
        assert models[0].name == "test-model"

if __name__ == "__main__":
    test_model_registry()
    print("ModelRegistry unit test passed!") 