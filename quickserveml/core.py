from model import ModelRegistry, ModelMetadata
from dataset import DatasetRegistry, DatasetMetadata
from typing import Optional, List, Dict, Any
import numpy as np
import time
import platform
import psutil
from benchmark import BenchmarkRegistry, BenchmarkMetadata

def register_model(metadata: ModelMetadata, db_path: Optional[str] = None):
    registry = ModelRegistry(db_path=db_path)
    registry.add_model(metadata)
    return True

def get_model(name: str, version: Optional[str] = None, db_path: Optional[str] = None) -> Optional[ModelMetadata]:
    registry = ModelRegistry(db_path=db_path)
    return registry.get_model(name, version)

def list_models(db_path: Optional[str] = None) -> List[ModelMetadata]:
    registry = ModelRegistry(db_path=db_path)
    return registry.list_models()

def register_dataset(metadata: DatasetMetadata, db_path: Optional[str] = None):
    registry = DatasetRegistry(db_path=db_path)
    registry.add_dataset(metadata)
    return True

def get_dataset(name: str, version: Optional[str] = None, db_path: Optional[str] = None) -> Optional[DatasetMetadata]:
    registry = DatasetRegistry(db_path=db_path)
    return registry.get_dataset(name, version)

def list_datasets(db_path: Optional[str] = None) -> List[DatasetMetadata]:
    registry = DatasetRegistry(db_path=db_path)
    return registry.list_datasets()

def run_benchmark(model: ModelMetadata, dataset: Optional[DatasetMetadata] = None, labels: Optional[np.ndarray] = None, run_params: Optional[Dict[str, Any]] = None, db_path: Optional[str] = None) -> BenchmarkMetadata:
    """
    Run a benchmark for a model on a dataset (or auto-generated data).
    Computes performance, quality, explainability, and resource metrics.
    """
    import onnxruntime as ort
    run_params = run_params or {}
    batch_size = run_params.get("batch_size", 16)
    num_runs = run_params.get("num_runs", 100)
    # Load model
    session = ort.InferenceSession(model.artifact_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_dtype = np.float32  # Default
    # Prepare data
    if dataset is not None and dataset.profiling.get("sample"):  # Use real data if available
        X = np.array(dataset.profiling["sample"])
    else:
        # Auto-generate random data
        shape = [batch_size] + [d if isinstance(d, int) and d > 0 else 1 for d in input_shape[1:]]
        X = np.random.rand(*shape).astype(input_dtype)
    # Run inference and collect metrics
    latencies = []
    mem_usages = []
    cpu_usages = []
    start_mem = psutil.Process().memory_info().rss
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: X})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms
        mem_usages.append(psutil.Process().memory_info().rss)
        cpu_usages.append(psutil.cpu_percent(interval=None))
    end_mem = psutil.Process().memory_info().rss
    # Performance metrics
    metrics = {
        "latency_mean": float(np.mean(latencies)),
        "latency_min": float(np.min(latencies)),
        "latency_max": float(np.max(latencies)),
        "latency_median": float(np.median(latencies)),
        "latency_p95": float(np.percentile(latencies, 95)),
        "latency_p99": float(np.percentile(latencies, 99)),
        "throughput": float(batch_size * num_runs / (np.sum(latencies) / 1000)),
        "model_load_time": 0.0,  # Could be measured separately
        "file_size": model.file_size,
        "memory_usage_peak": int(np.max(mem_usages)),
        "memory_usage_avg": int(np.mean(mem_usages)),
        "cpu_usage_avg": float(np.mean(cpu_usages)),
    }
    # Quality metrics (if labels provided)
    if labels is not None:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
        y_pred = session.run(None, {input_name: X})[0]
        y_pred_flat = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
        y_true_flat = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
        metrics["accuracy"] = float(accuracy_score(y_true_flat, y_pred_flat))
        metrics["f1"] = float(f1_score(y_true_flat, y_pred_flat, average="weighted"))
        metrics["precision"] = float(precision_score(y_true_flat, y_pred_flat, average="weighted"))
        metrics["recall"] = float(recall_score(y_true_flat, y_pred_flat, average="weighted"))
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true_flat, y_pred))
        except Exception:
            pass
        metrics["confusion_matrix"] = confusion_matrix(y_true_flat, y_pred_flat).tolist()
        # Regression metrics
        try:
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)))
            metrics["mae"] = float(mean_absolute_error(y_true_flat, y_pred_flat))
            metrics["r2"] = float(r2_score(y_true_flat, y_pred_flat))
        except Exception:
            pass
    # Explainability (placeholder)
    explainability = {}
    # Resource/operational
    resource_usage = {
        "memory_start": int(start_mem),
        "memory_end": int(end_mem),
        "hardware": platform.platform(),
    }
    # Save benchmark
    bench = BenchmarkMetadata(
        model_version=model.version,
        dataset_version=dataset.version if dataset else "auto",
        run_params=run_params,
        metrics=metrics,
        resource_usage=resource_usage,
        explainability=explainability,
        hardware_info={"platform": platform.platform()},
        logs="",
        user_notes="",
        tags=[]
    )
    BenchmarkRegistry(db_path=db_path).add_benchmark(bench)
    return bench

def get_benchmarks(model_version: str = None, dataset_version: str = None, db_path: Optional[str] = None) -> List[BenchmarkMetadata]:
    return BenchmarkRegistry(db_path=db_path).list_benchmarks(model_version, dataset_version)

def compare_models(model1: ModelMetadata, model2: ModelMetadata, metrics: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compare two models on all available metrics. Returns a dict with side-by-side values and differences.
    """
    m1 = model1.metrics
    m2 = model2.metrics
    all_metrics = set(m1.keys()) | set(m2.keys())
    if metrics:
        all_metrics = set(metrics) & all_metrics
    comparison = {}
    for key in sorted(all_metrics):
        v1 = m1.get(key)
        v2 = m2.get(key)
        diff = None
        if v1 is not None and v2 is not None:
            try:
                diff = v2 - v1 if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) else None
            except Exception:
                diff = None
        comparison[key] = {"model1": v1, "model2": v2, "difference": diff}
    return comparison 