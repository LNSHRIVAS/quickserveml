# quickserveml/benchmark.py

import time
import psutil
import numpy as np
from typing import Dict, List, Tuple, Optional
import statistics
from dataclasses import dataclass, field
from onnxruntime import InferenceSession
import threading
import queue
import sqlite3
import json
import os
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class BenchmarkResult:
    """Results from model benchmarking"""
    model_name: str
    input_shape: List[int]
    output_shape: List[int]
    avg_inference_time_ms: float
    min_inference_time_ms: float
    max_inference_time_ms: float
    p95_inference_time_ms: float
    p99_inference_time_ms: float
    throughput_rps: float
    memory_usage_mb: float
    cpu_usage_percent: float
    warmup_runs: int
    benchmark_runs: int
    total_time_seconds: float


@dataclass
class BenchmarkMetadata:
    model_version: str
    dataset_version: str
    run_params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)  # All computed metrics
    resource_usage: Dict[str, Any] = field(default_factory=dict)  # memory, CPU, GPU, etc.
    explainability: Dict[str, Any] = field(default_factory=dict)
    run_time: str = field(default_factory=lambda: datetime.now().isoformat())
    hardware_info: Dict[str, Any] = field(default_factory=dict)
    logs: str = ""
    user_notes: str = ""
    tags: List[str] = field(default_factory=list)


class BenchmarkRegistry:
    def __init__(self, db_path=None):
        self.db_path = db_path or os.path.expanduser("~/.quickserveml/registry/benchmarks.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_database()

    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS benchmarks (
                    model_version TEXT,
                    dataset_version TEXT,
                    run_params TEXT,
                    metrics TEXT,
                    resource_usage TEXT,
                    explainability TEXT,
                    run_time TEXT,
                    hardware_info TEXT,
                    logs TEXT,
                    user_notes TEXT,
                    tags TEXT,
                    PRIMARY KEY (model_version, dataset_version, run_time)
                )
            ''')
            conn.commit()

    def add_benchmark(self, metadata: BenchmarkMetadata):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO benchmarks (model_version, dataset_version, run_params, metrics, resource_usage, explainability, run_time, hardware_info, logs, user_notes, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.model_version, metadata.dataset_version, json.dumps(metadata.run_params), json.dumps(metadata.metrics),
                json.dumps(metadata.resource_usage), json.dumps(metadata.explainability), metadata.run_time, json.dumps(metadata.hardware_info),
                metadata.logs, metadata.user_notes, json.dumps(metadata.tags)
            ))
            conn.commit()

    def list_benchmarks(self, model_version: str = None, dataset_version: str = None) -> List[BenchmarkMetadata]:
        with sqlite3.connect(self.db_path) as conn:
            query = 'SELECT * FROM benchmarks'
            params = []
            conditions = []
            if model_version:
                conditions.append('model_version = ?')
                params.append(model_version)
            if dataset_version:
                conditions.append('dataset_version = ?')
                params.append(dataset_version)
            if conditions:
                query += ' WHERE ' + ' AND '.join(conditions)
            cursor = conn.execute(query, params)
            return [
                BenchmarkMetadata(
                    model_version=row[0], dataset_version=row[1], run_params=json.loads(row[2]), metrics=json.loads(row[3]),
                    resource_usage=json.loads(row[4]), explainability=json.loads(row[5]), run_time=row[6], hardware_info=json.loads(row[7]),
                    logs=row[8], user_notes=row[9], tags=json.loads(row[10])
                ) for row in cursor.fetchall()
            ]


class ModelBenchmarker:
    """Benchmark ONNX models for performance metrics"""
    
    def __init__(self, model_path: str, provider: str = "CPUExecutionProvider"):
        self.model_path = model_path
        self.provider = provider
        self.session = None
        self.input_name = None
        self.input_shape = None
        self.output_shape = None
        
    def load_model(self):
        """Load the ONNX model and extract metadata"""
        self.session = InferenceSession(self.model_path, providers=[self.provider])
        
        # Get input details
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.input_shape = [d if isinstance(d, int) and d > 0 else 1 for d in input_meta.shape]
        
        # Get output details
        output_meta = self.session.get_outputs()[0]
        self.output_shape = [d if isinstance(d, int) and d > 0 else 1 for d in output_meta.shape]
        
        print(f"✔ Loaded model: {self.model_path}")
        print(f"  Input: {self.input_name} {self.input_shape}")
        print(f"  Output: {output_meta.name} {self.output_shape}")
    
    def generate_dummy_input(self) -> np.ndarray:
        """Generate dummy input data for benchmarking"""
        # Use float32 for most models, but could be made configurable
        return np.random.randn(*self.input_shape).astype(np.float32)
    
    def warmup(self, num_runs: int = 10):
        """Warm up the model to ensure consistent performance"""
        print(f"🔥 Warming up model with {num_runs} runs...")
        
        for i in range(num_runs):
            dummy_input = self.generate_dummy_input()
            _ = self.session.run(None, {self.input_name: dummy_input})
            
            if (i + 1) % 5 == 0:
                print(f"  Warmup run {i + 1}/{num_runs}")
    
    def measure_inference_time(self, num_runs: int = 100) -> List[float]:
        """Measure inference times for multiple runs"""
        inference_times = []
        
        print(f"⏱️  Running {num_runs} inference benchmarks...")
        
        for i in range(num_runs):
            dummy_input = self.generate_dummy_input()
            
            start_time = time.perf_counter()
            _ = self.session.run(None, {self.input_name: dummy_input})
            end_time = time.perf_counter()
            
            inference_time_ms = (end_time - start_time) * 1000
            inference_times.append(inference_time_ms)
            
            if (i + 1) % 20 == 0:
                print(f"  Benchmark run {i + 1}/{num_runs}")
        
        return inference_times
    
    def measure_memory_usage(self) -> float:
        """Measure current memory usage in MB"""
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
    
    def measure_cpu_usage(self, duration: float = 5.0) -> float:
        """Measure CPU usage over a duration"""
        process = psutil.Process()
        
        # Run some inference during CPU measurement
        def inference_worker():
            for _ in range(50):  # Run 50 inferences
                dummy_input = self.generate_dummy_input()
                _ = self.session.run(None, {self.input_name: dummy_input})
        
        # Start inference in background
        inference_thread = threading.Thread(target=inference_worker)
        inference_thread.start()
        
        # Measure CPU usage
        cpu_percentages = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            cpu_percentages.append(process.cpu_percent())
            time.sleep(0.1)
        
        inference_thread.join()
        
        return statistics.mean(cpu_percentages) if cpu_percentages else 0.0
    
    def calculate_throughput(self, avg_inference_time_ms: float) -> float:
        """Calculate throughput in requests per second"""
        if avg_inference_time_ms == 0:
            return 0.0
        return 1000.0 / avg_inference_time_ms
    
    def run_benchmark(self, 
                     warmup_runs: int = 10, 
                     benchmark_runs: int = 100,
                     cpu_measurement_duration: float = 5.0) -> BenchmarkResult:
        """Run comprehensive benchmark and return results"""
        
        if self.session is None:
            self.load_model()
        
        print(f"\n🚀 Starting benchmark for {self.model_path}")
        print(f"   Warmup runs: {warmup_runs}")
        print(f"   Benchmark runs: {benchmark_runs}")
        
        # Measure initial memory
        initial_memory = self.measure_memory_usage()
        
        # Warmup
        self.warmup(warmup_runs)
        
        # Measure memory after warmup
        memory_after_warmup = self.measure_memory_usage()
        
        # Run inference benchmarks
        start_time = time.time()
        inference_times = self.measure_inference_time(benchmark_runs)
        total_time = time.time() - start_time
        
        # Calculate statistics
        avg_time = statistics.mean(inference_times)
        min_time = min(inference_times)
        max_time = max(inference_times)
        p95_time = np.percentile(inference_times, 95)
        p99_time = np.percentile(inference_times, 99)
        
        # Calculate throughput
        throughput = self.calculate_throughput(avg_time)
        
        # Measure CPU usage
        print("📊 Measuring CPU usage...")
        cpu_usage = self.measure_cpu_usage(cpu_measurement_duration)
        
        # Final memory measurement
        final_memory = self.measure_memory_usage()
        
        return BenchmarkResult(
            model_name=self.model_path,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            avg_inference_time_ms=avg_time,
            min_inference_time_ms=min_time,
            max_inference_time_ms=max_time,
            p95_inference_time_ms=p95_time,
            p99_inference_time_ms=p99_time,
            throughput_rps=throughput,
            memory_usage_mb=final_memory,
            cpu_usage_percent=cpu_usage,
            warmup_runs=warmup_runs,
            benchmark_runs=benchmark_runs,
            total_time_seconds=total_time
        )


def print_benchmark_results(result: BenchmarkResult):
    """Pretty print benchmark results"""
    print("\n" + "="*60)
    print("📊 BENCHMARK RESULTS")
    print("="*60)
    
    print(f"Model: {result.model_name}")
    print(f"Input Shape: {result.input_shape}")
    print(f"Output Shape: {result.output_shape}")
    
    print(f"\n⏱️  INFERENCE TIMING")
    print(f"  Average: {result.avg_inference_time_ms:.2f} ms")
    print(f"  Min:     {result.min_inference_time_ms:.2f} ms")
    print(f"  Max:     {result.max_inference_time_ms:.2f} ms")
    print(f"  95th %:  {result.p95_inference_time_ms:.2f} ms")
    print(f"  99th %:  {result.p99_inference_time_ms:.2f} ms")
    
    print(f"\n🚀 PERFORMANCE")
    print(f"  Throughput: {result.throughput_rps:.1f} requests/second")
    print(f"  Total benchmark time: {result.total_time_seconds:.1f} seconds")
    
    print(f"\n💾 RESOURCE USAGE")
    print(f"  Memory usage: {result.memory_usage_mb:.1f} MB")
    print(f"  CPU usage: {result.cpu_usage_percent:.1f}%")
    
    print(f"\n📈 BENCHMARK CONFIG")
    print(f"  Warmup runs: {result.warmup_runs}")
    print(f"  Benchmark runs: {result.benchmark_runs}")
    
    print("="*60)


def benchmark_model(model_path: str, 
                   warmup_runs: int = 10,
                   benchmark_runs: int = 100,
                   provider: str = "CPUExecutionProvider",
                   verbose: bool = True) -> BenchmarkResult:
    """
    Benchmark an ONNX model and return performance metrics.
    
    Args:
        model_path: Path to the ONNX model file
        warmup_runs: Number of warmup runs
        benchmark_runs: Number of benchmark runs
        provider: ONNX Runtime execution provider
        verbose: Whether to print progress and results
    
    Returns:
        BenchmarkResult with all performance metrics
    """
    benchmarker = ModelBenchmarker(model_path, provider)
    result = benchmarker.run_benchmark(warmup_runs, benchmark_runs)
    
    if verbose:
        print_benchmark_results(result)
    
    return result 