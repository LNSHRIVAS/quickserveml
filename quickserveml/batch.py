# quickserveml/batch.py

import time
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from onnxruntime import InferenceSession
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue


@dataclass
class BatchResult:
    """Results from batch processing"""
    batch_size: int
    total_time_seconds: float
    avg_time_per_sample_ms: float
    throughput_samples_per_second: float
    success_count: int
    error_count: int
    errors: List[str]


class BatchProcessor:
    """Handle batch processing for ONNX models"""
    
    def __init__(self, model_path: str, provider: str = "CPUExecutionProvider"):
        self.model_path = model_path
        self.provider = provider
        self.session = None
        self.input_name = None
        self.input_shape = None
        self._load_model()
    
    def _load_model(self):
        """Load the ONNX model"""
        self.session = InferenceSession(self.model_path, providers=[self.provider])
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.input_shape = [d if isinstance(d, int) and d > 0 else 1 for d in input_meta.shape]
    
    def _validate_batch_input(self, batch_data: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Validate and convert batch input data"""
        validated_batch = []
        
        for i, sample in enumerate(batch_data):
            if self.input_name not in sample:
                raise ValueError(f"Sample {i} missing required input: {self.input_name}")
            
            input_data = sample[self.input_name]
            
            # Convert to numpy array
            if isinstance(input_data, list):
                input_array = np.array(input_data, dtype=np.float32)
            elif isinstance(input_data, np.ndarray):
                input_array = input_data.astype(np.float32)
            else:
                raise ValueError(f"Sample {i} input must be a list or numpy array")
            
            # Validate shape
            if input_array.shape != tuple(self.input_shape):
                raise ValueError(f"Sample {i} has wrong shape. Expected {self.input_shape}, got {input_array.shape}")
            
            validated_batch.append(input_array)
        
        return validated_batch
    
    def process_batch_sync(self, batch_data: List[Dict[str, Any]]) -> BatchResult:
        """Process a batch of inputs synchronously"""
        start_time = time.time()
        errors = []
        success_count = 0
        error_count = 0
        
        try:
            # Validate batch
            validated_batch = self._validate_batch_input(batch_data)
            
            # Process each sample
            results = []
            for i, input_array in enumerate(validated_batch):
                try:
                    output = self.session.run(None, {self.input_name: input_array})
                    results.append(output)
                    success_count += 1
                except Exception as e:
                    errors.append(f"Sample {i}: {str(e)}")
                    error_count += 1
            
            total_time = time.time() - start_time
            
            return BatchResult(
                batch_size=len(batch_data),
                total_time_seconds=total_time,
                avg_time_per_sample_ms=(total_time * 1000) / len(batch_data),
                throughput_samples_per_second=len(batch_data) / total_time,
                success_count=success_count,
                error_count=error_count,
                errors=errors
            )
            
        except Exception as e:
            total_time = time.time() - start_time
            return BatchResult(
                batch_size=len(batch_data),
                total_time_seconds=total_time,
                avg_time_per_sample_ms=(total_time * 1000) / len(batch_data),
                throughput_samples_per_second=0,
                success_count=0,
                error_count=len(batch_data),
                errors=[f"Batch processing failed: {str(e)}"]
            )
    
    def process_batch_parallel(self, 
                             batch_data: List[Dict[str, Any]], 
                             max_workers: int = 4) -> BatchResult:
        """Process a batch of inputs in parallel"""
        start_time = time.time()
        errors = []
        success_count = 0
        error_count = 0
        
        def process_single_sample(sample_data):
            """Process a single sample"""
            try:
                if self.input_name not in sample_data:
                    return None, f"Missing required input: {self.input_name}"
                
                input_data = sample_data[self.input_name]
                
                # Convert to numpy array
                if isinstance(input_data, list):
                    input_array = np.array(input_data, dtype=np.float32)
                elif isinstance(input_data, np.ndarray):
                    input_array = input_data.astype(np.float32)
                else:
                    return None, f"Input must be a list or numpy array"
                
                # Validate shape
                if input_array.shape != tuple(self.input_shape):
                    return None, f"Wrong shape. Expected {self.input_shape}, got {input_array.shape}"
                
                # Run inference
                output = self.session.run(None, {self.input_name: input_array})
                return output, None
                
            except Exception as e:
                return None, str(e)
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(process_single_sample, sample): i 
                for i, sample in enumerate(batch_data)
            }
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result, error = future.result()
                    if error is None:
                        success_count += 1
                    else:
                        errors.append(f"Sample {index}: {error}")
                        error_count += 1
                except Exception as e:
                    errors.append(f"Sample {index}: {str(e)}")
                    error_count += 1
        
        total_time = time.time() - start_time
        
        return BatchResult(
            batch_size=len(batch_data),
            total_time_seconds=total_time,
            avg_time_per_sample_ms=(total_time * 1000) / len(batch_data),
            throughput_samples_per_second=len(batch_data) / total_time,
            success_count=success_count,
            error_count=error_count,
            errors=errors
        )
    
    def benchmark_batch_sizes(self, 
                            sample_data: Dict[str, Any],
                            batch_sizes: List[int] = [1, 4, 8, 16, 32],
                            runs_per_size: int = 5) -> Dict[int, BatchResult]:
        """Benchmark different batch sizes to find optimal performance"""
        print(f"🔍 Benchmarking batch sizes: {batch_sizes}")
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\n📊 Testing batch size: {batch_size}")
            
            # Create batch data
            batch_data = [sample_data.copy() for _ in range(batch_size)]
            
            # Run multiple times and take average
            batch_results = []
            for run in range(runs_per_size):
                result = self.process_batch_sync(batch_data)
                batch_results.append(result)
                
                if run < runs_per_size - 1:  # Don't print on last run
                    print(f"  Run {run + 1}: {result.throughput_samples_per_second:.1f} samples/sec")
            
            # Calculate average results
            avg_result = BatchResult(
                batch_size=batch_size,
                total_time_seconds=sum(r.total_time_seconds for r in batch_results) / len(batch_results),
                avg_time_per_sample_ms=sum(r.avg_time_per_sample_ms for r in batch_results) / len(batch_results),
                throughput_samples_per_second=sum(r.throughput_samples_per_second for r in batch_results) / len(batch_results),
                success_count=sum(r.success_count for r in batch_results) // len(batch_results),
                error_count=sum(r.error_count for r in batch_results) // len(batch_results),
                errors=[]
            )
            
            results[batch_size] = avg_result
            print(f"  Average: {avg_result.throughput_samples_per_second:.1f} samples/sec")
        
        return results


def print_batch_results(result: BatchResult):
    """Pretty print batch processing results"""
    print(f"\n📦 BATCH PROCESSING RESULTS")
    print(f"  Batch size: {result.batch_size}")
    print(f"  Total time: {result.total_time_seconds:.3f} seconds")
    print(f"  Avg time per sample: {result.avg_time_per_sample_ms:.2f} ms")
    print(f"  Throughput: {result.throughput_samples_per_second:.1f} samples/second")
    print(f"  Success: {result.success_count}/{result.batch_size}")
    print(f"  Errors: {result.error_count}")
    
    if result.errors:
        print(f"  Error details:")
        for error in result.errors[:5]:  # Show first 5 errors
            print(f"    - {error}")
        if len(result.errors) > 5:
            print(f"    ... and {len(result.errors) - 5} more errors")


def print_batch_benchmark_results(results: Dict[int, BatchResult]):
    """Print batch size benchmarking results"""
    print(f"\n📊 BATCH SIZE BENCHMARK RESULTS")
    print("=" * 60)
    
    print(f"{'Batch Size':<12} {'Throughput':<15} {'Avg Time':<12} {'Total Time':<12}")
    print("-" * 60)
    
    for batch_size in sorted(results.keys()):
        result = results[batch_size]
        print(f"{batch_size:<12} {result.throughput_samples_per_second:<15.1f} "
              f"{result.avg_time_per_sample_ms:<12.2f} {result.total_time_seconds:<12.3f}")
    
    # Find optimal batch size
    optimal_size = max(results.keys(), key=lambda x: results[x].throughput_samples_per_second)
    optimal_result = results[optimal_size]
    
    print("-" * 60)
    print(f"🎯 Optimal batch size: {optimal_size} "
          f"({optimal_result.throughput_samples_per_second:.1f} samples/sec)")


def process_batch(model_path: str,
                 batch_data: List[Dict[str, Any]],
                 parallel: bool = False,
                 max_workers: int = 4,
                 verbose: bool = True) -> BatchResult:
    """
    Process a batch of inputs through an ONNX model.
    
    Args:
        model_path: Path to the ONNX model
        batch_data: List of input dictionaries
        parallel: Whether to use parallel processing
        max_workers: Number of parallel workers
        verbose: Whether to print results
    
    Returns:
        BatchResult with processing metrics
    """
    processor = BatchProcessor(model_path)
    
    if parallel:
        result = processor.process_batch_parallel(batch_data, max_workers)
    else:
        result = processor.process_batch_sync(batch_data)
    
    if verbose:
        print_batch_results(result)
    
    return result


def benchmark_batch_sizes(model_path: str,
                         sample_data: Dict[str, Any],
                         batch_sizes: List[int] = [1, 4, 8, 16, 32],
                         runs_per_size: int = 5,
                         verbose: bool = True) -> Dict[int, BatchResult]:
    """
    Benchmark different batch sizes to find optimal performance.
    
    Args:
        model_path: Path to the ONNX model
        sample_data: Sample input data
        batch_sizes: List of batch sizes to test
        runs_per_size: Number of runs per batch size
        verbose: Whether to print results
    
    Returns:
        Dictionary mapping batch sizes to results
    """
    processor = BatchProcessor(model_path)
    results = processor.benchmark_batch_sizes(sample_data, batch_sizes, runs_per_size)
    
    if verbose:
        print_batch_benchmark_results(results)
    
    return results 