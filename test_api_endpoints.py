#!/usr/bin/env python3
"""
Comprehensive API Endpoint Testing Script

This script tests all QuickServeML API endpoints with synthetic data
and various edge cases to ensure robustness.
"""

import requests
import json
import time
import numpy as np
from typing import Dict, Any, List

# Configuration
BASE_URL = "http://localhost:8001"
TIMEOUT = 30

def print_test_header(test_name: str):
    """Print a formatted test header"""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING: {test_name}")
    print(f"{'='*60}")

def print_success(message: str):
    """Print success message"""
    print(f"✅ {message}")

def print_error(message: str):
    """Print error message"""
    print(f"❌ {message}")

def print_warning(message: str):
    """Print warning message"""
    print(f"⚠️  {message}")

def test_endpoint(method: str, endpoint: str, data: Dict = None, expected_status: int = 200) -> Dict:
    """Test an API endpoint and return the response"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, timeout=TIMEOUT)
        elif method.upper() == "POST":
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, json=data, headers=headers, timeout=TIMEOUT)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        if response.status_code == expected_status:
            print_success(f"{method} {endpoint} - Status: {response.status_code}")
            return response.json() if response.content else {}
        else:
            print_error(f"{method} {endpoint} - Expected {expected_status}, got {response.status_code}")
            print(f"Response: {response.text}")
            return {}
            
    except requests.exceptions.RequestException as e:
        print_error(f"{method} {endpoint} - Request failed: {e}")
        return {}
    except json.JSONDecodeError as e:
        print_error(f"{method} {endpoint} - JSON decode failed: {e}")
        return {}

def test_root_endpoint():
    """Test the root endpoint"""
    print_test_header("Root Endpoint")
    
    response = test_endpoint("GET", "/")
    
    if response:
        # Validate response structure
        required_keys = ["message", "version", "model", "endpoints"]
        for key in required_keys:
            if key in response:
                print_success(f"Response contains '{key}': {response[key]}")
            else:
                print_error(f"Missing required key: {key}")
        
        # Check if all expected endpoints are listed
        expected_endpoints = ["/predict", "/health", "/model/info", "/model/schema", 
                            "/model/benchmark", "/model/batch", "/model/compare"]
        endpoints = response.get("endpoints", {})
        for endpoint in expected_endpoints:
            if endpoint in str(endpoints):
                print_success(f"Endpoint listed: {endpoint}")
            else:
                print_warning(f"Endpoint not found in listing: {endpoint}")

def test_health_endpoint():
    """Test the health endpoint"""
    print_test_header("Health Endpoint")
    
    response = test_endpoint("GET", "/health")
    
    if response:
        # Validate health response
        required_keys = ["status", "model_loaded", "memory_usage_mb", "cpu_usage_percent"]
        for key in required_keys:
            if key in response:
                print_success(f"Health check contains '{key}': {response[key]}")
            else:
                print_error(f"Missing health key: {key}")
        
        # Check if model is loaded
        if response.get("status") == "healthy":
            print_success("Health status is healthy")
        else:
            print_error(f"Health status is not healthy: {response.get('status')}")
        
        # Validate resource metrics
        memory = response.get("memory_usage_mb", 0)
        cpu = response.get("cpu_usage_percent", 0)
        
        if isinstance(memory, (int, float)) and memory >= 0:
            print_success(f"Memory usage valid: {memory:.1f} MB")
        else:
            print_error(f"Invalid memory usage: {memory}")
        
        if isinstance(cpu, (int, float)) and 0 <= cpu <= 100:
            print_success(f"CPU usage valid: {cpu:.1f}%")
        else:
            print_error(f"Invalid CPU usage: {cpu}")

def test_model_info_endpoint():
    """Test the model info endpoint"""
    print_test_header("Model Info Endpoint")
    
    response = test_endpoint("GET", "/model/info")
    
    if response:
        # Validate model info structure
        required_keys = ["model_file", "inputs", "outputs", "session_providers"]
        for key in required_keys:
            if key in response:
                print_success(f"Model info contains '{key}'")
            else:
                print_error(f"Missing model info key: {key}")
        
        # Validate inputs and outputs
        inputs = response.get("inputs", [])
        outputs = response.get("outputs", [])
        
        if inputs:
            print_success(f"Model has {len(inputs)} input(s)")
            for i, inp in enumerate(inputs):
                if all(key in inp for key in ["name", "shape", "type"]):
                    print_success(f"Input {i+1} valid: {inp['name']} {inp['shape']}")
                else:
                    print_error(f"Input {i+1} missing required fields")
        else:
            print_error("No inputs found")
        
        if outputs:
            print_success(f"Model has {len(outputs)} output(s)")
            for i, out in enumerate(outputs):
                if all(key in out for key in ["name", "shape", "type"]):
                    print_success(f"Output {i+1} valid: {out['name']} {out['shape']}")
                else:
                    print_error(f"Output {i+1} missing required fields")
        else:
            print_error("No outputs found")

def test_model_schema_endpoint():
    """Test the model schema endpoint"""
    print_test_header("Model Schema Endpoint")
    
    response = test_endpoint("GET", "/model/schema")
    
    if response:
        # Validate schema structure
        required_keys = ["input_schema", "output_schema"]
        for key in required_keys:
            if key in response:
                print_success(f"Schema contains '{key}'")
            else:
                print_error(f"Missing schema key: {key}")
        
        # Validate input schema
        input_schema = response.get("input_schema", {})
        if "properties" in input_schema:
            print_success("Input schema has properties")
        else:
            print_error("Input schema missing properties")
        
        # Validate output schema
        output_schema = response.get("output_schema", {})
        if "properties" in output_schema:
            print_success("Output schema has properties")
        else:
            print_error("Output schema missing properties")

def generate_test_data(input_shape: List[int]) -> List[float]:
    """Generate synthetic test data"""
    # Create random data with the expected shape
    size = 1
    for dim in input_shape:
        if isinstance(dim, int) and dim > 0:
            size *= dim
        else:
            size *= 1  # Default to 1 for unknown dimensions
    
    return np.random.randn(size).astype(np.float32).tolist()

def test_prediction_endpoint():
    """Test the prediction endpoint with various scenarios"""
    print_test_header("Prediction Endpoint")
    
    # First, get model info to understand input requirements
    model_info = test_endpoint("GET", "/model/info")
    
    if not model_info or not model_info.get("inputs"):
        print_error("Cannot test prediction without model info")
        return
    
    input_name = model_info["inputs"][0]["name"]
    input_shape = model_info["inputs"][0]["shape"]
    
    # Test 1: Valid prediction
    print("\n📝 Test 1: Valid prediction")
    test_data = generate_test_data(input_shape)
    prediction_data = {"data": test_data}
    
    response = test_endpoint("POST", "/predict", prediction_data)
    
    if response:
        required_keys = ["prediction", "inference_time_ms", "input_shape", "output_shape"]
        for key in required_keys:
            if key in response:
                print_success(f"Prediction response contains '{key}'")
            else:
                print_error(f"Missing prediction key: {key}")
        
        # Validate inference time
        inference_time = response.get("inference_time_ms", 0)
        if isinstance(inference_time, (int, float)) and inference_time >= 0:
            print_success(f"Inference time: {inference_time:.3f} ms")
        else:
            print_error(f"Invalid inference time: {inference_time}")
    
    # Test 2: Empty data (edge case)
    print("\n📝 Test 2: Empty data (should fail)")
    test_endpoint("POST", "/predict", {"data": []}, expected_status=400)
    
    # Test 3: Wrong data type (edge case)
    print("\n📝 Test 3: Wrong data type (should fail)")
    test_endpoint("POST", "/predict", {"data": "not_a_list"}, expected_status=422)
    
    # Test 4: Missing data field (edge case)
    print("\n📝 Test 4: Missing data field (should fail)")
    test_endpoint("POST", "/predict", {}, expected_status=422)
    
    # Test 5: Wrong data size (edge case)
    print("\n📝 Test 5: Wrong data size (should fail)")
    wrong_size_data = {"data": [1.0, 2.0, 3.0]}  # Too small
    test_endpoint("POST", "/predict", wrong_size_data, expected_status=400)

def test_benchmark_endpoint():
    """Test the benchmark endpoint with various configurations"""
    print_test_header("Benchmark Endpoint")
    
    # Test 1: Default benchmark
    print("\n📝 Test 1: Default benchmark")
    response = test_endpoint("POST", "/model/benchmark", {})
    
    if response:
        # Validate benchmark response structure
        if "benchmark_config" in response and "results" in response:
            print_success("Benchmark response has correct structure")
            
            results = response["results"]
            required_metrics = [
                "avg_inference_time_ms", "min_inference_time_ms", "max_inference_time_ms",
                "p95_inference_time_ms", "p99_inference_time_ms", "throughput_rps",
                "memory_usage_mb", "cpu_usage_percent", "total_runs"
            ]
            
            for metric in required_metrics:
                if metric in results:
                    value = results[metric]
                    if isinstance(value, (int, float)) and value >= 0:
                        print_success(f"{metric}: {value}")
                    else:
                        print_error(f"Invalid {metric}: {value}")
                else:
                    print_error(f"Missing metric: {metric}")
        else:
            print_error("Benchmark response missing required sections")
    
    # Test 2: Custom benchmark configuration
    print("\n📝 Test 2: Custom benchmark configuration")
    custom_config = {
        "warmup_runs": 5,
        "benchmark_runs": 50,
        "provider": "CPUExecutionProvider"
    }
    response = test_endpoint("POST", "/model/benchmark", custom_config)
    
    if response:
        config = response.get("benchmark_config", {})
        if config.get("warmup_runs") == 5 and config.get("benchmark_runs") == 50:
            print_success("Custom configuration applied correctly")
        else:
            print_error("Custom configuration not applied")
    
    # Test 3: Invalid configuration (edge case)
    print("\n📝 Test 3: Invalid configuration (should fail)")
    invalid_config = {
        "warmup_runs": -1,  # Invalid negative value
        "benchmark_runs": 0  # Invalid zero value
    }
    test_endpoint("POST", "/model/benchmark", invalid_config, expected_status=422)
    
    # Test 4: Very large benchmark (edge case)
    print("\n📝 Test 4: Large benchmark (should work but take time)")
    large_config = {
        "warmup_runs": 10,
        "benchmark_runs": 1000,  # Large number
        "provider": "CPUExecutionProvider"
    }
    start_time = time.time()
    response = test_endpoint("POST", "/model/benchmark", large_config)
    end_time = time.time()
    
    if response:
        print_success(f"Large benchmark completed in {end_time - start_time:.2f} seconds")
    else:
        print_error("Large benchmark failed")

def test_batch_endpoint():
    """Test the batch endpoint with various scenarios"""
    print_test_header("Batch Endpoint")
    
    # Get model info for input requirements
    model_info = test_endpoint("GET", "/model/info")
    
    if not model_info or not model_info.get("inputs"):
        print_error("Cannot test batch without model info")
        return
    
    input_shape = model_info["inputs"][0]["shape"]
    
    # Test 1: Small batch
    print("\n📝 Test 1: Small batch (5 samples)")
    batch_data = []
    for i in range(5):
        sample_data = generate_test_data(input_shape)
        batch_data.append(sample_data)
    
    batch_request = {
        "batch": batch_data,
        "batch_size": 10,
        "parallel": False
    }
    
    response = test_endpoint("POST", "/model/batch", batch_request)
    
    if response:
        if "batch_config" in response and "results" in response:
            print_success("Batch response has correct structure")
            
            results = response["results"]
            required_keys = ["total_samples", "total_time_ms", "avg_time_per_sample_ms", 
                           "throughput_samples_per_sec", "predictions"]
            
            for key in required_keys:
                if key in results:
                    print_success(f"Batch results contain '{key}'")
                else:
                    print_error(f"Missing batch result key: {key}")
            
            # Validate predictions
            predictions = results.get("predictions", [])
            if len(predictions) == 5:
                print_success(f"Got {len(predictions)} predictions as expected")
            else:
                print_error(f"Expected 5 predictions, got {len(predictions)}")
        else:
            print_error("Batch response missing required sections")
    
    # Test 2: Empty batch (edge case)
    print("\n📝 Test 2: Empty batch (should fail)")
    empty_batch = {
        "batch": [],
        "batch_size": 10,
        "parallel": False
    }
    test_endpoint("POST", "/model/batch", empty_batch, expected_status=400)
    
    # Test 3: Batch too large (edge case)
    print("\n📝 Test 3: Batch too large (should fail)")
    large_batch = []
    for i in range(100):  # Create 100 samples
        sample_data = generate_test_data(input_shape)
        large_batch.append(sample_data)
    
    large_batch_request = {
        "batch": large_batch,
        "batch_size": 50,  # But limit is 50
        "parallel": False
    }
    test_endpoint("POST", "/model/batch", large_batch_request, expected_status=400)
    
    # Test 4: Invalid data in batch (edge case)
    print("\n📝 Test 4: Invalid data in batch (should fail)")
    invalid_batch = {
        "batch": [["not", "numbers"], [1, 2, 3]],  # Mixed data types
        "batch_size": 10,
        "parallel": False
    }
    test_endpoint("POST", "/model/batch", invalid_batch, expected_status=400)

def test_model_compare_endpoint():
    """Test the model comparison endpoint (placeholder feature)"""
    print_test_header("Model Compare Endpoint")
    
    response = test_endpoint("GET", "/model/compare")
    
    if response:
        if "message" in response and "current_model" in response:
            print_success("Compare endpoint returns expected structure")
            print_success(f"Message: {response['message']}")
            print_success(f"Current model: {response['current_model']}")
        else:
            print_error("Compare endpoint missing expected fields")

def test_error_handling():
    """Test various error scenarios"""
    print_test_header("Error Handling")
    
    # Test 1: Non-existent endpoint
    print("\n📝 Test 1: Non-existent endpoint")
    test_endpoint("GET", "/nonexistent", expected_status=404)
    
    # Test 2: Invalid JSON
    print("\n📝 Test 2: Invalid JSON (should fail)")
    try:
        response = requests.post(f"{BASE_URL}/predict", 
                               data="invalid json", 
                               headers={"Content-Type": "application/json"},
                               timeout=TIMEOUT)
        if response.status_code == 422:
            print_success("Invalid JSON properly rejected")
        else:
            print_error(f"Invalid JSON not properly handled: {response.status_code}")
    except Exception as e:
        print_error(f"Invalid JSON test failed: {e}")
    
    # Test 3: Missing Content-Type header
    print("\n📝 Test 3: Missing Content-Type header")
    try:
        response = requests.post(f"{BASE_URL}/predict", 
                               json={"data": [1.0, 2.0, 3.0]},
                               timeout=TIMEOUT)
        if response.status_code in [200, 400, 422]:  # Any of these are acceptable
            print_success("Missing Content-Type handled appropriately")
        else:
            print_error(f"Unexpected response for missing Content-Type: {response.status_code}")
    except Exception as e:
        print_error(f"Missing Content-Type test failed: {e}")

def test_performance():
    """Test API performance under load"""
    print_test_header("Performance Testing")
    
    # Test concurrent requests
    print("\n📝 Test: Concurrent health checks")
    import threading
    
    results = []
    errors = []
    
    def make_request():
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            results.append(response.status_code)
        except Exception as e:
            errors.append(str(e))
    
    # Start 10 concurrent requests
    threads = []
    for i in range(10):
        thread = threading.Thread(target=make_request)
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    success_count = len([r for r in results if r == 200])
    print_success(f"Concurrent requests: {success_count}/10 successful")
    
    if errors:
        print_warning(f"Errors during concurrent requests: {len(errors)}")

def main():
    """Run all tests"""
    print("🚀 QuickServeML API Comprehensive Testing")
    print(f"📡 Testing server at: {BASE_URL}")
    print(f"⏱️  Timeout: {TIMEOUT} seconds")
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print_success("Server is running and responding")
        else:
            print_error("Server is not responding correctly")
            return
    except Exception as e:
        print_error(f"Cannot connect to server: {e}")
        print("Please make sure the server is running with: quickserveml serve mnist-8.onnx --port 8002")
        return
    
    # Run all tests
    test_root_endpoint()
    test_health_endpoint()
    test_model_info_endpoint()
    test_model_schema_endpoint()
    test_prediction_endpoint()
    test_benchmark_endpoint()
    test_batch_endpoint()
    test_model_compare_endpoint()
    test_error_handling()
    test_performance()
    
    print(f"\n{'='*60}")
    print("🎉 API Testing Complete!")
    print(f"{'='*60}")
    print("📊 Summary:")
    print("✅ All endpoints tested with valid data")
    print("✅ Edge cases and error conditions tested")
    print("✅ Performance under load tested")
    print("✅ Error handling validated")
    print("\n💡 Next steps:")
    print("   • Review any warnings or errors above")
    print("   • Test with real model data if needed")
    print("   • Consider adding more edge cases")
    print("   • Document any issues found")

if __name__ == "__main__":
    main() 