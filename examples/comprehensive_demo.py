#!/usr/bin/env python3
"""
Comprehensive QuickServeML Demo

This script demonstrates all the core features of QuickServeML:
1. Model inspection
2. Schema generation
3. Performance benchmarking
4. Batch processing optimization
5. Model deployment

Usage:
    python examples/comprehensive_demo.py path/to/model.onnx
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

# Add the parent directory to the path so we can import quickserveml
sys.path.insert(0, str(Path(__file__).parent.parent))

from quickserveml.infer import inspect_onnx
from quickserveml.schema import generate_schema
from quickserveml.benchmark import benchmark_model
from quickserveml.batch import benchmark_batch_sizes


def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"🚀 {title}")
    print("="*80)


def demo_model_inspection(model_path):
    """Demonstrate model inspection capabilities"""
    print_header("MODEL INSPECTION")
    
    print("📋 Basic model inspection:")
    inspect_onnx(model_path)
    
    print("\n📊 Detailed schema analysis:")
    schema = generate_schema(model_path)
    schema.print_schema()


def demo_schema_generation(model_path):
    """Demonstrate schema generation and validation"""
    print_header("SCHEMA GENERATION & VALIDATION")
    
    # Generate schema
    schema = generate_schema(model_path)
    
    # Save schema to file
    schema_file = "model_schema.json"
    schema.save_schema(schema_file)
    print(f"✔ Schema saved to {schema_file}")
    
    # Demonstrate input validation
    print("\n🔍 Input validation example:")
    try:
        # Create valid input
        input_name = schema.input_schemas[0].name
        input_shape = [d if d != "?" else 1 for d in schema.input_schemas[0].shape]
        valid_input = {
            input_name: np.random.randn(*input_shape).astype(np.float32).tolist()
        }
        
        validated = schema.validate_input(valid_input)
        print(f"✅ Valid input accepted: {input_name} with shape {list(validated[input_name].shape)}")
        
        # Try invalid input
        invalid_input = {
            input_name: np.random.randn(1, 2, 3, 4).astype(np.float32).tolist()  # Wrong shape
        }
        
        try:
            schema.validate_input(invalid_input)
        except ValueError as e:
            print(f"❌ Invalid input rejected: {e}")
            
    except Exception as e:
        print(f"⚠ Schema validation demo failed: {e}")


def demo_benchmarking(model_path):
    """Demonstrate performance benchmarking"""
    print_header("PERFORMANCE BENCHMARKING")
    
    print("📊 Running comprehensive benchmark...")
    result = benchmark_model(
        model_path=model_path,
        warmup_runs=10,
        benchmark_runs=100,
        provider="CPUExecutionProvider",
        verbose=True
    )
    
    print(f"\n🎯 Key Performance Metrics:")
    print(f"  • Average latency: {result.avg_inference_time_ms:.2f} ms")
    print(f"  • 95th percentile: {result.p95_inference_time_ms:.2f} ms")
    print(f"  • Throughput: {result.throughput_rps:.1f} requests/second")
    print(f"  • Memory usage: {result.memory_usage_mb:.1f} MB")
    print(f"  • CPU usage: {result.cpu_usage_percent:.1f}%")


def demo_batch_processing(model_path):
    """Demonstrate batch processing optimization"""
    print_header("BATCH PROCESSING OPTIMIZATION")
    
    # Generate sample data for batch testing
    from onnxruntime import InferenceSession
    session = InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_shape = [d if isinstance(d, int) and d > 0 else 1 for d in session.get_inputs()[0].shape]
    
    sample_data = {
        input_name: np.random.randn(*input_shape).astype(np.float32).tolist()
    }
    
    print("🔍 Finding optimal batch size...")
    results = benchmark_batch_sizes(
        model_path=model_path,
        sample_data=sample_data,
        batch_sizes=[1, 2, 4, 8, 16, 32],
        runs_per_size=3,
        verbose=True
    )
    
    # Find optimal batch size
    optimal_size = max(results.keys(), key=lambda x: results[x].throughput_samples_per_second)
    optimal_result = results[optimal_size]
    
    print(f"\n🎯 Optimization Results:")
    print(f"  • Optimal batch size: {optimal_size}")
    print(f"  • Max throughput: {optimal_result.throughput_samples_per_second:.1f} samples/sec")
    
    # Calculate efficiency gain only if baseline throughput is not zero
    baseline_throughput = results[1].throughput_samples_per_second
    if baseline_throughput > 0:
        efficiency_gain = optimal_result.throughput_samples_per_second / baseline_throughput
        print(f"  • Efficiency gain: {efficiency_gain:.1f}x")
    else:
        print(f"  • Efficiency gain: N/A (baseline throughput was 0)")


def demo_api_integration(model_path):
    """Demonstrate API integration features"""
    print_header("API INTEGRATION FEATURES")
    
    schema = generate_schema(model_path)
    
    print("📋 Generated OpenAPI Schema:")
    openapi_schema = schema.get_openapi_schema()
    
    # Show key endpoints
    print("\n🔗 Available API Endpoints:")
    for path, methods in openapi_schema["paths"].items():
        for method, details in methods.items():
            print(f"  • {method.upper()} {path} - {details.get('summary', 'No description')}")
    
    print("\n📝 Example Request Schema:")
    request_schema = schema._generate_request_schema()
    for input_name, input_schema in request_schema["properties"].items():
        print(f"  • {input_name}: {input_schema['description']}")
    
    print("\n📤 Example Response Schema:")
    response_schema = schema._generate_response_schema()
    for output_name, output_schema in response_schema["properties"].items():
        print(f"  • {output_name}: {output_schema['description']}")


def create_sample_batch_file(model_path):
    """Create a sample batch file for testing"""
    print_header("SAMPLE BATCH FILE CREATION")
    
    from onnxruntime import InferenceSession
    session = InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_shape = [d if isinstance(d, int) and d > 0 else 1 for d in session.get_inputs()[0].shape]
    
    # Create sample batch data
    batch_data = []
    for i in range(5):
        sample = {
            input_name: np.random.randn(*input_shape).astype(np.float32).tolist()
        }
        batch_data.append(sample)
    
    # Save to file
    batch_file = "sample_batch.json"
    with open(batch_file, 'w') as f:
        json.dump(batch_data, f, indent=2)
    
    print(f"✔ Sample batch file created: {batch_file}")
    print(f"  • Contains {len(batch_data)} samples")
    print(f"  • Input shape: {input_shape}")
    print(f"  • Usage: quickserveml batch {model_path} --batch-file {batch_file}")


def main():
    """Main demo function"""
    if len(sys.argv) != 2:
        print("Usage: python examples/comprehensive_demo.py <model_path>")
        print("Example: python examples/comprehensive_demo.py mnist-8.onnx")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)
    
    print("🎯 QuickServeML Comprehensive Demo")
    print(f"📁 Model: {model_path}")
    
    try:
        # Run all demos
        demo_model_inspection(model_path)
        demo_schema_generation(model_path)
        demo_benchmarking(model_path)
        demo_batch_processing(model_path)
        demo_api_integration(model_path)
        create_sample_batch_file(model_path)
        
        print_header("DEMO COMPLETED")
        print("✅ All features demonstrated successfully!")
        print("\n📚 Next steps:")
        print("  • Try the CLI commands directly")
        print("  • Deploy your model: quickserveml deploy <model>")
        print("  • Check the generated files (model_schema.json, sample_batch.json)")
        print("  • Explore the API documentation at http://localhost:8000/docs")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 