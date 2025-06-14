# quickserveml/cli.py

import os
import subprocess
import shutil
import click
import json
import numpy as np
from jinja2 import Template
from quickserveml.infer import inspect_onnx
from quickserveml.benchmark import benchmark_model
from quickserveml.schema import generate_schema
from quickserveml.batch import process_batch, benchmark_batch_sizes

@click.group()
def cli():
    """QuickServeML CLI"""
    pass

@cli.command()
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
def inspect(model_path):
    """Inspect an ONNX model and show input/output info."""
    inspect_onnx(model_path)

@cli.command()
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--save", help="Save schema to JSON file")
@click.option("--verbose", is_flag=True, help="Show detailed schema information")
def schema(model_path, save, verbose):
    """Generate detailed input/output schema for an ONNX model."""
    try:
        schema_obj = generate_schema(model_path, save_path=save)
        
        if verbose:
            schema_obj.print_schema()
        else:
            # Show summary
            print(f"\n📋 Schema Summary for {model_path}")
            print(f"  Inputs: {len(schema_obj.input_schemas)}")
            print(f"  Outputs: {len(schema_obj.output_schemas)}")
            
            for i, input_schema in enumerate(schema_obj.input_schemas):
                shape_str = " x ".join(str(d) for d in input_schema.shape)
                print(f"  Input {i+1}: {input_schema.name} ({shape_str})")
            
            for i, output_schema in enumerate(schema_obj.output_schemas):
                shape_str = " x ".join(str(d) for d in output_schema.shape)
                print(f"  Output {i+1}: {output_schema.name} ({shape_str})")
        
        if save:
            print(f"✔ Schema saved to {save}")
            
    except Exception as e:
        print(f"❌ Schema generation failed: {e}")
        raise click.Abort()

@cli.command()
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--warmup-runs", default=10, help="Number of warmup runs")
@click.option("--benchmark-runs", default=100, help="Number of benchmark runs")
@click.option("--provider", default="CPUExecutionProvider", help="ONNX Runtime execution provider")
@click.option("--verbose", is_flag=True, help="Show detailed progress")
def benchmark(model_path, warmup_runs, benchmark_runs, provider, verbose):
    """Benchmark an ONNX model for performance metrics."""
    try:
        result = benchmark_model(
            model_path=model_path,
            warmup_runs=warmup_runs,
            benchmark_runs=benchmark_runs,
            provider=provider,
            verbose=verbose
        )
        
        if not verbose:
            # Show summary even if not verbose
            print(f"\n📊 Quick Summary:")
            print(f"  Avg inference time: {result.avg_inference_time_ms:.2f} ms")
            print(f"  Throughput: {result.throughput_rps:.1f} requests/second")
            print(f"  Memory usage: {result.memory_usage_mb:.1f} MB")
            print(f"  CPU usage: {result.cpu_usage_percent:.1f}%")
            
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        raise click.Abort()

@cli.command()
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--batch-file", help="JSON file containing batch data")
@click.option("--batch-size", default=10, help="Generate batch data with this size")
@click.option("--parallel", is_flag=True, help="Use parallel processing")
@click.option("--max-workers", default=4, help="Number of parallel workers")
@click.option("--optimize", is_flag=True, help="Find optimal batch size")
@click.option("--verbose", is_flag=True, help="Show detailed results")
def batch(model_path, batch_file, batch_size, parallel, max_workers, optimize, verbose):
    """Process batches of inputs through an ONNX model."""
    try:
        if optimize:
            # Generate sample data for optimization
            from onnxruntime import InferenceSession
            session = InferenceSession(model_path, providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name
            input_shape = [d if isinstance(d, int) and d > 0 else 1 for d in session.get_inputs()[0].shape]
            
            sample_data = {
                input_name: np.random.randn(*input_shape).astype(np.float32).tolist()
            }
            
            print(f"🔍 Optimizing batch size for {model_path}")
            results = benchmark_batch_sizes(
                model_path=model_path,
                sample_data=sample_data,
                batch_sizes=[1, 4, 8, 16, 32],
                runs_per_size=3,
                verbose=verbose
            )
            
        elif batch_file:
            # Load batch data from file
            with open(batch_file, 'r') as f:
                batch_data = json.load(f)
            
            if not isinstance(batch_data, list):
                raise ValueError("Batch file must contain a list of input dictionaries")
            
            print(f"📦 Processing batch of {len(batch_data)} samples")
            result = process_batch(
                model_path=model_path,
                batch_data=batch_data,
                parallel=parallel,
                max_workers=max_workers,
                verbose=verbose
            )
            
        else:
            # Generate synthetic batch data
            from onnxruntime import InferenceSession
            session = InferenceSession(model_path, providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name
            input_shape = [d if isinstance(d, int) and d > 0 else 1 for d in session.get_inputs()[0].shape]
            
            batch_data = []
            for i in range(batch_size):
                sample_data = {
                    input_name: np.random.randn(*input_shape).astype(np.float32).tolist()
                }
                batch_data.append(sample_data)
            
            print(f"📦 Processing synthetic batch of {batch_size} samples")
            result = process_batch(
                model_path=model_path,
                batch_data=batch_data,
                parallel=parallel,
                max_workers=max_workers,
                verbose=verbose
            )
            
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        raise click.Abort()

@cli.command()
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--port", default=8000, help="Port to run the API on")
@click.option("--basic", is_flag=True, help="Deploy basic API (prediction only)")
def deploy(model_path, port, basic):
    """Deploy an ONNX model as a FastAPI server."""
    from onnxruntime import InferenceSession

    # Load model and extract input shape
    session = InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_shape = session.get_inputs()[0].shape
    input_shape = [d if isinstance(d, int) and d > 0 else 1 for d in input_shape]
    model_filename = os.path.basename(model_path)

    # Copy the model to current dir for the server to access (only if different)
    if os.path.abspath(model_path) != os.path.abspath(f"./{model_filename}"):
        shutil.copy(model_path, f"./{model_filename}")

    # Choose template based on deployment type
    if basic:
        template_path = os.path.join("templates", "serve_template_basic.py.jinja")
        print("🚀 Deploying basic API (prediction only)...")
    else:
        template_path = os.path.join("templates", "serve_template.py.jinja")
        print("🚀 Deploying comprehensive API with all features...")

    # Load and render Jinja template
    with open(template_path, encoding='utf-8') as f:
        template = Template(f.read())

    rendered = template.render(model_filename=model_filename, input_shape=input_shape)

    # Write the generated FastAPI server
    with open("serve.py", "w", encoding='utf-8') as f:
        f.write(rendered)

    print(f"✔ FastAPI server generated as serve.py")
    
    if basic:
        print(f"🚀 Running basic server at http://localhost:{port}/predict ...")
    else:
        print(f"🚀 Running comprehensive server at http://localhost:{port}")
        print(f"📚 API Documentation: http://localhost:{port}/docs")
        print(f"🔍 Available endpoints:")
        print(f"   • GET  / - API information")
        print(f"   • GET  /health - Health check")
        print(f"   • GET  /model/info - Model information")
        print(f"   • GET  /model/schema - Input/output schema")
        print(f"   • POST /predict - Single prediction")
        print(f"   • POST /model/benchmark - Performance benchmarking")
        print(f"   • POST /model/batch - Batch processing")

    # Run the server with Uvicorn
    subprocess.run(["uvicorn", "serve:app", "--reload", "--port", str(port)])

@cli.command()
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--port", default=8000, help="Port to run the API on")
@click.option("--host", default="0.0.0.0", help="Host to bind the server to")
@click.option("--reload", is_flag=True, help="Enable auto-reload on code changes")
def serve(model_path, port, host, reload):
    """Deploy an ONNX model as a comprehensive FastAPI server with all features."""
    from onnxruntime import InferenceSession

    # Load model and extract input shape
    session = InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_shape = session.get_inputs()[0].shape
    input_shape = [d if isinstance(d, int) and d > 0 else 1 for d in input_shape]
    model_filename = os.path.basename(model_path)

    # Copy the model to current dir for the server to access (only if different)
    if os.path.abspath(model_path) != os.path.abspath(f"./{model_filename}"):
        shutil.copy(model_path, f"./{model_filename}")

    # Load and render enhanced Jinja template
    template_path = os.path.join("templates", "serve_template.py.jinja")
    with open(template_path, encoding='utf-8') as f:
        template = Template(f.read())

    rendered = template.render(model_filename=model_filename, input_shape=input_shape)

    # Write the generated FastAPI server
    with open("serve.py", "w", encoding='utf-8') as f:
        f.write(rendered)

    print(f"✔ Comprehensive FastAPI server generated as serve.py")
    print(f"🚀 Starting QuickServeML API server...")
    print(f"📡 Server URL: http://{host}:{port}")
    print(f"📚 Interactive API docs: http://{host}:{port}/docs")
    print(f"📖 ReDoc documentation: http://{host}:{port}/redoc")
    print(f"\n🔍 Available endpoints:")
    print(f"   • GET  / - API information and endpoints list")
    print(f"   • GET  /health - Health check and system status")
    print(f"   • GET  /model/info - Detailed model information")
    print(f"   • GET  /model/schema - Input/output schema for API docs")
    print(f"   • POST /predict - Single prediction with timing")
    print(f"   • POST /model/benchmark - Performance benchmarking")
    print(f"   • POST /model/batch - Batch processing with optimization")
    print(f"   • GET  /model/compare - Model comparison (future feature)")
    
    print(f"\n💡 Example usage:")
    print(f"   curl http://{host}:{port}/health")
    print(f"   curl http://{host}:{port}/model/info")
    print(f"   curl -X POST http://{host}:{port}/model/benchmark -H 'Content-Type: application/json' -d '{{\"benchmark_runs\": 50}}'")

    # Run the server with Uvicorn
    cmd = ["uvicorn", "serve:app", "--host", host, "--port", str(port)]
    if reload:
        cmd.append("--reload")
    
    subprocess.run(cmd)

if __name__ == "__main__":
    cli()
