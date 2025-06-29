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
from quickserveml.cli_utils import get_formatter, success, info, warning, error, section_header

@click.group()
def cli():
    """
    🚀 QuickServeML - Lightning-fast ONNX model deployment
    
    A powerful CLI tool to inspect, benchmark, and deploy ONNX models 
    as production-ready FastAPI servers with minimal configuration.
    """
    pass

@cli.command()
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
def inspect(model_path):
    """Inspect an ONNX model and show input/output info."""
    formatter = get_formatter()
    
    try:
        section_header("🔍 Model Inspection", f"Analyzing {model_path}")
        inspect_onnx(model_path)
        success("Model inspection completed successfully")
    except Exception as e:
        error(f"Model inspection failed: {e}")
        raise click.Abort()

@cli.command()
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--save", help="Save schema to JSON file")
@click.option("--verbose", is_flag=True, help="Show detailed schema information")
def schema(model_path, save, verbose):
    """Generate detailed input/output schema for an ONNX model."""
    formatter = get_formatter()
    
    try:
        section_header("📋 Schema Generation", f"Generating schema for {model_path}")
        schema_obj = generate_schema(model_path, save_path=save)
        
        if verbose:
            info("Displaying detailed schema information")
            schema_obj.print_schema()
        else:
            # Show summary with rich formatting
            table = formatter.create_table("Schema Summary", show_header=True)
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="white")
            
            table.add_row("Model Path", model_path)
            table.add_row("Input Count", str(len(schema_obj.input_schemas)))
            table.add_row("Output Count", str(len(schema_obj.output_schemas)))
            
            formatter.print_table(table)
            
            # Input details
            if schema_obj.input_schemas:
                section_header("Input Details")
                for i, input_schema in enumerate(schema_obj.input_schemas):
                    shape_str = " × ".join(str(d) for d in input_schema.shape)
                    info(f"Input {i+1}: {input_schema.name} ({shape_str})")
            
            # Output details  
            if schema_obj.output_schemas:
                section_header("Output Details")
                for i, output_schema in enumerate(schema_obj.output_schemas):
                    shape_str = " × ".join(str(d) for d in output_schema.shape)
                    info(f"Output {i+1}: {output_schema.name} ({shape_str})")
        
        if save:
            success(f"Schema saved to {save}")
            
    except Exception as e:
        error(f"Schema generation failed: {e}")
        raise click.Abort()

@cli.command()
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--warmup-runs", default=10, help="Number of warmup runs")
@click.option("--benchmark-runs", default=100, help="Number of benchmark runs")
@click.option("--provider", default="CPUExecutionProvider", help="ONNX Runtime execution provider")
@click.option("--verbose", is_flag=True, help="Show detailed progress")
def benchmark(model_path, warmup_runs, benchmark_runs, provider, verbose):
    """Benchmark an ONNX model for performance metrics."""
    formatter = get_formatter()
    
    try:
        section_header("⚡ Performance Benchmarking", f"Testing {model_path}")
        
        # Show benchmark configuration
        config_table = formatter.create_table("Benchmark Configuration")
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="white")
        
        config_table.add_row("Warmup Runs", str(warmup_runs))
        config_table.add_row("Benchmark Runs", str(benchmark_runs))
        config_table.add_row("Provider", provider)
        config_table.add_row("Verbose", "Yes" if verbose else "No")
        
        formatter.print_table(config_table)
        
        # Run benchmark with progress indication
        with formatter.create_progress() as progress:
            task = progress.add_task("Running benchmark...", total=warmup_runs + benchmark_runs)
            
            result = benchmark_model(
                model_path=model_path,
                warmup_runs=warmup_runs,
                benchmark_runs=benchmark_runs,
                provider=provider,
                verbose=verbose
            )
            
            progress.update(task, advance=warmup_runs + benchmark_runs)
        
        # Display results in a formatted table
        results_table = formatter.create_table("📊 Benchmark Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")
        results_table.add_column("Unit", style="dim")
        
        results_table.add_row("Average Inference Time", f"{result.avg_inference_time_ms:.2f}", "ms")
        results_table.add_row("Throughput", f"{result.throughput_rps:.1f}", "requests/second")
        results_table.add_row("Memory Usage", f"{result.memory_usage_mb:.1f}", "MB")
        results_table.add_row("CPU Usage", f"{result.cpu_usage_percent:.1f}", "%")
        
        formatter.print_table(results_table)
        success("Benchmark completed successfully")
            
    except Exception as e:
        error(f"Benchmark failed: {e}")
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
    formatter = get_formatter()
    
    try:
        if optimize:
            section_header("🔍 Batch Size Optimization", f"Finding optimal batch size for {model_path}")
            
            # Generate sample data for optimization
            from onnxruntime import InferenceSession
            session = InferenceSession(model_path, providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name
            input_shape = [d if isinstance(d, int) and d > 0 else 1 for d in session.get_inputs()[0].shape]
            
            sample_data = {
                input_name: np.random.randn(*input_shape).astype(np.float32).tolist()
            }
            
            info("Generating sample data for optimization")
            
            with formatter.create_progress() as progress:
                task = progress.add_task("Optimizing batch sizes...", total=5)
                
                results = benchmark_batch_sizes(
                    model_path=model_path,
                    sample_data=sample_data,
                    batch_sizes=[1, 4, 8, 16, 32],
                    runs_per_size=3,
                    verbose=verbose
                )
                
                progress.update(task, advance=5)
            
            success("Batch size optimization completed")
            
        elif batch_file:
            section_header("📦 Batch Processing", f"Processing data from {batch_file}")
            
            # Load batch data from file
            with open(batch_file, 'r') as f:
                batch_data = json.load(f)
            
            if not isinstance(batch_data, list):
                error("Batch file must contain a list of input dictionaries")
                raise ValueError("Invalid batch file format")
            
            info(f"Loaded {len(batch_data)} samples from batch file")
            
            config_table = formatter.create_table("Processing Configuration")
            config_table.add_column("Parameter", style="cyan")
            config_table.add_column("Value", style="white")
            
            config_table.add_row("Batch Size", str(len(batch_data)))
            config_table.add_row("Parallel Processing", "Yes" if parallel else "No")
            if parallel:
                config_table.add_row("Max Workers", str(max_workers))
            
            formatter.print_table(config_table)
            
            with formatter.create_progress() as progress:
                task = progress.add_task("Processing batch...", total=len(batch_data))
                
                result = process_batch(
                    model_path=model_path,
                    batch_data=batch_data,
                    parallel=parallel,
                    max_workers=max_workers,
                    verbose=verbose
                )
                
                progress.update(task, advance=len(batch_data))
            
        else:
            section_header("📦 Synthetic Batch Processing", f"Generating and processing {batch_size} synthetic samples")
            
            # Generate synthetic batch data
            from onnxruntime import InferenceSession
            session = InferenceSession(model_path, providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name
            input_shape = [d if isinstance(d, int) and d > 0 else 1 for d in session.get_inputs()[0].shape]
            
            info("Generating synthetic batch data")
            
            batch_data = []
            for i in range(batch_size):
                sample_data = {
                    input_name: np.random.randn(*input_shape).astype(np.float32).tolist()
                }
                batch_data.append(sample_data)
            
            config_table = formatter.create_table("Processing Configuration")
            config_table.add_column("Parameter", style="cyan")
            config_table.add_column("Value", style="white")
            
            config_table.add_row("Batch Size", str(batch_size))
            config_table.add_row("Parallel Processing", "Yes" if parallel else "No")
            if parallel:
                config_table.add_row("Max Workers", str(max_workers))
            
            formatter.print_table(config_table)
            
            with formatter.create_progress() as progress:
                task = progress.add_task("Processing synthetic batch...", total=batch_size)
                
                result = process_batch(
                    model_path=model_path,
                    batch_data=batch_data,
                    parallel=parallel,
                    max_workers=max_workers,
                    verbose=verbose
                )
                
                progress.update(task, advance=batch_size)
        
        success("Batch processing completed successfully")
            
    except Exception as e:
        error(f"Batch processing failed: {e}")
        raise click.Abort()

@cli.command()
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--port", default=8000, help="Port to run the API on")
@click.option("--basic", is_flag=True, help="Deploy basic API (prediction only)")
def deploy(model_path, port, basic):
    """Deploy an ONNX model as a FastAPI server."""
    formatter = get_formatter()
    
    try:
        deployment_type = "Basic API" if basic else "Comprehensive API"
        section_header("🚀 Model Deployment", f"Deploying {deployment_type}")
        
        from onnxruntime import InferenceSession

        # Load model and extract input shape
        info("Loading and analyzing model...")
        session = InferenceSession(model_path, providers=["CPUExecutionProvider"])
        input_shape = session.get_inputs()[0].shape
        input_shape = [d if isinstance(d, int) and d > 0 else 1 for d in input_shape]
        model_filename = os.path.basename(model_path)

        # Copy the model to current dir for the server to access (only if different)
        if os.path.abspath(model_path) != os.path.abspath(f"./{model_filename}"):
            info(f"Copying model to working directory: {model_filename}")
            shutil.copy(model_path, f"./{model_filename}")

        # Choose template based on deployment type
        if basic:
            template_path = os.path.join("templates", "serve_template_basic.py.jinja")
            info("Using basic template (prediction endpoint only)")
        else:
            template_path = os.path.join("templates", "serve_template.py.jinja")
            info("Using comprehensive template (all features)")

        # Load and render Jinja template
        info("Generating FastAPI server code...")
        with open(template_path, encoding='utf-8') as f:
            template = Template(f.read())

        rendered = template.render(model_filename=model_filename, input_shape=input_shape)

        # Write the generated FastAPI server
        with open("serve.py", "w", encoding='utf-8') as f:
            f.write(rendered)

        success("FastAPI server generated as serve.py")
        
        # Display server information
        server_table = formatter.create_table("🌐 Server Information")
        server_table.add_column("Property", style="cyan")
        server_table.add_column("Value", style="white")
        
        server_table.add_row("Port", str(port))
        server_table.add_row("Type", deployment_type)
        server_table.add_row("Model File", model_filename)
        
        if basic:
            server_table.add_row("Prediction URL", f"http://localhost:{port}/predict")
        else:
            server_table.add_row("Base URL", f"http://localhost:{port}")
            server_table.add_row("Documentation", f"http://localhost:{port}/docs")
        
        formatter.print_table(server_table)
        
        if not basic:
            section_header("� Available Endpoints")
            endpoints = [
                "GET  / - API information",
                "GET  /health - Health check", 
                "GET  /model/info - Model information",
                "GET  /model/schema - Input/output schema",
                "POST /predict - Single prediction",
                "POST /model/benchmark - Performance benchmarking",
                "POST /model/batch - Batch processing"
            ]
            
            for endpoint in endpoints:
                info(f"  {endpoint}")

        info("Starting server with Uvicorn...")
        # Run the server with Uvicorn
        subprocess.run(["uvicorn", "serve:app", "--reload", "--port", str(port)])
        
    except Exception as e:
        error(f"Deployment failed: {e}")
        raise click.Abort()

@cli.command()
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--port", default=8000, help="Port to run the API on")
@click.option("--host", default="0.0.0.0", help="Host to bind the server to")
@click.option("--reload", is_flag=True, help="Enable auto-reload on code changes")
def serve(model_path, port, host, reload):
    """Deploy an ONNX model as a comprehensive FastAPI server with all features."""
    formatter = get_formatter()
    
    try:
        section_header("🚀 QuickServeML Server", "Deploying comprehensive FastAPI server")
        
        from onnxruntime import InferenceSession

        # Load model and extract input shape
        info("Loading and analyzing model...")
        session = InferenceSession(model_path, providers=["CPUExecutionProvider"])
        input_shape = session.get_inputs()[0].shape
        input_shape = [d if isinstance(d, int) and d > 0 else 1 for d in input_shape]
        model_filename = os.path.basename(model_path)

        # Copy the model to current dir for the server to access (only if different)
        if os.path.abspath(model_path) != os.path.abspath(f"./{model_filename}"):
            info(f"Copying model to working directory: {model_filename}")
            shutil.copy(model_path, f"./{model_filename}")

        # Load and render enhanced Jinja template
        info("Generating comprehensive FastAPI server...")
        template_path = os.path.join("templates", "serve_template.py.jinja")
        with open(template_path, encoding='utf-8') as f:
            template = Template(f.read())

        rendered = template.render(model_filename=model_filename, input_shape=input_shape)

        # Write the generated FastAPI server
        with open("serve.py", "w", encoding='utf-8') as f:
            f.write(rendered)

        success("Comprehensive FastAPI server generated as serve.py")
        
        # Server configuration table
        config_table = formatter.create_table("🌐 Server Configuration")
        config_table.add_column("Property", style="cyan")
        config_table.add_column("Value", style="white")
        
        config_table.add_row("Host", host)
        config_table.add_row("Port", str(port))
        config_table.add_row("Auto-reload", "Yes" if reload else "No")
        config_table.add_row("Model File", model_filename)
        
        formatter.print_table(config_table)
        
        # URLs table
        urls_table = formatter.create_table("� Important URLs")
        urls_table.add_column("Service", style="cyan")
        urls_table.add_column("URL", style="green")
        
        urls_table.add_row("API Base", f"http://{host}:{port}")
        urls_table.add_row("Interactive Docs", f"http://{host}:{port}/docs")
        urls_table.add_row("ReDoc Documentation", f"http://{host}:{port}/redoc")
        urls_table.add_row("Health Check", f"http://{host}:{port}/health")
        
        formatter.print_table(urls_table)
        
        # Endpoints information
        section_header("� Available API Endpoints")
        endpoints = [
            ("GET", "/", "API information and endpoints list"),
            ("GET", "/health", "Health check and system status"),
            ("GET", "/model/info", "Detailed model information"),
            ("GET", "/model/schema", "Input/output schema for API docs"),
            ("POST", "/predict", "Single prediction with timing"),
            ("POST", "/model/benchmark", "Performance benchmarking"),
            ("POST", "/model/batch", "Batch processing with optimization"),
            ("GET", "/model/compare", "Model comparison (future feature)")
        ]
        
        endpoints_table = formatter.create_table()
        endpoints_table.add_column("Method", style="bold blue")
        endpoints_table.add_column("Endpoint", style="cyan")
        endpoints_table.add_column("Description", style="white")
        
        for method, endpoint, description in endpoints:
            endpoints_table.add_row(method, endpoint, description)
        
        formatter.print_table(endpoints_table)
        
        # Example usage
        section_header("💡 Example Usage")
        examples = [
            f"curl http://{host}:{port}/health",
            f"curl http://{host}:{port}/model/info",
            f"curl -X POST http://{host}:{port}/model/benchmark -H 'Content-Type: application/json' -d '{{\"benchmark_runs\": 50}}'"
        ]
        
        for example in examples:
            info(f"  {example}")

        info("Starting server with Uvicorn...")
        
        # Run the server with Uvicorn
        cmd = ["uvicorn", "serve:app", "--host", host, "--port", str(port)]
        if reload:
            cmd.append("--reload")
        
        subprocess.run(cmd)
        
    except Exception as e:
        error(f"Server deployment failed: {e}")
        raise click.Abort()

if __name__ == "__main__":
    cli()
