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
        section_header("🔍 Model Inspection", f"Analyzing model: {os.path.basename(model_path)}")
        
        from onnxruntime import InferenceSession
        import onnx

        # Load model
        info("Loading ONNX model...")
        session = InferenceSession(model_path, providers=["CPUExecutionProvider"])
        model = onnx.load(model_path)

        # Get model metadata
        success("Model loaded successfully")
        
        # Display model information
        info("📋 Model Information:")
        model_table = formatter.create_table()
        model_table.add_column("Property", style="cyan")
        model_table.add_column("Value", style="white")
        
        model_table.add_row("Model Path", model_path)
        model_table.add_row("File Size", f"{os.path.getsize(model_path):,} bytes")
        model_table.add_row("ONNX Version", str(model.ir_version))
        model_table.add_row("Producer", model.producer_name or "Unknown")
        model_table.add_row("Domain", model.domain or "Unknown")
        
        formatter.print_table(model_table)

        # Display input information
        section_header("📥 Model Inputs")
        inputs_table = formatter.create_table()
        inputs_table.add_column("Input", style="bold cyan")
        inputs_table.add_column("Shape", style="green")
        inputs_table.add_column("Type", style="yellow")
        
        for input_info in session.get_inputs():
            shape_str = "x".join(str(dim) if dim > 0 else "?" for dim in input_info.shape)
            inputs_table.add_row(input_info.name, shape_str, input_info.type)
        
        formatter.print_table(inputs_table)

        # Display output information
        section_header("📤 Model Outputs")
        outputs_table = formatter.create_table()
        outputs_table.add_column("Output", style="bold cyan")
        outputs_table.add_column("Shape", style="green")
        outputs_table.add_column("Type", style="yellow")
        
        for output_info in session.get_outputs():
            shape_str = "x".join(str(dim) if dim > 0 else "?" for dim in output_info.shape)
            outputs_table.add_row(output_info.name, shape_str, output_info.type)
        
        formatter.print_table(outputs_table)

        success("✅ Model inspection completed")
        
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
        section_header("📋 Schema Generation", f"Generating schema for: {os.path.basename(model_path)}")
        
        from onnxruntime import InferenceSession
        import onnx
        import json

        # Load model
        info("Loading ONNX model...")
        session = InferenceSession(model_path, providers=["CPUExecutionProvider"])
        model = onnx.load(model_path)

        # Generate schema
        success("Model loaded successfully")
        info("Generating input/output schema...")
        
        schema_data = {
            "model_info": {
                "name": os.path.basename(model_path),
                "path": model_path,
                "file_size": os.path.getsize(model_path),
                "onnx_version": model.ir_version,
                "producer": model.producer_name or "Unknown",
                "domain": model.domain or "Unknown"
            },
            "inputs": [],
            "outputs": []
        }

        # Process inputs
        for input_info in session.get_inputs():
            input_schema = {
                "name": input_info.name,
                "type": input_info.type,
                "shape": input_info.shape,
                "shape_str": "x".join(str(dim) if dim > 0 else "?" for dim in input_info.shape)
            }
            schema_data["inputs"].append(input_schema)

        # Process outputs
        for output_info in session.get_outputs():
            output_schema = {
                "name": output_info.name,
                "type": output_info.type,
                "shape": output_info.shape,
                "shape_str": "x".join(str(dim) if dim > 0 else "?" for dim in output_info.shape)
            }
            schema_data["outputs"].append(output_schema)

        # Display schema
        if verbose:
            info("📋 Detailed Schema:")
            schema_table = formatter.create_table()
            schema_table.add_column("Component", style="bold cyan")
            schema_table.add_column("Name", style="green")
            schema_table.add_column("Shape", style="yellow")
            schema_table.add_column("Type", style="white")
            
            for input_schema in schema_data["inputs"]:
                schema_table.add_row("INPUT", input_schema["name"], input_schema["shape_str"], input_schema["type"])
            
            for output_schema in schema_data["outputs"]:
                schema_table.add_row("OUTPUT", output_schema["name"], output_schema["shape_str"], output_schema["type"])
            
            formatter.print_table(schema_table)
        else:
            info(f"📥 Inputs: {len(schema_data['inputs'])}")
            for input_schema in schema_data["inputs"]:
                info(f"  - {input_schema['name']}: {input_schema['shape_str']} ({input_schema['type']})")
            
            info(f"📤 Outputs: {len(schema_data['outputs'])}")
            for output_schema in schema_data["outputs"]:
                info(f"  - {output_schema['name']}: {output_schema['shape_str']} ({output_schema['type']})")

        # Save schema if requested
        if save:
            with open(save, 'w') as f:
                json.dump(schema_data, f, indent=2)
            success(f"✅ Schema saved to: {save}")

        success("✅ Schema generation completed")
            
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
        section_header("⚡ Performance Benchmarking", f"Benchmarking: {os.path.basename(model_path)}")
        
        from onnxruntime import InferenceSession
        import numpy as np
        import time
        import statistics

        # Load model
        info("Loading ONNX model...")
        session = InferenceSession(model_path, providers=[provider])
        
        # Get input details
        input_meta = session.get_inputs()[0]
        input_name = input_meta.name
        input_shape = [d if isinstance(d, int) and d > 0 else 1 for d in input_meta.shape]
        
        # Get output details
        output_meta = session.get_outputs()[0]
        output_shape = [d if isinstance(d, int) and d > 0 else 1 for d in output_meta.shape]
        
        success("Model loaded successfully")
        info(f"Input: {input_name} {input_shape}")
        info(f"Output: {output_meta.name} {output_shape}")
        
        # Generate dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        info(f"🔥 Warming up model with {warmup_runs} runs...")
        for i in range(warmup_runs):
            _ = session.run(None, {input_name: dummy_input})
            if verbose and (i + 1) % 5 == 0:
                info(f"  Warmup run {i + 1}/{warmup_runs}")
        
        # Benchmark
        info(f"⏱️  Running {benchmark_runs} inference benchmarks...")
        inference_times = []
        
        for i in range(benchmark_runs):
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            
            start_time = time.perf_counter()
            _ = session.run(None, {input_name: dummy_input})
            end_time = time.perf_counter()
            
            inference_time_ms = (end_time - start_time) * 1000
            inference_times.append(inference_time_ms)
            
            if verbose and (i + 1) % 20 == 0:
                info(f"  Benchmark run {i + 1}/{benchmark_runs}")
        
        # Calculate statistics
        avg_time = statistics.mean(inference_times)
        min_time = min(inference_times)
        max_time = max(inference_times)
        p95_time = statistics.quantiles(inference_times, n=20)[18]  # 95th percentile
        p99_time = statistics.quantiles(inference_times, n=100)[98]  # 99th percentile
        
        # Calculate throughput
        throughput_rps = 1000 / avg_time if avg_time > 0 else 0
        
        # Measure memory usage
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Display results
        section_header("📊 Benchmark Results")
        
        results_table = formatter.create_table()
        results_table.add_column("Metric", style="bold cyan")
        results_table.add_column("Value", style="white")
        
        results_table.add_row("Average Latency", f"{avg_time:.2f} ms")
        results_table.add_row("Min Latency", f"{min_time:.2f} ms")
        results_table.add_row("Max Latency", f"{max_time:.2f} ms")
        results_table.add_row("95th Percentile", f"{p95_time:.2f} ms")
        results_table.add_row("99th Percentile", f"{p99_time:.2f} ms")
        results_table.add_row("Throughput", f"{throughput_rps:.1f} requests/second")
        results_table.add_row("Memory Usage", f"{memory_mb:.1f} MB")
        
        formatter.print_table(results_table)
        
        success("✅ Benchmarking completed")
        
    except Exception as e:
        error(f"Benchmarking failed: {e}")
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
        section_header("📦 Batch Processing", f"Processing: {os.path.basename(model_path)}")
        
        from onnxruntime import InferenceSession
        import numpy as np
        import time
        import statistics

        # Load model
        info("Loading ONNX model...")
        session = InferenceSession(model_path, providers=["CPUExecutionProvider"])
        
        # Get input details
        input_meta = session.get_inputs()[0]
        input_name = input_meta.name
        input_shape = [d if isinstance(d, int) and d > 0 else 1 for d in input_meta.shape]
        
        success("Model loaded successfully")
        info(f"Input shape: {input_shape}")
        
        # Generate or load batch data
        if batch_file:
            info(f"Loading batch data from: {batch_file}")
            with open(batch_file, 'r') as f:
                batch_data = json.load(f)
        else:
            info(f"Generating synthetic batch data (size: {batch_size})")
            batch_data = []
            for i in range(batch_size):
                # Generate random data matching the input shape
                data = np.random.randn(*input_shape).astype(np.float32).tolist()
                batch_data.append(data)
        
        info(f"Processing {len(batch_data)} inputs...")
        
        # Process batch
        start_time = time.perf_counter()
        
        if parallel:
            info(f"Using parallel processing with {max_workers} workers")
            from concurrent.futures import ThreadPoolExecutor
            
            def process_single(input_data):
                return session.run(None, {input_name: np.array(input_data, dtype=np.float32)})
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(process_single, batch_data))
        else:
            info("Using sequential processing")
            results = []
            for i, input_data in enumerate(batch_data):
                result = session.run(None, {input_name: np.array(input_data, dtype=np.float32)})
                results.append(result)
                
                if verbose and (i + 1) % 10 == 0:
                    info(f"  Processed {i + 1}/{len(batch_data)} inputs")
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Calculate statistics
        throughput = len(batch_data) / total_time
        avg_time_per_input = total_time / len(batch_data) * 1000  # Convert to ms
        
        # Display results
        section_header("📊 Batch Processing Results")
        
        results_table = formatter.create_table()
        results_table.add_column("Metric", style="bold cyan")
        results_table.add_column("Value", style="white")
        
        results_table.add_row("Total Inputs", str(len(batch_data)))
        results_table.add_row("Processing Mode", "Parallel" if parallel else "Sequential")
        if parallel:
            results_table.add_row("Workers", str(max_workers))
        results_table.add_row("Total Time", f"{total_time:.3f} seconds")
        results_table.add_row("Average Time per Input", f"{avg_time_per_input:.2f} ms")
        results_table.add_row("Throughput", f"{throughput:.1f} inputs/second")
        
        formatter.print_table(results_table)
        
        # Optimize batch size if requested
        if optimize:
            section_header("🔍 Batch Size Optimization")
            info("Testing different batch sizes to find optimal throughput...")
            
            batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
            optimization_results = []
            
            for test_size in batch_sizes:
                if test_size > len(batch_data):
                    break
                    
                test_batch = batch_data[:test_size]
                start_time = time.perf_counter()
                
                for _ in range(3):  # Run 3 times for averaging
                    for input_data in test_batch:
                        session.run(None, {input_name: np.array(input_data, dtype=np.float32)})
                
                end_time = time.perf_counter()
                avg_time = (end_time - start_time) / 3
                throughput = test_size / avg_time
                
                optimization_results.append({
                    "batch_size": test_size,
                    "throughput": throughput,
                    "avg_time": avg_time
                })
                
                if verbose:
                    info(f"  Batch size {test_size}: {throughput:.1f} inputs/second")
            
            # Find optimal batch size
            if optimization_results:
                optimal = max(optimization_results, key=lambda x: x["throughput"])
                success(f"✅ Optimal batch size: {optimal['batch_size']} (throughput: {optimal['throughput']:.1f} inputs/second)")
        
        success("✅ Batch processing completed")
        
    except Exception as e:
        error(f"Batch processing failed: {e}")
        raise click.Abort()

@cli.command()
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--port", default=8000, help="Port to run the API on")
@click.option("--host", default="0.0.0.0", help="Host to bind the server to")
@click.option("--reload", is_flag=True, help="Enable auto-reload on code changes")
def serve(model_path, port, host, reload):
    """Deploy an ONNX model as a FastAPI server."""
    formatter = get_formatter()
    
    try:
        section_header("🚀 Model Deployment", f"Deploying {os.path.basename(model_path)}")
        
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

        # Load and render Jinja template
        info("Generating FastAPI server code...")
        template_path = os.path.join("templates", "serve_template.py.jinja")
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
        
        server_table.add_row("Host", "localhost" if host == "0.0.0.0" else host)
        server_table.add_row("Port", str(port))
        server_table.add_row("Type", "Comprehensive API")
        server_table.add_row("Model File", model_filename)
        
        formatter.print_table(server_table)
        
        # URLs table
        urls_table = formatter.create_table(" Important URLs")
        urls_table.add_column("Service", style="cyan")
        urls_table.add_column("URL", style="green")
        
        display_host = "localhost" if host == "0.0.0.0" else host
        urls_table.add_row("API Base", f"http://{display_host}:{port}")
        urls_table.add_row("Interactive Docs", f"http://{display_host}:{port}/docs")
        urls_table.add_row("ReDoc Documentation", f"http://{display_host}:{port}/redoc")
        urls_table.add_row("Health Check", f"http://{display_host}:{port}/health")
        
        formatter.print_table(urls_table)
        
        # Endpoints information
        section_header(" Available API Endpoints")
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
            f"curl http://{display_host}:{port}/health",
            f"curl http://{display_host}:{port}/model/info",
            f"curl -X POST http://{display_host}:{port}/model/benchmark -H 'Content-Type: application/json' -d '{{\"benchmark_runs\": 50}}'"
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

@cli.command()
@click.argument("name")
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--version", help="Model version (auto-generated if not provided)")
@click.option("--description", help="Model description")
@click.option("--tags", help="Comma-separated tags")
@click.option("--author", help="Model author")
@click.option("--status", type=click.Choice(["draft", "validated", "staging", "production", "archived"]), 
              default="draft", help="Model status")
def registry_add(name, model_path, version, description, tags, author, status):
    """Add a model to the registry with versioning and metadata."""
    formatter = get_formatter()
    
    try:
        section_header("📦 Model Registry", f"Adding model '{name}' to registry")
        
        from .registry import ModelRegistry
        
        # Parse tags
        tag_list = [tag.strip() for tag in tags.split(",")] if tags else []
        
        # Initialize registry
        registry = ModelRegistry()
        
        # Add model to registry
        metadata = registry.add_model(
            name=name,
            model_path=model_path,
            version=version,
            description=description or "",
            tags=tag_list,
            author=author or "",
            status=status
        )
        
        # Display model information
        info("📋 Model Details:")
        model_table = formatter.create_table()
        model_table.add_column("Property", style="cyan")
        model_table.add_column("Value", style="white")
        
        model_table.add_row("Name", metadata.name)
        model_table.add_row("Version", metadata.version)
        model_table.add_row("Status", metadata.status)
        model_table.add_row("Author", metadata.author or "Unknown")
        model_table.add_row("File Size", f"{metadata.file_size:,} bytes")
        model_table.add_row("Hash", metadata.model_hash[:16] + "...")
        
        if metadata.tags:
            model_table.add_row("Tags", ", ".join(metadata.tags))
        
        formatter.print_table(model_table)
        
        success(f"✅ Model '{name}:{metadata.version}' successfully added to registry")
        
    except Exception as e:
        error(f"Failed to add model to registry: {e}")
        raise click.Abort()

@cli.command()
@click.option("--status", help="Filter by status")
@click.option("--tags", help="Filter by tags (comma-separated)")
@click.option("--verbose", is_flag=True, help="Show detailed information")
def registry_list(status, tags, verbose):
    """List all models in the registry."""
    formatter = get_formatter()
    
    try:
        section_header("📋 Model Registry", "Listing registered models")
        
        from .registry import ModelRegistry
        
        # Parse tags filter
        tag_list = [tag.strip() for tag in tags.split(",")] if tags else None
        
        # Initialize registry
        registry = ModelRegistry()
        
        # Get models
        models = registry.list_models(status=status, tags=tag_list)
        
        if not models:
            info("No models found in registry")
            return
        
        # Display models
        if verbose:
            # Detailed table
            models_table = formatter.create_table("📦 Registered Models")
            models_table.add_column("Name", style="bold cyan")
            models_table.add_column("Version", style="green")
            models_table.add_column("Status", style="yellow")
            models_table.add_column("Author", style="white")
            models_table.add_column("Created", style="dim")
            models_table.add_column("Size", style="dim")
            
            for model in models:
                created_date = model.created_at.split("T")[0] if model.created_at else "Unknown"
                size_mb = f"{model.file_size / 1024 / 1024:.1f} MB" if model.file_size else "Unknown"
                
                models_table.add_row(
                    model.name,
                    model.version,
                    model.status,
                    model.author or "Unknown",
                    created_date,
                    size_mb
                )
            
            formatter.print_table(models_table)
        else:
            # Simple list
            info(f"Found {len(models)} model(s) in registry:")
            for model in models:
                status_icon = {
                    "draft": "📝",
                    "validated": "✅",
                    "staging": "🚀",
                    "production": "🏭",
                    "archived": "📦"
                }.get(model.status, "❓")
                
                print(f"  {status_icon} {model.name}:{model.version} ({model.status})")
        
        success(f"✅ Listed {len(models)} model(s)")
        
    except Exception as e:
        error(f"Failed to list models: {e}")
        raise click.Abort()

@cli.command()
@click.argument("name")
@click.argument("version")
@click.option("--status", type=click.Choice(["draft", "validated", "staging", "production", "archived"]), 
              help="Update model status")
@click.option("--description", help="Update model description")
@click.option("--tags", help="Update tags (comma-separated)")
@click.option("--accuracy", type=float, help="Update accuracy metric")
@click.option("--latency", type=float, help="Update latency metric (ms)")
@click.option("--throughput", type=float, help="Update throughput metric (RPS)")
@click.option("--notes", help="Add notes about the model")
def registry_update(name, version, status, description, tags, accuracy, latency, throughput, notes):
    """Update model metadata in the registry."""
    formatter = get_formatter()
    
    try:
        section_header("✏️ Model Registry", f"Updating '{name}:{version}'")
        
        from .registry import ModelRegistry
        
        # Initialize registry
        registry = ModelRegistry()
        
        # Prepare update data
        update_data = {}
        if status:
            update_data["status"] = status
        if description:
            update_data["description"] = description
        if tags:
            update_data["tags"] = [tag.strip() for tag in tags.split(",")]
        if accuracy is not None:
            update_data["accuracy"] = accuracy
        if latency is not None:
            update_data["latency_ms"] = latency
        if throughput is not None:
            update_data["throughput_rps"] = throughput
        if notes:
            update_data["notes"] = notes
        
        if not update_data:
            error("No update data provided. Use options to specify what to update.")
            return
        
        # Update model
        updated_model = registry.update_model(name, version, **update_data)
        
        # Show what was updated
        info("📝 Updated fields:")
        for field, value in update_data.items():
            info(f"  {field}: {value}")
        
        success(f"✅ Model '{name}:{version}' updated successfully")
        
    except Exception as e:
        error(f"Failed to update model: {e}")
        raise click.Abort()

@cli.command()
@click.argument("name")
@click.argument("version")
@click.option("--force", is_flag=True, help="Force deletion without confirmation")
def registry_delete(name, version, force):
    """Delete a model from the registry."""
    formatter = get_formatter()
    
    try:
        section_header("🗑️ Model Registry", f"Deleting '{name}:{version}'")
        
        from .registry import ModelRegistry
        
        # Initialize registry
        registry = ModelRegistry()
        
        # Check if model exists
        model = registry.get_model(name, version)
        if not model:
            error(f"Model '{name}:{version}' not found in registry")
            return
        
        # Confirm deletion
        if not force:
            confirm = click.confirm(f"Are you sure you want to delete '{name}:{version}'? This action cannot be undone.")
            if not confirm:
                info("Deletion cancelled")
                return
        
        # Delete model
        registry.delete_model(name, version)
        
        success(f"✅ Model '{name}:{version}' deleted from registry")
        
    except Exception as e:
        error(f"Failed to delete model: {e}")
        raise click.Abort()

@cli.command()
@click.argument("name")
@click.argument("version1")
@click.argument("version2")
def registry_compare(name, version1, version2):
    """Compare two versions of a model."""
    formatter = get_formatter()
    
    try:
        section_header("⚖️ Model Comparison", f"Comparing '{name}:{version1}' vs '{name}:{version2}'")
        
        from .registry import ModelRegistry
        
        # Initialize registry
        registry = ModelRegistry()
        
        # Compare models
        comparison = registry.compare_models(name, version1, version2)
        
        # Display comparison results
        comparison_table = formatter.create_table("📊 Comparison Results")
        comparison_table.add_column("Metric", style="cyan")
        comparison_table.add_column(f"{version1}", style="green")
        comparison_table.add_column(f"{version2}", style="blue")
        comparison_table.add_column("Difference", style="yellow")
        
        for metric, data in comparison["metrics"].items():
            v1_val = data["version1"]
            v2_val = data["version2"]
            diff = data["difference"]
            # Format values based on type, handling None gracefully
            if v1_val is None:
                v1_str = "-"
            elif isinstance(v1_val, float):
                v1_str = f"{v1_val:.4f}"
            else:
                v1_str = str(v1_val)
            if v2_val is None:
                v2_str = "-"
            elif isinstance(v2_val, float):
                v2_str = f"{v2_val:.4f}"
            else:
                v2_str = str(v2_val)
            if diff is None:
                diff_str = "-"
            elif isinstance(diff, float):
                diff_str = f"{diff:+.4f}"
            elif isinstance(diff, int):
                diff_str = f"{diff:+d}"
            else:
                diff_str = str(diff)
            comparison_table.add_row(metric, v1_str, v2_str, diff_str)
        
        formatter.print_table(comparison_table)
        
        success(f"✅ Comparison completed for '{name}'")
        
    except Exception as e:
        error(f"Failed to compare models: {e}")
        raise click.Abort()

@cli.command()
@click.argument("name")
@click.argument("output_path")
@click.option("--version", help="Model version (latest if not specified)")
def registry_export(name, output_path, version):
    """Export a model from the registry to a file."""
    formatter = get_formatter()
    
    try:
        section_header("📤 Model Export", f"Exporting '{name}' to {output_path}")
        
        from .registry import ModelRegistry
        
        # Initialize registry
        registry = ModelRegistry()
        
        # Export model
        registry.export_model(name, version, output_path)
        
        # Get model info for display
        model = registry.get_model(name, version)
        if model:
            info(f"📋 Exported model details:")
            info(f"  Name: {model.name}")
            info(f"  Version: {model.version}")
            info(f"  Size: {model.file_size:,} bytes")
            info(f"  Hash: {model.model_hash[:16]}...")
        
        success(f"✅ Model exported successfully to {output_path}")
        
    except Exception as e:
        error(f"Failed to export model: {e}")
        raise click.Abort()

@cli.command()
@click.argument("model_name")
@click.option("--version", help="Model version (latest if not specified)")
@click.option("--warmup-runs", default=10, help="Number of warmup runs")
@click.option("--benchmark-runs", default=100, help="Number of benchmark runs")
@click.option("--provider", default="CPUExecutionProvider", help="ONNX Runtime execution provider")
@click.option("--verbose", is_flag=True, help="Show detailed progress")
@click.option("--save-metrics", is_flag=True, help="Save benchmark metrics to registry")
def benchmark_registry(model_name, version, warmup_runs, benchmark_runs, provider, verbose, save_metrics):
    """Benchmark a model from the registry."""
    formatter = get_formatter()
    
    try:
        section_header("⚡ Registry Benchmark", f"Benchmarking: {model_name}")
        
        from .registry import ModelRegistry
        
        # Get model from registry
        registry = ModelRegistry()
        model_metadata = registry.get_model(model_name, version)
        if not model_metadata:
            error(f"Model '{model_name}' not found in registry")
            return
        
        info(f"📦 Using model: {model_name}:{model_metadata.version}")
        
        from onnxruntime import InferenceSession
        import numpy as np
        import time
        import statistics

        # Load model
        info("Loading ONNX model...")
        session = InferenceSession(model_metadata.model_path, providers=[provider])
        
        # Get input shape
        input_info = session.get_inputs()[0]
        input_shape = input_info.shape
        input_shape = [d if isinstance(d, int) and d > 0 else 1 for d in input_shape]
        
        success("Model loaded successfully")
        
        # Display configuration
        info("📊 Benchmark Configuration:")
        config_table = formatter.create_table()
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="white")
        
        config_table.add_row("Model", f"{model_name}:{model_metadata.version}")
        config_table.add_row("Input Shape", "x".join(map(str, input_shape)))
        config_table.add_row("Provider", provider)
        config_table.add_row("Warmup Runs", str(warmup_runs))
        config_table.add_row("Benchmark Runs", str(benchmark_runs))
        
        formatter.print_table(config_table)

        # Generate random input data
        info("Generating test data...")
        input_data = np.random.random(input_shape).astype(np.float32)
        
        # Warmup runs
        if verbose:
            info(f"🔥 Warming up with {warmup_runs} runs...")
        
        with formatter.progress_bar(total=warmup_runs, description="Warmup") as progress:
            for _ in range(warmup_runs):
                session.run(None, {input_info.name: input_data})
                progress.advance(1)

        # Benchmark runs
        if verbose:
            info(f"⚡ Benchmarking with {benchmark_runs} runs...")
        
        latencies = []
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        with formatter.progress_bar(total=benchmark_runs, description="Benchmark") as progress:
            for _ in range(benchmark_runs):
                start_time = time.perf_counter()
                session.run(None, {input_info.name: input_data})
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
                progress.advance(1)
        
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_used = end_memory - start_memory

        # Calculate statistics
        latencies = np.array(latencies)
        avg_latency = np.mean(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        throughput = 1000 / avg_latency  # requests per second

        # Display results
        section_header("📊 Benchmark Results")
        results_table = formatter.create_table()
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="white")
        results_table.add_column("Unit", style="dim")
        
        results_table.add_row("Average Latency", f"{avg_latency:.2f}", "ms")
        results_table.add_row("Min Latency", f"{min_latency:.2f}", "ms")
        results_table.add_row("Max Latency", f"{max_latency:.2f}", "ms")
        results_table.add_row("95th Percentile", f"{p95_latency:.2f}", "ms")
        results_table.add_row("99th Percentile", f"{p99_latency:.2f}", "ms")
        results_table.add_row("Throughput", f"{throughput:.2f}", "RPS")
        results_table.add_row("Memory Usage", f"{memory_used:.2f}", "MB")
        
        formatter.print_table(results_table)

        # Save metrics to registry if requested
        if save_metrics:
            info("💾 Saving all metrics to registry...")
            # Build the full metrics dictionary
            metrics = {
                "average_latency": avg_latency,
                "min_latency": min_latency,
                "max_latency": max_latency,
                "p95_latency": p95_latency,
                "p99_latency": p99_latency,
                "throughput": throughput,
                "memory_usage": memory_used,
            }
            # Save all metrics to the model's metrics field
            registry.update_model(
                model_name,
                model_metadata.version,
                metrics=metrics
            )
            success("✅ All metrics saved to registry")

        success("✅ Registry benchmark completed successfully")
        
    except Exception as e:
        error(f"Registry benchmark failed: {e}")
        raise click.Abort()

@cli.command()
@click.argument("model_name")
@click.option("--version", help="Model version (latest if not specified)")
@click.option("--port", default=8000, help="Port to run the API on")
@click.option("--host", default="0.0.0.0", help="Host to bind the server to")
@click.option("--reload", is_flag=True, help="Enable auto-reload on code changes")
def serve_registry(model_name, version, port, host, reload):
    """Deploy a model from the registry as a comprehensive FastAPI server."""
    formatter = get_formatter()
    
    try:
        section_header("🚀 Registry Server", f"Deploying {model_name} from registry")
        
        from .registry import ModelRegistry
        
        # Get model from registry
        registry = ModelRegistry()
        model_metadata = registry.get_model(model_name, version)
        if not model_metadata:
            error(f"Model '{model_name}' not found in registry")
            return
        
        info(f"📦 Using model: {model_name}:{model_metadata.version}")
        
        from onnxruntime import InferenceSession

        # Load model and extract input shape
        info("Loading and analyzing model...")
        session = InferenceSession(model_metadata.model_path, providers=["CPUExecutionProvider"])
        input_shape = session.get_inputs()[0].shape
        input_shape = [d if isinstance(d, int) and d > 0 else 1 for d in input_shape]
        model_filename = os.path.basename(model_metadata.model_path)

        # Copy the model to current dir for the server to access
        if os.path.abspath(model_metadata.model_path) != os.path.abspath(f"./{model_filename}"):
            info(f"Copying model to working directory: {model_filename}")
            shutil.copy(model_metadata.model_path, f"./{model_filename}")

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
        
        config_table.add_row("Model", f"{model_name}:{model_metadata.version}")
        config_table.add_row("Host", host)
        config_table.add_row("Port", str(port))
        config_table.add_row("Auto-reload", "Yes" if reload else "No")
        config_table.add_row("Model File", model_filename)
        
        formatter.print_table(config_table)
        
        # URLs table
        urls_table = formatter.create_table(" Important URLs")
        urls_table.add_column("Service", style="cyan")
        urls_table.add_column("URL", style="green")
        
        urls_table.add_row("API Base", f"http://{host}:{port}")
        urls_table.add_row("Interactive Docs", f"http://{host}:{port}/docs")
        urls_table.add_row("ReDoc Documentation", f"http://{host}:{port}/redoc")
        urls_table.add_row("Health Check", f"http://{host}:{port}/health")
        
        formatter.print_table(urls_table)
        
        # Endpoints information
        section_header(" Available API Endpoints")
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
        error(f"Registry server deployment failed: {e}")
        raise click.Abort()

@cli.command()
@click.argument("model_name")
@click.option("--version", help="Model version (latest if not specified)")
def inspect_registry(model_name, version):
    """Inspect a model from the registry."""
    formatter = get_formatter()
    
    try:
        section_header("🔍 Registry Inspection", f"Analyzing: {model_name}")
        
        from .registry import ModelRegistry
        
        # Get model from registry
        registry = ModelRegistry()
        model_metadata = registry.get_model(model_name, version)
        if not model_metadata:
            error(f"Model '{model_name}' not found in registry")
            return
        
        info(f"📦 Using model: {model_name}:{model_metadata.version}")
        
        from onnxruntime import InferenceSession
        import onnx

        # Load model
        info("Loading ONNX model...")
        session = InferenceSession(model_metadata.model_path, providers=["CPUExecutionProvider"])
        model = onnx.load(model_metadata.model_path)

        # Get model metadata
        success("Model loaded successfully")
        
        # Display registry information
        info("📋 Registry Information:")
        registry_table = formatter.create_table()
        registry_table.add_column("Property", style="cyan")
        registry_table.add_column("Value", style="white")
        
        registry_table.add_row("Name", model_metadata.name)
        registry_table.add_row("Version", model_metadata.version)
        registry_table.add_row("Status", model_metadata.status)
        registry_table.add_row("Author", model_metadata.author or "Unknown")
        registry_table.add_row("Description", model_metadata.description or "No description")
        registry_table.add_row("Created", model_metadata.created_at)
        registry_table.add_row("Updated", model_metadata.updated_at)
        
        if model_metadata.tags:
            registry_table.add_row("Tags", ", ".join(model_metadata.tags))
        
        if model_metadata.accuracy is not None:
            registry_table.add_row("Accuracy", f"{model_metadata.accuracy:.4f}")
        
        if model_metadata.latency_ms is not None:
            registry_table.add_row("Latency", f"{model_metadata.latency_ms:.2f} ms")
        
        if model_metadata.throughput_rps is not None:
            registry_table.add_row("Throughput", f"{model_metadata.throughput_rps:.2f} RPS")
        
        formatter.print_table(registry_table)
        
        # Display model information
        info("📋 Model Information:")
        model_table = formatter.create_table()
        model_table.add_column("Property", style="cyan")
        model_table.add_column("Value", style="white")
        
        model_table.add_row("Model Path", model_metadata.model_path)
        model_table.add_row("File Size", f"{model_metadata.file_size:,} bytes")
        model_table.add_row("Hash", model_metadata.model_hash[:16] + "...")
        model_table.add_row("ONNX Version", str(model.ir_version))
        model_table.add_row("Producer", model.producer_name or "Unknown")
        model_table.add_row("Domain", model.domain or "Unknown")
        
        formatter.print_table(model_table)

        # Display input information
        section_header("📥 Model Inputs")
        inputs_table = formatter.create_table()
        inputs_table.add_column("Input", style="bold cyan")
        inputs_table.add_column("Shape", style="green")
        inputs_table.add_column("Type", style="yellow")
        
        for input_info in session.get_inputs():
            shape_str = "x".join(str(dim) if dim > 0 else "?" for dim in input_info.shape)
            inputs_table.add_row(input_info.name, shape_str, input_info.type)
        
        formatter.print_table(inputs_table)

        # Display output information
        section_header("📤 Model Outputs")
        outputs_table = formatter.create_table()
        outputs_table.add_column("Output", style="bold cyan")
        outputs_table.add_column("Shape", style="green")
        outputs_table.add_column("Type", style="yellow")
        
        for output_info in session.get_outputs():
            shape_str = "x".join(str(dim) if dim > 0 else "?" for dim in output_info.shape)
            outputs_table.add_row(output_info.name, shape_str, output_info.type)
        
        formatter.print_table(outputs_table)

        success("✅ Registry model inspection completed")
        
    except Exception as e:
        error(f"Registry model inspection failed: {e}")
        raise click.Abort()

@cli.command()
def registry_help():
    """Show help for registry integration and workflows."""
    formatter = get_formatter()
    
    section_header("📚 QuickServeML Registry Integration", "Complete ML Workflow Guide")
    
    info("🚀 **Registry-Aware Commands:**")
    registry_commands = [
        ("inspect-registry", "Inspect models from registry with metadata"),
        ("benchmark-registry", "Benchmark registry models and save metrics"),
        ("serve-registry", "Deploy registry models as APIs"),
        ("registry-add", "Add models to registry with versioning"),
        ("registry-list", "List all models in registry"),
        ("registry-get", "Get detailed model information"),
        ("registry-update", "Update model metadata and metrics"),
        ("registry-compare", "Compare model versions"),
        ("registry-export", "Export models from registry"),
        ("registry-delete", "Delete models from registry")
    ]
    
    for cmd, desc in registry_commands:
        info(f"  {cmd}: {desc}")
    
    section_header("💡 **Typical Workflows:**")
    
    workflows = [
        ("📝 Research Workflow:", [
            "1. Train model → registry-add my-model model.onnx",
            "2. Test performance → benchmark-registry my-model --save-metrics",
            "3. Share with team → serve-registry my-model --port 8000",
            "4. Compare versions → registry-compare my-model v1.0.0 v1.0.1"
        ]),
        ("🏭 Production Workflow:", [
            "1. Validate model → registry-update my-model v1.0.1 --status validated",
            "2. Deploy to staging → serve-registry my-model --port 8000",
            "3. Performance test → benchmark-registry my-model --benchmark-runs 1000",
            "4. Promote to production → registry-update my-model v1.0.1 --status production"
        ]),
        ("🔍 Analysis Workflow:", [
            "1. List available models → registry-list --verbose",
            "2. Get model details → registry-get my-model",
            "3. Compare performance → registry-compare my-model v1.0.0 v1.0.1",
            "4. Export for sharing → registry-export my-model shared_model.onnx"
        ])
    ]
    
    for title, steps in workflows:
        info(title)
        for step in steps:
            info(f"  {step}")
    
    section_header("🎯 **Key Benefits:**")
    benefits = [
        "✅ Version Control: Track all model versions automatically",
        "✅ Metadata Management: Store descriptions, tags, metrics",
        "✅ Team Collaboration: Share models by name, not file paths",
        "✅ Performance Tracking: Save and compare benchmark results",
        "✅ Easy Deployment: Deploy any model with one command",
        "✅ Model Lifecycle: Track status from draft to production"
    ]
    
    for benefit in benefits:
        info(benefit)
    
    success("✅ Registry integration ready for use!")

@cli.command()
@click.argument("name")
@click.option("--version", help="Model version (latest if not specified)")
def registry_get(name, version):
    """Get detailed information about a specific model."""
    formatter = get_formatter()
    
    try:
        section_header("🔍 Model Details", f"Information for '{name}'")
        
        from .registry import ModelRegistry
        
        # Initialize registry
        registry = ModelRegistry()
        
        # Get model
        model = registry.get_model(name, version)
        if not model:
            error(f"Model '{name}' not found in registry")
            return
        
        # Display detailed information
        info("📋 Model Information:")
        model_table = formatter.create_table()
        model_table.add_column("Property", style="cyan")
        model_table.add_column("Value", style="white")
        
        model_table.add_row("Name", model.name)
        model_table.add_row("Version", model.version)
        model_table.add_row("Status", model.status)
        model_table.add_row("Description", model.description or "No description")
        model_table.add_row("Author", model.author or "Unknown")
        model_table.add_row("Created", model.created_at)
        model_table.add_row("Updated", model.updated_at)
        model_table.add_row("File Size", f"{model.file_size:,} bytes")
        model_table.add_row("Hash", model.model_hash)
        model_table.add_row("Framework", model.framework)
        
        if model.tags:
            model_table.add_row("Tags", ", ".join(model.tags))
        
        if model.accuracy is not None:
            model_table.add_row("Accuracy", f"{model.accuracy:.4f}")
        
        if model.latency_ms is not None:
            model_table.add_row("Latency", f"{model.latency_ms:.2f} ms")
        
        if model.throughput_rps is not None:
            model_table.add_row("Throughput", f"{model.throughput_rps:.2f} RPS")
        
        if model.input_shape:
            model_table.add_row("Input Shape", str(model.input_shape))
        
        if model.output_shape:
            model_table.add_row("Output Shape", str(model.output_shape))
        
        formatter.print_table(model_table)
        
        # Show deployment history if available
        if model.deployment_history:
            section_header("🚀 Deployment History")
            history_table = formatter.create_table()
            history_table.add_column("Date", style="cyan")
            history_table.add_column("Environment", style="green")
            history_table.add_column("Status", style="yellow")
            
            for deployment in model.deployment_history[-5:]:  # Show last 5
                history_table.add_row(
                    deployment.get("date", "Unknown"),
                    deployment.get("environment", "Unknown"),
                    deployment.get("status", "Unknown")
                )
            
            formatter.print_table(history_table)
        
        success(f"✅ Retrieved model information for '{name}:{model.version}'")
        
    except Exception as e:
        error(f"Failed to get model information: {e}")
        raise click.Abort()

if __name__ == "__main__":
    cli()
