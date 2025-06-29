# quickserveml/cli.py

import os
import sys
import subprocess
import shutil
import click
from pathlib import Path
from jinja2 import Template
from quickserveml.infer import inspect_onnx
import json

# Import rich formatting utilities
try:
    from quickserveml.cli_utils import get_formatter, success, info, warning, error, section_header
except ImportError:
    # Fallback for simple print statements if rich is not available
    def success(msg, title=None): click.echo(f"✅ {title + ': ' if title else ''}{msg}")
    def info(msg, title=None): click.echo(f"ℹ️  {title + ': ' if title else ''}{msg}")
    def warning(msg, title=None): click.echo(f"⚠️  {title + ': ' if title else ''}{msg}")
    def error(msg, title=None): click.echo(f"❌ {title + ': ' if title else ''}{msg}")
    def section_header(title, subtitle=None): 
        click.echo(f"\n=== {title} ===")
        if subtitle: click.echo(subtitle)
        click.echo()

@click.group()
@click.version_option(version="0.1.0", prog_name="QuickServeML")
def cli():
    """
    🚀 QuickServeML - Lightning-fast ONNX model deployment
    
    A lightweight CLI tool to inspect and serve ONNX models as local APIs.
    This tool helps you quickly deploy your ONNX models as FastAPI servers
    for testing and inference without writing boilerplate code.
    """
    pass

@cli.command()
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--verbose", "-v", is_flag=True, help="Enable detailed layer information")
@click.option("--export-json", type=click.Path(path_type=Path), help="Export model information to JSON file")
@click.option("--schema", is_flag=True, help="Generate and display model schema")
def inspect(model_path, verbose, export_json, schema):
    """
    Inspect an ONNX model and display detailed information.
    
    This command analyzes the model structure, inputs/outputs, and provides
    detailed layer information when used with --verbose flag.
    """
    try:
        section_header("🔍 Model Inspection", f"Analyzing {model_path}")
        
        # Basic inspection
        info("Loading and analyzing model structure...")
        model_info = inspect_onnx(str(model_path), verbose=verbose, export_json=export_json)
        
        # Generate schema if requested
        if schema:
            section_header("📋 Model Schema")
            try:
                from quickserveml.infer import generate_model_schema
                schema_info = generate_model_schema(str(model_path))
                
                info("Input Schema:")
                for inp in schema_info["inputs"]:
                    click.echo(f"  • {inp['name']}: {inp['shape']} ({inp['type']})")
                
                info("Output Schema:")
                for out in schema_info["outputs"]:
                    click.echo(f"  • {out['name']}: {out['shape']} ({out['type']})")
                
                info("Example Request Format:")
                click.echo(json.dumps(schema_info["example_request"], indent=2))
                
            except Exception as e:
                warning(f"Failed to generate schema: {e}")
        
        success("Model inspection completed successfully")
        
    except Exception as e:
        error(f"Model inspection failed: {e}")
        sys.exit(1)

@cli.command()
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--port", "-p", default=8000, help="Port to run the API on (default: 8000)")
@click.option("--host", "-h", default="127.0.0.1", help="Host to bind the server to (default: 127.0.0.1)")
@click.option("--reload", is_flag=True, help="Enable auto-reload on code changes")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--visualize", is_flag=True, help="Visualize the ONNX model in Netron before serving")
def deploy(model_path, port, host, reload, verbose, visualize):
    """
    Deploy an ONNX model as a FastAPI server.
    
    MODEL_PATH: Path to the ONNX model file (.onnx)
    """
    import platform
    import subprocess
    import sys
    import shutil
    from pathlib import Path
    from jinja2 import Template
    
    try:
        section_header("🚀 Model Deployment", f"Deploying {model_path}")
        
        # Visualize with Netron if requested
        if visualize:
            info("Launching Netron to visualize the model in your browser...")
            venv_path = Path(sys.executable).parent
            if platform.system() == "Windows":
                netron_exe = venv_path / "netron.exe"
            else:
                netron_exe = venv_path / "netron"
            if not netron_exe.exists():
                info("Netron not found in venv. Installing Netron...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "netron"])
            # Launch Netron in the background
            subprocess.Popen([str(netron_exe), str(model_path)])
        
        # Import here to avoid issues if onnxruntime is not available
        from onnxruntime import InferenceSession
        
        # Load model and extract input shape
        info("Loading ONNX model...")
        
        session = InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_shape = session.get_inputs()[0].shape
        input_shape = [d if isinstance(d, int) and d > 0 else 1 for d in input_shape]
        model_filename = Path(model_path).name
        
        success(f"Model loaded successfully")
        if verbose:
            info(f"Input shape: {input_shape}")
        
        # Copy the model to current dir for the server to access
        info(f"Copying model to working directory: {model_filename}")
        shutil.copy(model_path, f"./{model_filename}")
        
        # Get the template path relative to the package
        package_dir = Path(__file__).parent.parent
        template_path = package_dir / "templates" / "serve_template.py.jinja"
        
        if not template_path.exists():
            error(f"Template not found at {template_path}")
            raise FileNotFoundError(f"Template not found at {template_path}")
        
        # Load and render Jinja template with UTF-8 encoding
        info("Generating FastAPI server code...")
        with open(template_path, 'r', encoding='utf-8') as f:
            template = Template(f.read())
        
        rendered = template.render(
            model_filename=model_filename, 
            input_shape=input_shape,
            host=host,
            port=port
        )
        
        # Write the generated FastAPI server
        with open("serve.py", "w", encoding='utf-8') as f:
            f.write(rendered)
        
        success("FastAPI server generated as serve.py")
        
        # Display server information  
        section_header("🌐 Server Information")
        info(f"Server URL: http://{host}:{port}")
        info(f"API endpoint: http://{host}:{port}/predict")
        info(f"API documentation: http://{host}:{port}/docs")
        
        info("Starting server with Uvicorn...")
        
        # Run the server with Uvicorn
        cmd = ["uvicorn", "serve:app", "--host", host, "--port", str(port)]
        if reload:
            cmd.append("--reload")
        subprocess.run(cmd)
        
    except ImportError as e:
        error(f"Missing dependency: {e}")
        info("Try installing with: pip install onnxruntime")
        sys.exit(1)
    except Exception as e:
        error(f"Error deploying model: {e}")
        sys.exit(1)

@cli.command()
@click.option("--port", "-p", default=8000, help="Port to test (default: 8000)")
@click.option("--host", "-h", default="127.0.0.1", help="Host to test (default: 127.0.0.1)")
def test(port, host):
    """
    Test the deployed API with a sample request.
    
    This command sends a test request to the running API server.
    """
    try:
        section_header("🧪 API Testing", f"Testing server at {host}:{port}")
        
        import requests
        import numpy as np
        
        # Create a dummy input (you might want to make this configurable)
        dummy_input = np.zeros((1, 3, 224, 224), dtype=np.float32)
        
        url = f"http://{host}:{port}/predict"
        
        info(f"Sending test request to {url}")
        
        response = requests.post(
            url,
            json={"data": dummy_input.tolist()},
            timeout=10
        )
        
        if response.status_code == 200:
            success("API test successful!")
            info(f"Response: {response.json()}")
        else:
            error(f"API test failed with status {response.status_code}")
            warning(f"Response: {response.text}")
            
    except ImportError:
        error("Missing dependency: requests")
        info("Install with: pip install requests")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        error(f"Could not connect to server at {host}:{port}")
        info("Make sure the server is running with: quickserveml deploy <model>")
        sys.exit(1)
    except Exception as e:
        error(f"Error testing API: {e}")
        sys.exit(1)

if __name__ == "__main__":
    cli()
