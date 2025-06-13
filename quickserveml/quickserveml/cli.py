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

@click.group()
@click.version_option(version="0.1.0", prog_name="QuickServeML")
def cli():
    """
    QuickServeML - A lightweight CLI tool to inspect and serve ONNX models as local APIs.
    
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
        # Basic inspection
        model_info = inspect_onnx(str(model_path), verbose=verbose, export_json=export_json)
        
        # Generate schema if requested
        if schema:
            click.echo("\n📋 Model Schema:")
            try:
                from quickserveml.infer import generate_model_schema
                schema_info = generate_model_schema(str(model_path))
                
                click.echo("Input Schema:")
                for inp in schema_info["inputs"]:
                    click.echo(f"  - {inp['name']}: {inp['shape']} ({inp['type']})")
                
                click.echo("\nOutput Schema:")
                for out in schema_info["outputs"]:
                    click.echo(f"  - {out['name']}: {out['shape']} ({out['type']})")
                
                click.echo("\nExample Request Format:")
                click.echo(json.dumps(schema_info["example_request"], indent=2))
                
            except Exception as e:
                click.echo(f"⚠️  Failed to generate schema: {e}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
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
        if verbose:
            click.echo(f"🚀 Starting deployment of model: {model_path}")
        
        # Visualize with Netron if requested
        if visualize:
            click.echo("🖼️  Launching Netron to visualize the model in your browser...")
            venv_path = Path(sys.executable).parent
            if platform.system() == "Windows":
                netron_exe = venv_path / "netron.exe"
            else:
                netron_exe = venv_path / "netron"
            if not netron_exe.exists():
                click.echo("🔎 Netron not found in venv. Installing Netron...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "netron"])
            # Launch Netron in the background
            subprocess.Popen([str(netron_exe), str(model_path)])
        
        # Import here to avoid issues if onnxruntime is not available
        from onnxruntime import InferenceSession
        
        # Load model and extract input shape
        if verbose:
            click.echo("📥 Loading ONNX model...")
        
        session = InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_shape = session.get_inputs()[0].shape
        input_shape = [d if isinstance(d, int) and d > 0 else 1 for d in input_shape]
        model_filename = Path(model_path).name
        
        if verbose:
            click.echo(f"✅ Model loaded successfully")
            click.echo(f"📊 Input shape: {input_shape}")
        
        # Copy the model to current dir for the server to access
        shutil.copy(model_path, f"./{model_filename}")
        
        # Get the template path relative to the package
        package_dir = Path(__file__).parent.parent
        template_path = package_dir / "templates" / "serve_template.py.jinja"
        
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found at {template_path}")
        
        # Load and render Jinja template with UTF-8 encoding
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
        
        click.echo(f"✅ FastAPI server generated as serve.py")
        click.echo(f"🌐 Server will be available at http://{host}:{port}")
        click.echo(f"📝 API endpoint: http://{host}:{port}/predict")
        click.echo(f"📚 API docs: http://{host}:{port}/docs")
        click.echo("🚀 Starting server...")
        
        # Run the server with Uvicorn
        cmd = ["uvicorn", "serve:app", "--host", host, "--port", str(port)]
        if reload:
            cmd.append("--reload")
        subprocess.run(cmd)
        
    except ImportError as e:
        click.echo(f"❌ Missing dependency: {e}", err=True)
        click.echo("💡 Try installing with: pip install onnxruntime", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error deploying model: {e}", err=True)
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
        import requests
        import numpy as np
        
        # Create a dummy input (you might want to make this configurable)
        dummy_input = np.zeros((1, 3, 224, 224), dtype=np.float32)
        
        url = f"http://{host}:{port}/predict"
        
        click.echo(f"🧪 Testing API at {url}")
        
        response = requests.post(
            url,
            json={"data": dummy_input.tolist()},
            timeout=10
        )
        
        if response.status_code == 200:
            click.echo("✅ API test successful!")
            click.echo(f"📊 Response: {response.json()}")
        else:
            click.echo(f"❌ API test failed with status {response.status_code}")
            click.echo(f"📄 Response: {response.text}")
            
    except ImportError:
        click.echo("❌ Missing dependency: requests", err=True)
        click.echo("💡 Install with: pip install requests", err=True)
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        click.echo(f"❌ Could not connect to server at {host}:{port}", err=True)
        click.echo("💡 Make sure the server is running with: quickserveml deploy <model>", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error testing API: {e}", err=True)
        sys.exit(1)

if __name__ == "__main__":
    cli()
