# quickserveml/cli.py

import os
import subprocess
import shutil
import click
from jinja2 import Template
from quickserveml.infer import inspect_onnx

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
@click.option("--port", default=8000, help="Port to run the API on")
def deploy(model_path, port):
    """Deploy an ONNX model as a FastAPI server."""
    from onnxruntime import InferenceSession

    # Load model and extract input shape
    session = InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_shape = session.get_inputs()[0].shape
    input_shape = [d if isinstance(d, int) and d > 0 else 1 for d in input_shape]
    model_filename = os.path.basename(model_path)

    # Copy the model to current dir for the server to access
    shutil.copy(model_path, f"./{model_filename}")

    # Load and render Jinja template
    template_path = os.path.join("templates", "serve_template.py.jinja")
    with open(template_path) as f:
        template = Template(f.read())

    rendered = template.render(model_filename=model_filename, input_shape=input_shape)

    # Write the generated FastAPI server
    with open("serve.py", "w") as f:
        f.write(rendered)

    print(f"✔ FastAPI server generated as serve.py")
    print(f"🚀 Running server at http://localhost:{port}/predict ...")

    # Run the server with Uvicorn
    subprocess.run(["uvicorn", "serve:app", "--reload", "--port", str(port)])

if __name__ == "__main__":
    cli()
