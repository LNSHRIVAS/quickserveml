# quickserveml/infer.py

import click
from pathlib import Path
import json
from typing import Dict, List, Any

def inspect_onnx(path: str, verbose: bool = False, export_json: str = None):
    """
    Load an ONNX model, infer shapes, and print detailed I/O information.
    
    Args:
        path: Path to the ONNX model file
        verbose: Enable detailed layer information
        export_json: Path to export detailed model info as JSON
    """
    try:
        import onnx
        from onnx import shape_inference
        from onnxruntime import InferenceSession
        import numpy as np

        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        click.echo(f"Loading model from {model_path}")
        
        # Load the ONNX model
        model = onnx.load(str(model_path))
        click.echo(f"Model loaded successfully")
        
        # Collect model information
        model_info = {
            "file_path": str(model_path),
            "file_size_mb": model_path.stat().st_size / (1024*1024),
            "ir_version": model.ir_version,
            "producer_name": model.producer_name,
            "producer_version": model.producer_version,
            "model_version": model.model_version,
            "domain": model.domain,
            "doc_string": model.doc_string,
            "graph": {}
        }
        
        # Get model metadata
        if model.ir_version:
            click.echo(f"IR Version: {model.ir_version}")
        if model.producer_name:
            click.echo(f"Producer: {model.producer_name}")
        if model.producer_version:
            click.echo(f"Producer Version: {model.producer_version}")
        if model.domain:
            click.echo(f"Domain: {model.domain}")
        
        # Try shape inference
        try:
            model = shape_inference.infer_shapes(model)
            click.echo("Shape inference completed")
        except Exception as e:
            click.echo(f"Shape inference failed: {e}")
        
        # Analyze graph structure
        graph = model.graph
        model_info["graph"]["name"] = graph.name
        model_info["graph"]["nodes_count"] = len(graph.node)
        
        # Display input information
        click.echo("\nInputs:")
        inputs_info = []
        for i, inp in enumerate(graph.input):
            dims = [d.dim_value if d.dim_value > 0 else "?" for d in inp.type.tensor_type.shape.dim]
            elem_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(inp.type.tensor_type.elem_type, "unknown")
            
            input_info = {
                "name": inp.name,
                "shape": dims,
                "type": str(elem_type),
                "index": i
            }
            inputs_info.append(input_info)
            
            click.echo(f"  {i+1}. {inp.name}: shape = {dims}")
            click.echo(f"     Type: {elem_type}")
        
        model_info["graph"]["inputs"] = inputs_info
        
        # Display output information
        click.echo("\nOutputs:")
        outputs_info = []
        for i, outp in enumerate(graph.output):
            dims = [d.dim_value if d.dim_value > 0 else "?" for d in outp.type.tensor_type.shape.dim]
            elem_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(outp.type.tensor_type.elem_type, "unknown")
            
            output_info = {
                "name": outp.name,
                "shape": dims,
                "type": str(elem_type),
                "index": i
            }
            outputs_info.append(output_info)
            
            click.echo(f"  {i+1}. {outp.name}: shape = {dims}")
            click.echo(f"     Type: {elem_type}")
        
        model_info["graph"]["outputs"] = outputs_info
        
        # Analyze layers/nodes if verbose
        if verbose:
            click.echo("\nLayer Analysis:")
            layers_info = []
            
            # Count operator types
            op_counts = {}
            
            for i, node in enumerate(graph.node):
                op_type = node.op_type
                op_counts[op_type] = op_counts.get(op_type, 0) + 1
                
                layer_info = {
                    "index": i,
                    "name": node.name,
                    "op_type": op_type,
                    "inputs": list(node.input),
                    "outputs": list(node.output),
                    "attributes": {}
                }
                
                # Extract attributes
                for attr in node.attribute:
                    if attr.type == onnx.AttributeProto.FLOAT:
                        layer_info["attributes"][attr.name] = attr.f
                    elif attr.type == onnx.AttributeProto.INT:
                        layer_info["attributes"][attr.name] = attr.i
                    elif attr.type == onnx.AttributeProto.STRING:
                        layer_info["attributes"][attr.name] = attr.s.decode('utf-8')
                    elif attr.type == onnx.AttributeProto.FLOATS:
                        layer_info["attributes"][attr.name] = list(attr.floats)
                    elif attr.type == onnx.AttributeProto.INTS:
                        layer_info["attributes"][attr.name] = list(attr.ints)
                
                layers_info.append(layer_info)
                
                # Display layer info
                click.echo(f"  {i+1}. {node.name or f'Node_{i}'} ({op_type})")
                click.echo(f"     Inputs: {list(node.input)}")
                click.echo(f"     Outputs: {list(node.output)}")
                
                if layer_info["attributes"]:
                    click.echo(f"     Attributes: {layer_info['attributes']}")
            
            model_info["graph"]["layers"] = layers_info
            model_info["graph"]["operator_counts"] = op_counts
            
            # Show operator summary
            click.echo(f"\nOperator Summary:")
            for op_type, count in sorted(op_counts.items()):
                click.echo(f"  {op_type}: {count}")
        
        # Test inference with ONNX Runtime
        click.echo("\nTesting inference with ONNX Runtime...")
        try:
            session = InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            
            # Get input details
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            
            click.echo(f"ONNX Runtime session created")
            click.echo(f"Available providers: {session.get_providers()}")
            
            # Test with dummy input
            if inputs:
                input_meta = inputs[0]
                input_name = input_meta.name
                input_shape = input_meta.shape
                
                # Create dummy input with proper shape
                dummy_shape = [d if isinstance(d, int) and d > 0 else 1 for d in input_shape]
                dummy_input = np.zeros(dummy_shape, dtype=np.float32)
                
                click.echo(f"Testing with input shape: {dummy_shape}")
                
                # Run inference
                output = session.run(None, {input_name: dummy_input})
                
                click.echo(f"Inference successful!")
                click.echo(f"Output shape: {output[0].shape}")
                
                # Show output details
                for i, (out_meta, out_data) in enumerate(zip(outputs, output)):
                    click.echo(f"  Output {i+1}: {out_meta.name} -> shape {out_data.shape}")
                    
            else:
                click.echo("No inputs found in model")
                
        except Exception as e:
            click.echo(f"ONNX Runtime inference failed: {e}")
        
        # Show model size
        model_size = model_path.stat().st_size
        click.echo(f"\nModel file size: {model_size / (1024*1024):.2f} MB")
        
        # Export to JSON if requested
        if export_json:
            export_path = Path(export_json)
            with open(export_path, 'w') as f:
                json.dump(model_info, f, indent=2, default=str)
            click.echo(f"Model information exported to: {export_path}")
        
        return model_info
        
    except ImportError as e:
        click.echo(f"Missing dependency: {e}", err=True)
        click.echo("Install required packages: pip install onnx onnxruntime", err=True)
        raise
    except Exception as e:
        click.echo(f"Error inspecting model: {e}", err=True)
        raise

def generate_model_schema(model_path: str) -> Dict[str, Any]:
    """
    Generate a JSON schema for the model's input/output format.
    
    Args:
        model_path: Path to the ONNX model file
        
    Returns:
        Dictionary containing the model schema
    """
    try:
        import onnx
        import numpy as np
        from onnxruntime import InferenceSession
        
        model = onnx.load(model_path)
        session = InferenceSession(model_path, providers=["CPUExecutionProvider"])
        
        schema = {
            "model_path": model_path,
            "inputs": [],
            "outputs": [],
            "example_request": {}
        }
        
        # Input schema
        for inp in session.get_inputs():
            input_schema = {
                "name": inp.name,
                "shape": inp.shape,
                "type": str(inp.type),
                "description": f"Input tensor '{inp.name}' with shape {inp.shape}"
            }
            schema["inputs"].append(input_schema)
            
            # Generate example data
            if len(inp.shape) > 0:
                example_shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
                schema["example_request"][inp.name] = {
                    "shape": example_shape,
                    "data": [0.0] * int(np.prod(example_shape))
                }
        
        # Output schema
        for out in session.get_outputs():
            output_schema = {
                "name": out.name,
                "shape": out.shape,
                "type": str(out.type),
                "description": f"Output tensor '{out.name}' with shape {out.shape}"
            }
            schema["outputs"].append(output_schema)
        
        return schema
        
    except Exception as e:
        raise Exception(f"Failed to generate schema: {e}")

