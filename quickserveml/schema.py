# quickserveml/schema.py

import json
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, validator
import onnx
from onnxruntime import InferenceSession


class ONNXInputSchema(BaseModel):
    """Schema for ONNX model input validation"""
    name: str
    shape: List[Union[int, str]]
    dtype: str
    description: Optional[str] = None
    
    @validator('shape')
    def validate_shape(cls, v):
        """Validate that shape contains valid dimensions"""
        for dim in v:
            if not (isinstance(dim, int) or dim == "?"):
                raise ValueError(f"Invalid dimension: {dim}")
        return v


class ONNXOutputSchema(BaseModel):
    """Schema for ONNX model output validation"""
    name: str
    shape: List[Union[int, str]]
    dtype: str
    description: Optional[str] = None


class ModelSchema:
    """Complete schema for an ONNX model"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.session = None
        self.input_schemas = []
        self.output_schemas = []
        self._load_model()
    
    def _load_model(self):
        """Load ONNX model and extract schema information"""
        self.model = onnx.load(self.model_path)
        self.session = InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        
        # Extract input schemas
        for input_meta in self.session.get_inputs():
            shape = [d if isinstance(d, int) and d > 0 else "?" for d in input_meta.shape]
            schema = ONNXInputSchema(
                name=input_meta.name,
                shape=shape,
                dtype=str(input_meta.type),
                description=f"Input tensor: {input_meta.name}"
            )
            self.input_schemas.append(schema)
        
        # Extract output schemas
        for output_meta in self.session.get_outputs():
            shape = [d if isinstance(d, int) and d > 0 else "?" for d in output_meta.shape]
            schema = ONNXOutputSchema(
                name=output_meta.name,
                shape=shape,
                dtype=str(output_meta.type),
                description=f"Output tensor: {output_meta.name}"
            )
            self.output_schemas.append(schema)
    
    def get_openapi_schema(self) -> Dict[str, Any]:
        """Generate OpenAPI schema for FastAPI integration"""
        schema = {
            "openapi": "3.0.0",
            "info": {
                "title": f"ONNX Model API - {self.model_path}",
                "description": f"API for ONNX model: {self.model_path}",
                "version": "1.0.0"
            },
            "paths": {
                "/predict": {
                    "post": {
                        "summary": "Run inference on the model",
                        "description": "Submit input data and get model predictions",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": self._generate_request_schema()
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "Successful prediction",
                                "content": {
                                    "application/json": {
                                        "schema": self._generate_response_schema()
                                    }
                                }
                            },
                            "400": {
                                "description": "Invalid input data"
                            },
                            "500": {
                                "description": "Model inference error"
                            }
                        }
                    }
                },
                "/health": {
                    "get": {
                        "summary": "Health check",
                        "description": "Check if the model is ready",
                        "responses": {
                            "200": {
                                "description": "Model is healthy",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "status": {"type": "string"},
                                                "model": {"type": "string"},
                                                "inputs": {"type": "array", "items": {"type": "string"}},
                                                "outputs": {"type": "array", "items": {"type": "string"}}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "/schema": {
                    "get": {
                        "summary": "Get model schema",
                        "description": "Get detailed input/output schema information",
                        "responses": {
                            "200": {
                                "description": "Model schema",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "inputs": {"type": "array"},
                                                "outputs": {"type": "array"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return schema
    
    def _generate_request_schema(self) -> Dict[str, Any]:
        """Generate request body schema for OpenAPI"""
        properties = {}
        required = []
        
        for input_schema in self.input_schemas:
            # Convert ONNX shape to JSON schema
            shape_desc = " x ".join(str(d) for d in input_schema.shape)
            
            properties[input_schema.name] = {
                "type": "array",
                "description": f"Input tensor {input_schema.name} with shape {shape_desc}",
                "items": {"type": "number"},
                "example": self._generate_example_data(input_schema.shape)
            }
            required.append(input_schema.name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
    
    def _generate_response_schema(self) -> Dict[str, Any]:
        """Generate response body schema for OpenAPI"""
        properties = {}
        
        for output_schema in self.output_schemas:
            shape_desc = " x ".join(str(d) for d in output_schema.shape)
            
            properties[output_schema.name] = {
                "type": "array",
                "description": f"Output tensor {output_schema.name} with shape {shape_desc}",
                "items": {"type": "number"}
            }
        
        return {
            "type": "object",
            "properties": properties
        }
    
    def _generate_example_data(self, shape: List[Union[int, str]]) -> List[float]:
        """Generate example input data for the schema"""
        # Convert shape to actual dimensions for example generation
        actual_shape = []
        for dim in shape:
            if dim == "?" or dim <= 0:
                actual_shape.append(1)  # Use 1 for dynamic dimensions
            else:
                actual_shape.append(dim)
        
        # Generate small example data
        if len(actual_shape) == 0:
            return [0.0]
        
        # Limit example size for readability
        max_size = 10
        for i, dim in enumerate(actual_shape):
            if dim > max_size:
                actual_shape[i] = max_size
        
        example_array = np.random.randn(*actual_shape).astype(np.float32)
        return example_array.flatten().tolist()
    
    def validate_input(self, input_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Validate and convert input data to numpy arrays"""
        validated_inputs = {}
        
        for input_schema in self.input_schemas:
            if input_schema.name not in input_data:
                raise ValueError(f"Missing required input: {input_schema.name}")
            
            input_value = input_data[input_schema.name]
            
            # Convert to numpy array
            if isinstance(input_value, list):
                input_array = np.array(input_value, dtype=np.float32)
            elif isinstance(input_value, np.ndarray):
                input_array = input_value.astype(np.float32)
            else:
                raise ValueError(f"Input {input_schema.name} must be a list or numpy array")
            
            # Validate shape (ignoring dynamic dimensions)
            expected_shape = input_schema.shape
            actual_shape = input_array.shape
            
            if len(expected_shape) != len(actual_shape):
                raise ValueError(f"Input {input_schema.name} has wrong number of dimensions. "
                               f"Expected {len(expected_shape)}, got {len(actual_shape)}")
            
            # Check fixed dimensions
            for i, (expected_dim, actual_dim) in enumerate(zip(expected_shape, actual_shape)):
                if expected_dim != "?" and expected_dim > 0 and expected_dim != actual_dim:
                    raise ValueError(f"Input {input_schema.name} dimension {i} mismatch. "
                                   f"Expected {expected_dim}, got {actual_dim}")
            
            validated_inputs[input_schema.name] = input_array
        
        return validated_inputs
    
    def format_output(self, output_data: List[np.ndarray]) -> Dict[str, List[float]]:
        """Format output data for JSON response"""
        formatted_outputs = {}
        
        for i, output_schema in enumerate(self.output_schemas):
            if i < len(output_data):
                output_array = output_data[i]
                formatted_outputs[output_schema.name] = output_array.flatten().tolist()
        
        return formatted_outputs
    
    def print_schema(self):
        """Print human-readable schema information"""
        print(f"\n📋 MODEL SCHEMA: {self.model_path}")
        print("=" * 60)
        
        print("\n📥 INPUTS:")
        for i, input_schema in enumerate(self.input_schemas):
            shape_str = " x ".join(str(d) for d in input_schema.shape)
            print(f"  {i+1}. {input_schema.name}")
            print(f"     Shape: {shape_str}")
            print(f"     Type: {input_schema.dtype}")
            if input_schema.description:
                print(f"     Description: {input_schema.description}")
            print()
        
        print("📤 OUTPUTS:")
        for i, output_schema in enumerate(self.output_schemas):
            shape_str = " x ".join(str(d) for d in output_schema.shape)
            print(f"  {i+1}. {output_schema.name}")
            print(f"     Shape: {shape_str}")
            print(f"     Type: {output_schema.dtype}")
            if output_schema.description:
                print(f"     Description: {output_schema.description}")
            print()
    
    def save_schema(self, output_path: str):
        """Save schema to JSON file"""
        schema_data = {
            "model_path": self.model_path,
            "inputs": [schema.dict() for schema in self.input_schemas],
            "outputs": [schema.dict() for schema in self.output_schemas],
            "openapi_schema": self.get_openapi_schema()
        }
        
        with open(output_path, 'w') as f:
            json.dump(schema_data, f, indent=2)
        
        print(f"✔ Schema saved to {output_path}")


def generate_schema(model_path: str, save_path: Optional[str] = None) -> ModelSchema:
    """
    Generate schema for an ONNX model.
    
    Args:
        model_path: Path to the ONNX model
        save_path: Optional path to save schema JSON file
    
    Returns:
        ModelSchema object with complete schema information
    """
    schema = ModelSchema(model_path)
    
    if save_path:
        schema.save_schema(save_path)
    
    return schema 