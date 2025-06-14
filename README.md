# QuickServeML 🚀

**One-command ONNX model serving, benchmarking, and visual inspection for researchers and ML engineers.**

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-LNSHRIVAS%2Fquickserveml-blue?logo=github)](https://github.com/LNSHRIVAS/quickserveml)
[![PyPI](https://img.shields.io/badge/PyPI-quickserveml-red.svg)](https://pypi.org/project/quickserveml/)

---

## 🎯 **What is QuickServeML?**

QuickServeML is a comprehensive CLI tool that transforms ONNX models into production-ready APIs with just one command. It provides essential features for ML model analysis, optimization, and deployment:

- **🚀 Instant Deployment**: Deploy any ONNX model as a FastAPI server
- **📊 Performance Benchmarking**: Comprehensive latency, throughput, and resource metrics
- **🔍 Schema Validation**: Automatic input/output validation and OpenAPI documentation
- **📦 Batch Processing**: Optimize throughput with configurable batch sizes
- **🖼️ Visual Inspection**: Interactive model graph visualization with Netron
- **📚 Auto Documentation**: Swagger UI with example requests and responses

---

## ✨ **Key Features**

### 🚀 **One-Command Deployment**
```bash
quickserveml deploy model.onnx --port 8000
```
Instantly creates a production-ready FastAPI server with automatic documentation.

### 📊 **Performance Benchmarking**
```bash
quickserveml benchmark model.onnx --verbose
```
Get comprehensive performance metrics:
- **Latency**: Average, min, max, 95th/99th percentiles
- **Throughput**: Requests per second
- **Resource Usage**: Memory and CPU utilization
- **Warmup Analysis**: Consistent performance measurements

### 🔍 **Schema Generation & Validation**
```bash
quickserveml schema model.onnx --save schema.json
```
- **Automatic OpenAPI generation** for API documentation
- **Runtime input validation** with detailed error messages
- **Example data generation** for testing
- **JSON export** for external tool integration

### 📦 **Batch Processing Optimization**
```bash
quickserveml batch model.onnx --optimize --verbose
```
- **Automatic batch size optimization** for maximum throughput
- **Parallel processing** with configurable worker threads
- **Real data file support** (JSON format)
- **Performance comparison** across different configurations

### 🖼️ **Model Visualization**
```bash
quickserveml deploy model.onnx --visualize
```
Opens Netron for interactive model graph inspection.

---

## 🛠️ **Installation**

### From Source (Recommended)
```bash
git clone https://github.com/LNSHRIVAS/quickserveml.git
cd quickserveml
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -e .
```

### From PyPI (Coming Soon)
```bash
pip install quickserveml
```

---

## 🏁 **Quick Start**

### 1. **Deploy Your First Model**
```bash
# Download a sample model
wget https://github.com/onnx/models/raw/main/vision/classification/mnist/model/mnist-8.onnx

# Deploy with visualization
quickserveml deploy mnist-8.onnx --port 8000 --visualize
```

### 2. **Benchmark Performance**
```bash
quickserveml benchmark mnist-8.onnx --benchmark-runs 100 --verbose
```

### 3. **Generate Schema**
```bash
quickserveml schema mnist-8.onnx --save model_schema.json --verbose
```

### 4. **Optimize Batch Processing**
```bash
quickserveml batch mnist-8.onnx --optimize --verbose
```

### 5. **Access Your API**
- **API**: http://localhost:8000/predict
- **Docs**: http://localhost:8000/docs
- **Netron**: http://localhost:8080

---

## 📖 **Detailed Usage**

### **Model Inspection**
```bash
quickserveml inspect model.onnx
```
Shows model inputs, outputs, and runs a test inference.

### **Performance Benchmarking**
```bash
# Quick benchmark
quickserveml benchmark model.onnx

# Detailed benchmark with custom settings
quickserveml benchmark model.onnx \
  --warmup-runs 20 \
  --benchmark-runs 200 \
  --provider CPUExecutionProvider \
  --verbose
```

**Example Output:**
```
============================================================
📊 BENCHMARK RESULTS
============================================================
Model: mnist-8.onnx
Input Shape: [1, 1, 28, 28]
Output Shape: [1, 10]

⏱️  INFERENCE TIMING
  Average: 0.16 ms
  Min:     0.11 ms
  Max:     0.34 ms
  95th %:  0.28 ms
  99th %:  0.32 ms

🚀 PERFORMANCE
  Throughput: 6131.4 requests/second
  Total benchmark time: 0.0 seconds

💾 RESOURCE USAGE
  Memory usage: 71.6 MB
  CPU usage: 2.4%
============================================================
```

### **Schema Generation**
```bash
# Generate and display schema
quickserveml schema model.onnx --verbose

# Save schema to JSON file
quickserveml schema model.onnx --save model_schema.json
```

**Example Output:**
```
📋 MODEL SCHEMA: mnist-8.onnx
============================================================

📥 INPUTS:
  1. Input3
     Shape: 1 x 1 x 28 x 28
     Type: tensor(float)
     Description: Input tensor: Input3

📤 OUTPUTS:
  1. Plus214_Output_0
     Shape: 1 x 10
     Type: tensor(float)
     Description: Output tensor: Plus214_Output_0
```

### **Batch Processing**
```bash
# Process synthetic batch
quickserveml batch model.onnx --batch-size 50 --verbose

# Process batch from JSON file
quickserveml batch model.onnx --batch-file data.json --parallel

# Find optimal batch size
quickserveml batch model.onnx --optimize --verbose
```

**Example Output:**
```
📊 BATCH SIZE BENCHMARK RESULTS
============================================================
Batch Size   Throughput      Avg Time     Total Time
------------------------------------------------------------
1            993.7           1.01         0.001
4            2058.0          0.24         0.001
8            3159.3          0.33         0.003
16           3017.6          0.34         0.005
32           3040.8          0.35         0.011
------------------------------------------------------------
🎯 Optimal batch size: 8 (3159.3 samples/sec)
```

### **Model Deployment**
```bash
# Basic deployment
quickserveml deploy model.onnx --port 8000

# With visualization
quickserveml deploy model.onnx --port 8000 --visualize
```

---

## 🔧 **API Usage**

Once deployed, your model is available as a REST API:

### **Single Prediction**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Input3": [0.1, 0.2, 0.3, ...]
  }'
```

### **Health Check**
```bash
curl http://localhost:8000/health
```

### **Get Schema**
```bash
curl http://localhost:8000/schema
```

### **Interactive Documentation**
Visit http://localhost:8000/docs for Swagger UI with:
- **Automatic schema validation**
- **Example requests** with sample data
- **Interactive testing** interface
- **Response schemas** and examples

---

## 📊 **Performance Features**

### **Comprehensive Benchmarking**
- **Latency Analysis**: Average, min, max, percentiles
- **Throughput Calculation**: Requests per second
- **Resource Monitoring**: Memory and CPU usage
- **Warmup Runs**: Ensures consistent measurements
- **Configurable Testing**: Customizable run counts

### **Batch Optimization**
- **Automatic Optimization**: Finds optimal batch size
- **Parallel Processing**: Multi-threaded inference
- **Error Handling**: Graceful failure management
- **Performance Comparison**: Across different configurations

### **Schema Validation**
- **Runtime Validation**: Input shape and type checking
- **OpenAPI Integration**: Automatic documentation
- **Example Generation**: Sample data for testing
- **Error Reporting**: Detailed validation messages

---

## 🎯 **Use Cases**

### **For Researchers**
- **Quick Model Testing**: Instant deployment for experimentation
- **Performance Analysis**: Benchmark different model architectures
- **Visual Inspection**: Interactive model graph exploration
- **Schema Understanding**: Detailed input/output analysis

### **For ML Engineers**
- **Production Deployment**: One-command API creation
- **Performance Optimization**: Batch size and resource tuning
- **API Documentation**: Automatic OpenAPI generation
- **Input Validation**: Runtime error prevention

### **For DevOps Teams**
- **Health Monitoring**: Built-in health check endpoints
- **Resource Tracking**: Memory and CPU utilization
- **Performance Metrics**: Latency and throughput monitoring
- **Easy Integration**: Standard REST API interface

---

## 🧪 **Testing**

Run the comprehensive demo to test all features:

```bash
python examples/comprehensive_demo.py mnist-8.onnx
```

This will demonstrate:
- Model inspection
- Schema generation and validation
- Performance benchmarking
- Batch processing optimization
- API integration features

---

## 📁 **Project Structure**

```
quickserveml/
├── quickserveml/
│   ├── __init__.py
│   ├── cli.py              # Main CLI interface
│   ├── infer.py            # Model inspection
│   ├── benchmark.py        # Performance benchmarking
│   ├── schema.py           # Schema generation & validation
│   ├── batch.py            # Batch processing
│   └── serve.py            # FastAPI server
├── templates/
│   └── serve_template.py.jinja  # Server template
├── examples/
│   └── comprehensive_demo.py    # Feature demonstration
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md).

### **Development Setup**
```bash
git clone https://github.com/LNSHRIVAS/quickserveml.git
cd quickserveml
python -m venv venv
source venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

### **Running Tests**
```bash
# Test all features
python examples/comprehensive_demo.py mnist-8.onnx

# Test individual commands
quickserveml inspect mnist-8.onnx
quickserveml benchmark mnist-8.onnx --verbose
quickserveml schema mnist-8.onnx --verbose
quickserveml batch mnist-8.onnx --optimize
```

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 **Author**

**Lakshminarayan Shrivas**  
- GitHub: [@LNSHRIVAS](https://github.com/LNSHRIVAS)
- Email: [lakshminarayanshrivas7@gmail.com]

---

## 🙏 **Acknowledgments**

- [ONNX Runtime](https://github.com/microsoft/onnxruntime) for model inference
- [FastAPI](https://fastapi.tiangolo.com/) for API framework
- [Netron](https://github.com/lutzroeder/netron) for model visualization
- [Click](https://click.palletsprojects.com/) for CLI framework

---

## ⭐ **Star this repo if you find it useful!**

[![GitHub stars](https://img.shields.io/github/stars/LNSHRIVAS/quickserveml?style=social)](https://github.com/LNSHRIVAS/quickserveml/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/LNSHRIVAS/quickserveml?style=social)](https://github.com/LNSHRIVAS/quickserveml/network)
[![GitHub issues](https://img.shields.io/github/issues/LNSHRIVAS/quickserveml)](https://github.com/LNSHRIVAS/quickserveml/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/LNSHRIVAS/quickserveml)](https://github.com/LNSHRIVAS/quickserveml/pulls)
