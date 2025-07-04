# QuickServeML

A lightweight CLI tool to inspect and serve ONNX models as local APIs using FastAPI.

---

## 🚀 Features

- Inspect ONNX input/output shapes
- Serve ONNX model as an HTTP endpoint (`/predict`)
- FastAPI backend with ONNX Runtime

## Model Registry

QuickServeML includes a model registry for versioning, metadata, and easy deployment:

- Register a model: `quickserveml registry-add my-model model.onnx`
- List models: `quickserveml registry-list --verbose`
- Serve from registry: `quickserveml serve-registry my-model --version v1.0.0 --port 8000`

See the main README for a full workflow and more details.

---

## 🔧 Installation

```bash
git clone https://github.com/yourusername/quickserveml.git
cd quickserveml
pip install -e .

```
