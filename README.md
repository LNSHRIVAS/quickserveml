# QuickServeML

**One-command ONNX model serving and visual inspection for researchers and ML engineers.**

[![GitHub](https://img.shields.io/badge/GitHub-LNSHRIVAS%2Fquickserveml-blue?logo=github)](https://github.com/LNSHRIVAS/quickserveml)

---

## 🚀 Features

- **One-command ONNX model serving**: Instantly deploy any ONNX model as a FastAPI server.
- **Visual model inspection**: Launches Netron for interactive model graph visualization.
- **Automatic API docs**: Swagger UI at `/docs` for easy testing.
- **Cross-platform**: Works on Windows, Linux, and Mac.
- **No code required**: Just your model and one command.

---

## 📦 Installation

```bash
git clone https://github.com/LNSHRIVAS/quickserveml.git
cd quickserveml
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -e .
```

---

## 🏁 Quick Start

```bash
quickserveml deploy path/to/model.onnx --port 9000 --visualize --verbose
```

- API: [http://127.0.0.1:9000/predict](http://127.0.0.1:9000/predict)
- Docs: [http://127.0.0.1:9000/docs](http://127.0.0.1:9000/docs)
- Netron: [http://localhost:8080](http://localhost:8080)

---

## 🛠️ CLI Usage

```bash
quickserveml --help
quickserveml inspect path/to/model.onnx --verbose --schema
quickserveml deploy path/to/model.onnx --port 9000 --visualize
```

---

## 📚 API Docs

Once deployed, visit [http://127.0.0.1:9000/docs](http://127.0.0.1:9000/docs) for interactive Swagger UI.

---

## 🖼️ Model Visualization

With the `--visualize` flag, Netron will open in your browser at [http://localhost:8080](http://localhost:8080) for interactive ONNX model graph inspection.

---

## 🧑‍💻 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 🙋 FAQ

- **Is my model data private?**  
  Yes, everything runs locally. No data is sent to the cloud.

- **What models are supported?**  
  Any ONNX model.

- **How do I visualize my model?**  
  Use the `--visualize` flag or open Netron at [http://localhost:8080](http://localhost:8080).

- **How do I contribute?**  
  Fork the repo, create a branch, and open a pull request!

---

## 👤 Author

Lakshminarayan Shrivas  
[https://github.com/LNSHRIVAS](https://github.com/LNSHRIVAS)

---

## ⭐️ Star this repo if you find it useful! 