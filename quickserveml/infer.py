# quickserveml/infer.py

def inspect_onnx(path: str):
    """
    Load an ONNX model, infer shapes, and print I/O information.
    """

    import onnx
    from onnx import shape_inference
    from onnxruntime import InferenceSession
    import numpy as np

    model = onnx.load(path)
    print(f"✔ Loaded model from {path}")

    try:
        model = shape_inference.infer_shapes(model)
        print("✔ Shape inference succeeded")
    except Exception as e:
        print(f"⚠ Shape inference failed: {e}")

    print("\nInputs:")
    for inp in model.graph.input:
        dims = [d.dim_value if d.dim_value > 0 else "?" for d in inp.type.tensor_type.shape.dim]
        print(f" • {inp.name}: shape = {dims}")

    print("\nOutputs:")
    for outp in model.graph.output:
        dims = [d.dim_value if d.dim_value > 0 else "?" for d in outp.type.tensor_type.shape.dim]
        print(f" • {outp.name}: shape = {dims}")

    sess = InferenceSession(path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in sess.get_inputs()[0].shape]
    dummy = np.zeros(shape, dtype=np.float32)
    out = sess.run(None, {name: dummy})
    print(f"\n✔ Inference ran. Output shape: {out[0].shape}")

