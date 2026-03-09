uv run --with accelerate --with onnx --with onnxruntime --with sentence-transformers --with 'optimum[onnx]' optimum-cli export onnx --model sentence-transformers/all-MiniLM-L6-v2 onnx_model/
