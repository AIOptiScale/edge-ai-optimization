# Edge AI Optimization Tools

A collection of Python utilities for optimizing and deploying AI models on edge devices.

## Features

- Model quantization using PyTorch
- Weight pruning for model compression
- Knowledge distillation framework
- ONNX export with optimization
- Performance benchmarking tools

## Requirements

```
torch>=1.9.0
onnx>=1.10.0
onnxruntime>=1.8.0
numpy>=1.19.0
```

## Installation

```bash
git clone https://github.com/yourusername/edge-ai-optimization
cd edge-ai-optimization
pip install -r requirements.txt
```

## Usage

### Quantization

Reduces model precision to decrease size while maintaining accuracy:

```python
from edge_optimization import quantize_model

quantized_model = quantize_model(your_model, calibration_data)
```

### Pruning

Removes unnecessary weights to reduce model size:

```python
from edge_optimization import prune_model

pruned_model = prune_model(your_model, amount=0.3)  # Removes 30% of weights
```

### Knowledge Distillation

Trains a smaller student model using a larger teacher model:

```python
distillation_loss = DistillationLoss(temperature=3.0)
loss = distillation_loss(student_logits, teacher_logits)
```

### ONNX Export

Exports and optimizes models for edge deployment:

```python
from edge_optimization import optimize_for_edge

optimize_for_edge(model, sample_input, "model.onnx")
```

### Benchmarking

Measures model performance metrics:

```python
metrics = benchmark_model(model, test_data, device='cuda')
print(f"FPS: {metrics['fps']}")
```

## Performance Considerations

- Quantization typically reduces model size by 75% with minimal accuracy loss
- Pruning can reduce model size by 30-50% depending on architecture
- ONNX optimization can improve inference speed by 20-40%
- Consider batch size and input dimensions for optimal performance

## Known Limitations

- Quantization requires calibration data for best results
- Pruning may affect model accuracy on complex tasks
- ONNX optimization is model-architecture dependent
- GPU required for maximum performance benefits

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{edge_ai_optimization,
  author = {dewitt4},
  title = {Edge AI Optimization Tools},
  year = {2024},
  url = {https://github.com/dewitt4/edge-ai-optimization}
}
```
