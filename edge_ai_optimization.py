# 1. Quantization using PyTorch
import torch
import torch.quantization

def quantize_model(model, calibration_data):
    # Configure quantization
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Prepare for quantization
    model_prepared = torch.quantization.prepare(model)
    
    # Calibrate with sample data
    with torch.no_grad():
        for data in calibration_data:
            model_prepared(data)
    
    # Convert to quantized model
    model_quantized = torch.quantization.convert(model_prepared)
    
    return model_quantized

# 2. Model Pruning
import torch.nn.utils.prune as prune

def prune_model(model, amount=0.3):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return model

# 3. Knowledge Distillation
class DistillationLoss(torch.nn.Module):
    def __init__(self, temperature=3.0):
        super().__init__()
        self.temperature = temperature
        self.kl_div = torch.nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_logits, teacher_logits):
        soft_targets = torch.nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        student_log_probs = torch.nn.functional.log_softmax(student_logits / self.temperature, dim=1)
        return self.kl_div(student_log_probs, soft_targets) * (self.temperature ** 2)

# 4. ONNX Export with Optimization
def optimize_for_edge(model, sample_input, path):
    # Export to ONNX
    torch.onnx.export(model, 
                     sample_input,
                     path,
                     opset_version=13,
                     do_constant_folding=True,
                     input_names=['input'],
                     output_names=['output'],
                     dynamic_axes={'input': {0: 'batch_size'},
                                 'output': {0: 'batch_size'}})
    
    # Optimize with ONNX Runtime
    import onnxruntime as ort
    from onnxruntime.transformers import optimizer
    
    opt_model = optimizer.optimize_model(
        path,
        model_type='bert',  # Change based on your model
        optimizer_level=99,
        use_gpu=True
    )
    opt_model.save_model_to_file(path[:-5] + '_optimized.onnx')

# 5. Model Benchmarking
def benchmark_model(model, test_data, device):
    import time
    
    times = []
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        for data in test_data:
            start = time.time()
            _ = model(data.to(device))
            times.append(time.time() - start)
    
    avg_inference_time = sum(times) / len(times)
    return {
        'avg_inference_time': avg_inference_time,
        'fps': 1.0 / avg_inference_time,
        'total_params': sum(p.numel() for p in model.parameters()),
        'model_size_mb': sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    }

# Usage Example
if __name__ == "__main__":
    # Load your model and data
    model = YourModel()
    data = load_data()
    
    # Quantize
    model_quantized = quantize_model(model, data)
    
    # Prune
    model_pruned = prune_model(model_quantized)
    
    # Export and optimize for edge
    sample_input = torch.randn(1, 3, 224, 224)
    optimize_for_edge(model_pruned, sample_input, 'model.onnx')
    
    # Benchmark
    metrics = benchmark_model(model_pruned, data, 'cuda')
    print(f"Inference speed: {metrics['fps']:.2f} FPS")
    print(f"Model size: {metrics['model_size_mb']:.2f} MB")
