
import torch
import triton
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../../")))

from indextts.s2mel.modules.bigvgan.alias_free_activation.torch.act import Activation1d
from indextts.s2mel.modules.bigvgan.activations import SnakeBeta
from indextts.s2mel.modules.bigvgan.alias_free_activation.triton.anti_alias_activation_triton import FusedAntiAliasActivation as TritonFused

def benchmark():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    B, C, T = 16, 512, 2048
    x = torch.randn(B, C, T).cuda()
    
    act = SnakeBeta(C, alpha_logscale=True).cuda()
    model_ref = Activation1d(act, fused=False).cuda()
    
    up_filter = model_ref.upsample.filter
    down_filter = model_ref.downsample.lowpass.filter
    alpha = act.alpha
    beta = act.beta
    
    print(f"Benchmarking with Input: {x.shape}")

    # Warmup Triton
    for _ in range(10):
        y = TritonFused.apply(x, up_filter, down_filter, alpha, beta)
        
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        y = TritonFused.apply(x, up_filter, down_filter, alpha, beta)
    torch.cuda.synchronize()
    end = time.time()
    print(f"Triton time: {(end - start)/100 * 1000:.2f} ms")
    
    # Torch
    # Warmup
    for _ in range(10):
        y = model_ref(x)
        
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        y = model_ref(x)
    torch.cuda.synchronize()
    end = time.time()
    print(f"Torch time: {(end - start)/100 * 1000:.2f} ms")
    
    # CUDA
    try:
        from indextts.s2mel.modules.bigvgan.alias_free_activation.torch.act import FusedAntiAliasActivation as CudaFused
        # Warmup
        for _ in range(10):
            y = CudaFused.apply(x, up_filter, down_filter, alpha, beta)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            y = CudaFused.apply(x, up_filter, down_filter, alpha, beta)
        torch.cuda.synchronize()
        end = time.time()
        print(f"CUDA time: {(end - start)/100 * 1000:.2f} ms")
    except ImportError:
        print("CUDA implementation not found.")
    except Exception as e:
        print(f"CUDA benchmark skipped: {e}")

if __name__ == "__main__":
    benchmark()
