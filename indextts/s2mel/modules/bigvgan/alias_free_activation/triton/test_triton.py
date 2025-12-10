
import torch
import sys
import os

# Add project root to path
# Current: indextts/s2mel/modules/bigvgan/alias_free_activation/triton
# Root: d:/opensource/index-tts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../../")))

from indextts.s2mel.modules.bigvgan.alias_free_activation.torch.act import Activation1d
from indextts.s2mel.modules.bigvgan.activations import SnakeBeta
from indextts.s2mel.modules.bigvgan.alias_free_activation.triton.anti_alias_activation_triton import FusedAntiAliasActivation as TritonFused

def test_accuracy():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    torch.manual_seed(0)
    B, C, T = 2, 4, 128
    x = torch.randn(B, C, T).cuda()
    
    # Create model
    # SnakeBeta with logscale=True
    act = SnakeBeta(C, alpha_logscale=True).cuda()
    model_ref = Activation1d(act).cuda()
    
    # Run reference
    with torch.no_grad():
        y_ref = model_ref(x)
        
    # Run Triton
    up_filter = model_ref.upsample.filter
    down_filter = model_ref.downsample.lowpass.filter
    alpha = act.alpha
    beta = act.beta
    
    with torch.no_grad():
        y_triton = TritonFused.apply(x, up_filter, down_filter, alpha, beta)
        
    # Compare
    diff = (y_ref - y_triton).abs().max()
    print(f"Max difference (Triton vs Torch): {diff.item()}")
    
    if diff < 1e-4:
        print("Triton vs Torch Test Passed!")
    else:
        print("Triton vs Torch Test Failed!")
        
    # Compare with CUDA
    try:
        from indextts.s2mel.modules.bigvgan.alias_free_activation.torch.act import FusedAntiAliasActivation as CudaFused
        with torch.no_grad():
            y_cuda = CudaFused.apply(x, up_filter, down_filter, alpha, beta)
        diff_cuda = (y_cuda - y_triton).abs().max()
        print(f"Max difference (Triton vs CUDA): {diff_cuda.item()}")
        
        if diff_cuda < 1e-4:
             print("Triton vs CUDA Test Passed!")
        else:
             print("Triton vs CUDA Test Failed!")

    except ImportError:
        print("CUDA implementation not found or failed to load.")
    except Exception as e:
        print(f"CUDA test skipped: {e}")

if __name__ == "__main__":
    test_accuracy()
