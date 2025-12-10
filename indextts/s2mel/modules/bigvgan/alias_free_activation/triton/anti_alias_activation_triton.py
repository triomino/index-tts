
import torch
import triton
import triton.language as tl

@triton.jit
def anti_alias_activation_kernel(
    dst_ptr, src_ptr,
    up_ftr_ptr, down_ftr_ptr,
    alpha_ptr, beta_ptr,
    batch_size, channels, seq_len,
    BLOCK_SIZE: tl.constexpr,
    FILTER_SIZE: tl.constexpr,
    UPSAMPLE_PAD: tl.constexpr
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)

    # Base pointers for this sequence
    seq_offset = (pid_z * channels + pid_y) * seq_len
    src_base = src_ptr + seq_offset
    dst_base = dst_ptr + seq_offset

    # Load Alpha/Beta
    # alpha/beta shape [channels]
    alpha_val = tl.load(alpha_ptr + pid_y)
    beta_val = tl.load(beta_ptr + pid_y)
    
    # Apply exp as in CUDA
    alpha_val = tl.exp(alpha_val)
    beta_val = tl.exp(beta_val)

    # Block offsets
    block_start = pid_x * BLOCK_SIZE
    
    # Output offsets
    out_offsets = tl.arange(0, BLOCK_SIZE)
    global_out_indices = block_start + out_offsets
    
    # Accumulator for output
    out_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Loop over downsampling filter
    for f in range(FILTER_SIZE):
        # We want to compute Activation(Upsample) at index corresponding to f
        # Virtual index G = 2*global_out + f + 1
        # We compute this for all i in BLOCK simultaneously.
        
        # Upsample convolution for this f
        up_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        for u in range(FILTER_SIZE):
            # Z = G + u - 12
            # Z = 2 * global_out_indices + f + 1 + u - 12
            # Z = 2 * global_out_indices + (f + u - 11)
            
            offset_const = f + u - 11
            Z = 2 * global_out_indices + offset_const
            
            mask_even = (Z % 2) == 0
            p = Z // 2
            
            # Load src
            p_clamped = tl.maximum(0, tl.minimum(seq_len - 1, p))
            val = tl.load(src_base + p_clamped, mask=mask_even, other=0.0)
            
            # Valid mask for padding
            valid_mask = (p >= -UPSAMPLE_PAD) & (p < seq_len + UPSAMPLE_PAD)
            
            term = tl.where(mask_even & valid_mask, 2.0 * val, 0.0)
            
            # Load up_filter coefficient
            coef_up = tl.load(up_ftr_ptr + u)
            up_acc += term * coef_up
            
        # Activation
        no_div_by_zero = 1e-9
        sin_val = tl.sin(up_acc * alpha_val)
        act_val = up_acc + (1.0 / (beta_val + no_div_by_zero)) * sin_val * sin_val
        
        # Downsample accumulation
        coef_down = tl.load(down_ftr_ptr + f)
        out_acc += act_val * coef_down

    # Store output
    mask_out = global_out_indices < seq_len
    tl.store(dst_base + global_out_indices, out_acc, mask=mask_out)

def anti_alias_activation_forward(x, up_filter, down_filter, alpha, beta):
    # x: [B, C, T]
    batch, channels, seq_len = x.shape
    
    # Output has same shape
    y = torch.empty_like(x)
    
    # Grid
    BLOCK_SIZE = 64
    grid = (triton.cdiv(seq_len, BLOCK_SIZE), channels, batch)
    
    # Constants
    FILTER_SIZE = 12
    UPSAMPLE_PAD = 5
    
    anti_alias_activation_kernel[grid](
        y, x,
        up_filter, down_filter,
        alpha, beta,
        batch, channels, seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
        FILTER_SIZE=FILTER_SIZE,
        UPSAMPLE_PAD=UPSAMPLE_PAD
    )
    
    return y

class FusedAntiAliasActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, up_ftr, down_ftr, alpha, beta):
        return anti_alias_activation_forward(inputs, up_ftr, down_ftr, alpha, beta)

    @staticmethod
    def backward(ctx, output_grads):
        raise NotImplementedError
