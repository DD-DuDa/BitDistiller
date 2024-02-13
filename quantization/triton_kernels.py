# Copyright (C) 2023 - Vedant Roy
import torch
import triton
import triton.language as tl


# Auottuner configs:
# https://github.com/fpgaminer/GPTQ-triton/blob/main/src/gptq_triton/quant_linear.py
# https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#sphx-glr-getting-started-tutorials-03-matrix-multiplication-py
# Autotuner source:
# https://github.com/openai/triton/blob/main/python/triton/runtime/autotuner.py
# Custom autotuner:
# https://github.com/fpgaminer/GPTQ-triton/blob/main/src/gptq_triton/custom_autotune.py
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
    ],
    key=['M', 'N', 'K'],
    warmup=0,
)
@triton.jit
def quant_matmul_kernel(
    # Pointers to matrices
    a_ptr, qw_ptr, c_ptr, scales_ptr, zeros_ptr,
    # Matrix dimensions
    M, N, K, 
    pack_num, w_bit,
    # Quantization parameters
    group_size, offset,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, 
):
    """
    Kernel for computing the matmul C = A x qw

    a: (M, K)
    qw: (K // pack_num, N)
    scales: (K // group_size, N)
    qzeros: (K // group_size // pack_num, N)
    """

    stride_zeros_k = N
    stride_scales_k = N
    stride_a_m = K
    stride_qw_k = N

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)  # (K,)
    qw_shifter = (offs_k % pack_num) * w_bit

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_offs = (k * BLOCK_SIZE_K) + (offs_am[:, None] * stride_a_m + offs_k[None, :])  # (M, K)
        a = tl.load(a_ptr + a_offs)

        # load weight
        qw_offs = (((k * BLOCK_SIZE_K) + offs_k[:, None]) // pack_num) * stride_qw_k + offs_bn[
            None, :
        ]  # (K, N)
        qw_packed = tl.load(qw_ptr + qw_offs)  # (K, N)
        qw_unpacked = (qw_packed >> qw_shifter[:, None]) & offset

        # load sacle
        k_iters_per_quant_group = group_size // BLOCK_SIZE_K
        grp_idx = k // k_iters_per_quant_group
        col_offs = offs_bn
        scales = tl.load(scales_ptr + (stride_scales_k * grp_idx) + col_offs)  # (N,)

        # load zeros
        packed_zeros = tl.load(
            zeros_ptr + stride_zeros_k * (grp_idx // pack_num) + col_offs
        )  # (N,)
        unpacked_zeros = (packed_zeros >> ((grp_idx % pack_num) * w_bit)) & offset

        dequantized = scales[None, :].to(tl.float32) * (
            qw_unpacked.to(tl.float32) - unpacked_zeros[None, :].to(tl.float32)
        )
        accumulator += tl.dot(a, dequantized.to(tl.float16))
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    stride_cm = N
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def quant_matmul_v2(a, qw, qzeros, scales, *, M, N, K, pack_num, group_size, w_bit, offset):
    c = torch.empty((M, N), dtype=torch.float16, device=a.device)
    assert qw.shape == (K // pack_num, N)
    # assert qzeros.shape == (K // group_size // pack_num, N)
    # assert scales.shape == (K // group_size, N)
    assert all(x.is_contiguous() for x in [a, qw, c, qzeros, scales])
    # BLOCK_SIZE_K has possible values of 32, 64
    # group_size, K must be divisible by BLOCK_SIZE_K
    assert group_size % 64 == 0, f"group_size {group_size} is not a multiple of 64"
    assert K % 64 == 0, f"K {K} is not a multiple of 64"
    # BLOCK_SIZE_N has possible values of 32, 64, 128, 256
    # N must be divisible by BLOCK_SIZE_N
    assert N % 256 == 0, f"N {N} is not a multiple of 256"

    grid_1d = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    quant_matmul_kernel[grid_1d](
        a_ptr=a,
        qw_ptr=qw,
        c_ptr=c,
        scales_ptr=scales,
        zeros_ptr=qzeros,
        M=M,
        N=N,
        K=K,
        pack_num=pack_num,
        w_bit=w_bit,
        group_size=group_size,
        offset=offset
    )
    return c



def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("c_ptr")
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("c_ptr")
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("c_ptr")
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 2048}, num_warps=8, pre_hook=init_to_zero("c_ptr")
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("c_ptr")
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("c_ptr")
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("c_ptr")
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 2048}, num_warps=8, pre_hook=init_to_zero("c_ptr")
        ),
    ],
    key=["CACHE_KEY_M", "CACHE_KEY_N", "BATCHSIZE"],
)
@triton.heuristics(
    {
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def transposed_gemv_atomicadd_kernel(
    # Pointers to matrices
    a_ptr, qw_ptr, c_ptr, scales_ptr, zeros_ptr,
    # Matrix dimensions
    M, N,
    pack_num, w_bit,
    # Quantization parameters
    group_size, offset,
    CACHE_KEY_M,
    CACHE_KEY_N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am,
    # Quantization parameters
    # Meta-parameters
    BATCHSIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """
    Kernel for computing GEMV C = qw.T @ a
    - Input a: (Batch_size, M)
    - Weight qw: (M // pack_num, N)
    - scales: (M // group_size, N)
    - qzeros: (M // group_size // pack_num, N)
    - Output Y: (Batch_size, N)
    """
    start_m = tl.program_id(0)
    start_n = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a_ptr = a_ptr + rm
    c_ptr = c_ptr + rn

    # load weight
    qw_shifter = (offs_m % pack_num) * w_bit
    qw_off = (rm[:, None] // pack_num) * stride_am + rn[None, :]
    qw_ptr = qw_ptr + qw_off
    qw_packed = tl.load(qw_ptr) if EVEN_N else tl.load(qw_ptr, mask=rn[None, :] < N, other=0.0)
    qw_unpacked = (qw_packed >> qw_shifter[:, None]) & offset

    # load sacle
    m_iters_per_quant_group = group_size // BLOCK_M
    grp_idx = start_m // m_iters_per_quant_group
    col_offs = rn
    scales = tl.load(scales_ptr + (stride_am * grp_idx) + col_offs)  # (N,)

    # load zeros
    packed_zeros = tl.load(
        zeros_ptr + stride_am * (grp_idx // pack_num) + col_offs
    )
    unpacked_zeros = (packed_zeros >> ((grp_idx % pack_num) * w_bit)) & offset

    # dequant w
    dequantized_w = scales[None, :].to(tl.float32) * (
        qw_unpacked.to(tl.float32) - unpacked_zeros[None, :].to(tl.float32)
    )

    # if BATCHSIZE == 1:
    a0 = tl.load(a_ptr, mask=rm < M, other=0.0)
    acc0 = tl.sum(dequantized_w.to(tl.float32) * a0.to(tl.float32)[:, None], 0)

    # rematerialize rm and rn to save registers
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.atomic_add(c_ptr, acc0, mask=rn < N)


def quant_gemv_v2(a, qw, qzeros, scales, M, N, pack_num, group_size, w_bit, offset):
    """
    :param x: input tensor, (batch, M)
    :param weight: weight matrix, (M // pack_num, N)
    :param qzeros: (M // group_size // pack_num, N)
    :param scales: (M // group_size, N)
    :return: result tensor, (batch, N)
    """
    batch, _ = a.shape
    c = torch.empty((batch, N), dtype=torch.float16, device=a.device)

    assert qw.shape == (M // pack_num, N)
    # assert qzeros.shape == (M // group_size // pack_num, N)
    # assert scales.shape == (M // group_size, N)
    assert all(x.is_contiguous() for x in [a, qw, c, qzeros, scales])
    # BLOCK_SIZE_K has possible values of 32, 64
    # group_size, K must be divisible by BLOCK_SIZE_K
    assert group_size % 64 == 0, f"group_size {group_size} is not a multiple of 64"
    assert M % 64 == 0, f"K {K} is not a multiple of 64"
    # BLOCK_SIZE_N has possible values of 32, 64, 128, 256
    # N must be divisible by BLOCK_SIZE_N
    assert N % 256 == 0, f"N {N} is not a multiple of 256"

    
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),
                         triton.cdiv(N, META["BLOCK_N"]),) 
    transposed_gemv_atomicadd_kernel[grid](
        a_ptr=a,
        qw_ptr=qw,
        c_ptr=c,
        scales_ptr=scales,
        zeros_ptr=qzeros,
        M=M,
        N=N,
        pack_num=pack_num,
        w_bit=w_bit,
        group_size=group_size,
        offset=offset,
        CACHE_KEY_M=M,  # key for triton cache (limit number of compilations)
        CACHE_KEY_N=N,
        stride_am=qw.stride(0),
        BATCHSIZE=batch
    )

    return c


# def triton_transposed_gemv_v2(x: torch.Tensor,
#                            weight: torch.Tensor) -> torch.Tensor:
#     """
#     :param x: input tensor, (batch, M)
#     :param weight: weight matrix, (M, N)
#     :return: result tensor, (batch, N)
#     """
#     M, N = weight.shape
#     batch, _ = x.shape
#     assert x.shape == (batch, M)
#     assert batch in [1]
#     assert all(x.is_contiguous() for x in [x, weight])

#     output = torch.empty(batch, N, device=x.device, dtype=x.dtype)

#     # 1D launch kernel where each block gets its own program.
#     grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),
#                          triton.cdiv(N, META["BLOCK_N"]),) 
#     transposed_gemv_atomicadd_kernel[grid](
#         x,weight,output,
#         M,N,
#         M // 1024,  # key for triton cache (limit number of compilations)
#         N // 32,  # key for triton cache (limit number of compilations)
#         weight.stride(0),  # strides
#         batch,  # Can't use kwargs because auto-tuner requires args
#     )

#     return output

# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4),
#     ],
#     key=["CACHE_KEY_M", "CACHE_KEY_N", "BATCHSIZE"],
# )

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 256}, num_warps=4),
        # triton.Config({"BLOCK_M": 8, "BLOCK_N": 512}, num_warps=4),
        # triton.Config({"BLOCK_M": 8, "BLOCK_N": 1024}, num_warps=4),
        # triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4),
        # triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4),
        # triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=4),
    ],
    key=["CACHE_KEY_M", "CACHE_KEY_N", "BATCHSIZE"],
)
@triton.heuristics(
    {
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def gemv_kernel_v3(
    # Pointers to matrices
    a_ptr, qw_ptr, c_ptr, scales_ptr, zeros_ptr,
    # Matrix dimensions
    M, N,
    pack_num, group_size,
    # Quantization parameters
    w_bit, offset,
    CACHE_KEY_M,
    CACHE_KEY_N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_qw, stride_scale, stride_zeros,
    # Quantization parameters
    # Meta-parameters
    BATCHSIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """
    Kernel for computing GEMV C = qw @ a
    - Input a: (Batch_size, N)
    - Weight qw: (M, N // pack_num)
    - scales: (M, N // group_size)
    - qzeros: (M, N // group_size // pack_num)
    - Output Y: (Batch_size, M)
    """
    start_m = tl.program_id(0)

    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)

    a_ptr = a_ptr + rn

    # load weight
    qw_shifter = (rn % pack_num) * w_bit
    qw_off = rm[:, None] * stride_qw + rn[None, :] // pack_num
    qw_ptr = qw_ptr + qw_off

    acc0 = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for n in range(0, N, BLOCK_N): 
        # load activation
        a0 = tl.load(a_ptr) if EVEN_N else tl.load(a_ptr, mask=rn[None, :] < N, other=0.0)

        # unpack weight
        qw_packed = tl.load(qw_ptr)
        qw_unpacked = (qw_packed >> qw_shifter[None, :]) & offset

        # load scale
        grp_idx = rn[None, :] // group_size;
        scales = tl.load(scales_ptr + rm[:, None] * stride_scale + grp_idx)

        # load zero
        packed_zeros = tl.load(
            zeros_ptr + rm[:, None] * stride_zeros + (grp_idx // pack_num)
        )
        unpacked_zeros = (packed_zeros >> ((grp_idx % pack_num) * w_bit)) & offset

        # dequant w
        dequantized_w = scales.to(tl.float32) * (
            qw_unpacked.to(tl.float32) - unpacked_zeros.to(tl.float32)
        )

        acc0 += tl.sum(dequantized_w.to(tl.float32) * a0.to(tl.float32)[None, :], 1)

        qw_ptr += BLOCK_N // pack_num
        scales_ptr += BLOCK_N // group_size
        zeros_ptr += BLOCK_N // group_size // pack_num
        a_ptr += BLOCK_N

    # rematerialize rm and rn to save registers
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back result
    c_ptr = c_ptr + rm
    tl.store(c_ptr, acc0, mask=rm < M)


def quant_gemv_v3(a, qw, qzeros, scales, M, N, pack_num, group_size, w_bit, offset):
    """
    :param a: input tensor, (batch, N)
    :param weight: weight matrix, (M, N // pack_num)
    :param qzeros: (M, N // group_size // pack_num)
    :param scales: (M, N // group_size)
    :return: result tensor, (batch, M)
    """
    batch, _ = a.shape
    c = torch.empty((batch, M), dtype=torch.float16, device=a.device)

    assert qw.shape == (M, N // pack_num)
    # assert qzeros.shape == (M // group_size // pack_num, N)
    # assert scales.shape == (M // group_size, N)
    assert all(x.is_contiguous() for x in [a, qw, c, qzeros, scales])
    assert group_size % 64 == 0, f"group_size {group_size} is not a multiple of 64"
    # assert (N // pack_num) % 256 == 0, f"N // pack_num {N // pack_num} is not a multiple of 256"

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)  # noqa

    gemv_kernel_v3[grid](
        a_ptr=a,
        qw_ptr=qw,
        c_ptr=c,
        scales_ptr=scales,
        zeros_ptr=qzeros,
        M=M,
        N=N,
        pack_num=pack_num,
        w_bit=w_bit,
        group_size=group_size,
        offset=offset,
        CACHE_KEY_M=M // 512,  # key for triton cache (limit number of compilations)
        CACHE_KEY_N=N // 1024,
        stride_qw=qw.stride(0),
        stride_scale=scales.stride(0),
        stride_zeros=qzeros.stride(0),
        BATCHSIZE=batch
    )

    return c