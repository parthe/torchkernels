"""Input checks for the resident multi-GPU matrix-vector operator."""

import math
from numbers import Real
import torch


def validate_matrix(matrix, memory_fraction, reserve_bytes, *, allow_mps=False):
    if not isinstance(matrix, torch.Tensor):
        raise TypeError("matrix must be a torch.Tensor")
    allowed = ("cpu", "cuda", "mps") if allow_mps else ("cpu", "cuda")
    if matrix.layout != torch.strided or matrix.device.type not in allowed:
        raise ValueError(f"matrix must be a dense tensor on one of {allowed}")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.shape[0]:
        raise ValueError("matrix must be a nonempty square matrix")
    if matrix.dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
        raise TypeError("cuBLAS SYMV supports float32, float64, complex64, and complex128")
    if not math.isfinite(memory_fraction) or not 0 < memory_fraction <= 1:
        raise ValueError("memory_fraction must be in (0, 1]")
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("reserve_bytes must be a nonnegative integer")


def cuda_devices(devices):
    count = torch.cuda.device_count()
    if devices is None:
        devices = range(count)
    result = []
    for value in devices:
        device = torch.device("cuda", value) if type(value) is int else torch.device(value)
        if device.type != "cuda":
            raise ValueError("devices must contain only CUDA devices")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= count:
            raise ValueError(f"CUDA device index out of range: {index}")
        device = torch.device("cuda", index)
        if device in result:
            raise ValueError("devices must not contain duplicates")
        result.append(device)
    if not result:
        raise RuntimeError("MultiGpuKernelLinearOperator requires at least one CUDA GPU")
    return result


def validate_vector(vector, size, dtype, *, allow_mps=False):
    if not isinstance(vector, torch.Tensor):
        raise TypeError("vector must be a torch.Tensor")
    allowed = ("cpu", "cuda", "mps") if allow_mps else ("cpu", "cuda")
    if vector.layout != torch.strided or vector.device.type not in allowed:
        raise ValueError(f"vector must be a dense tensor on one of {allowed}")
    if vector.shape != (size,):
        raise ValueError(f"vector must have shape ({size},)")
    if vector.dtype != dtype:
        raise TypeError("vector and matrix must have the same dtype")


def validate_options(block_size, load_balance, tflops):
    if type(block_size) is not int or not 1 <= block_size < 2**31 - 1:
        raise ValueError("block_size must be a positive integer below 2**31 - 1")
    if load_balance not in ("memory", "compute"):
        raise ValueError("load_balance must be 'memory' or 'compute'")
    if load_balance == "memory":
        if tflops is not None:
            raise ValueError("tflops is only used with load_balance='compute'")
        return None
    if tflops is None:
        return None
    try:
        ratings = tuple(tflops)
    except TypeError as error:
        raise TypeError("tflops must be a sequence of positive finite ratings") from error
    if not ratings or any(isinstance(value, bool) or not isinstance(value, Real)
                          or not math.isfinite(value) or value <= 0 for value in ratings):
        raise ValueError("tflops ratings must be positive finite numbers")
    return tuple(float(value) for value in ratings)
