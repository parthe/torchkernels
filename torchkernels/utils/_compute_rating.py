"""Estimate effective GEMV TFLOPS for automatic compute placement."""

import math
from statistics import median

import torch


@torch.no_grad()
def _measure_device(device, stream, dtype, width):
    """Time both directions of one rectangle, excluding allocation/warmup.

    Bounded to three samples of ten paired products. Uses no random numbers
    or global backend-setting changes. The private stream is synchronized
    before local tensor storage can be released, including on exceptions.
    """
    iterations = 10
    elapsed = []
    with torch.cuda.device(device), torch.cuda.stream(stream):
        try:
            matrix = torch.empty((width, width), dtype=dtype, device=device)
            matrix.fill_(1 / width)
            vector = torch.ones(width, dtype=dtype, device=device)
            output = torch.empty_like(vector)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(3):
                torch.mv(matrix, vector, out=output)
                torch.mv(matrix.T, vector, out=output)
            for _ in range(3):
                start.record(stream)
                for _ in range(iterations):
                    torch.mv(matrix, vector, out=output)
                    torch.mv(matrix.T, vector, out=output)
                end.record(stream)
                end.synchronize()
                elapsed.append(start.elapsed_time(end))
        finally:
            stream.synchronize()
    milliseconds = median(elapsed)
    if not math.isfinite(milliseconds) or milliseconds <= 0:
        raise RuntimeError("CUDA benchmark returned an invalid duration; provide explicit tflops ratings")
    operations = iterations * 4 * width**2
    if dtype in (torch.complex64, torch.complex128):
        operations *= 4
    return operations / (milliseconds * 1e9)


def estimate_tflops(devices, streams, dtype, size, block_size, capacities):
    """Benchmark a common tile width fitting every eligible selected GPU.

    Return ratings in selected-device order; zero-capacity GPUs get zero and
    are excluded from placement. The matrix plus two short vectors fits the
    existing matrix-capacity and full-vector budget. At most 1024² values are
    allocated per device, and devices are benchmarked sequentially.
    """
    required = size * (size + 1) // 2
    if sum(capacities) < required:
        raise MemoryError("Insufficient GPU storage for the packed matrix before compute calibration")
    if capacities[0] < size:
        raise MemoryError("First selected GPU cannot hold even the principal diagonal")
    smallest = min(capacity for capacity in capacities if capacity > 0)
    width = min(1024, block_size, max(1, size // 2), math.isqrt(smallest))
    ratings = []
    for device, stream, capacity in zip(devices, streams, capacities):
        if capacity == 0:
            ratings.append(0.0)
            continue
        rating = _measure_device(device, stream, dtype, width)
        if not math.isfinite(rating) or rating <= 0:
            raise RuntimeError("Invalid automatic compute rating; provide explicit tflops ratings")
        ratings.append(rating)
    return tuple(ratings)
