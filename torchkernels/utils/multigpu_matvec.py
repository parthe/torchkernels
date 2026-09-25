"""Memory- or compute-balanced symmetric matrix storage across CUDA GPUs."""

from threading import Lock

import torch

from ..linalg.linear_operator import KernelLinearOperator
from ._cublas_symv import CublasSymv
from ._compute_rating import estimate_tflops
from ._multigpu_validation import cuda_devices, validate_matrix, validate_options, validate_vector
from ._triangle_tiles import accumulate_rectangles, pack_diagonal_pairs, pack_rectangles, plan_tiles


def _storage_capacities(size, itemsize, free_bytes, memory_fraction, reserve_bytes):
    # Each active GPU retains one input and one partial output vector.
    buffers = 2 * size * itemsize
    return [max(0, int(free * memory_fraction) - reserve_bytes - buffers) // itemsize
            for free in free_bytes]


class MultiGpuKernelLinearOperator(KernelLinearOperator):
    """Resident symmetric operator; construct once and use ``Kmat @ x``.

    The upper triangle is authoritative, including for complex symmetric
    matrices (no conjugation). Exactly n*(n+1)/2 matrix elements are retained.
    Paired diagonal tiles live on the FIRST SELECTED GPU (CUDA 0 by default).
    Each pair occupies a column-major (b+1) x b rectangle: cuBLAS SYMV reads
    its upper triangle at offset 0 and lower triangle at offset 1. No diagonal
    expansion, scratch matrix, or vector permutation is used. Odd n has one
    final scalar diagonal tile. Only float32/64 and complex64/128 are supported.

    ``load_balance='memory'`` allocates total matrix elements in proportion to
    usable free memory, subject to the mandatory diagonal allocation.
    ``load_balance='compute'`` automatically estimates effective TFLOPS using
    a short, dtype-matched GEMV benchmark on each eligible GPU. Optional
    ``tflops=[...]`` in selected-device order overrides calibration. These
    weights balance estimated work, include GPU 0's diagonal work, and respect
    hard memory caps. ``tflops_source`` is 'benchmark', 'provided', or None.
    Memory mode (the default) never benchmarks. Measurements reflect tile size,
    cache and current GPU activity; they are not theoretical peak ratings.
    Placement stays fixed and does not model communication or SYMV performance.

    ``block_size`` bounds diagonal block width (default 1024). Paired blocks
    shrink to fit GPU 0 and its fair work/storage share; rectangles may split
    further for balance.
    ``memory_fraction`` uses 90% of currently free memory by default, less
    ``reserve_bytes`` (64 MiB per GPU) and two full vectors. The reserve covers
    CUDA workspaces; fragmentation or concurrent allocations may still OOM.
    Insufficient capacity raises MemoryError before matrix allocation. GPU 0
    must at least fit the n principal diagonal elements after reservations.

    ``devices``, ``diagonal_blocks``, ``tile_ranges`` (rectangles per device),
    ``storage_numel``, and ``estimated_flops`` describe the actual placement.
    CPU matrix input avoids retaining an extra source allocation on a GPU.
    No source reference or autograd graph is retained. Products accept shape
    (n,) with matching dtype, return fresh CPU vectors, and serialize callers
    for safe buffer reuse. CUDA input vectors and cross-device source matrix
    tiles are staged through CPU, avoiding peer-copy dependencies. Each device's
    partial result is summed on CPU.
    """

    @torch.no_grad()
    def __init__(self, matrix, *, devices=None, memory_fraction=0.9,
                 reserve_bytes=64 * 1024**2, block_size=1024,
                 load_balance="memory", tflops=None):
        validate_matrix(matrix, memory_fraction, reserve_bytes)
        ratings = validate_options(block_size, load_balance, tflops)
        selected = cuda_devices(devices)
        if ratings is not None and len(ratings) != len(selected):
            raise ValueError("tflops must have one rating per selected GPU")
        self.shape = tuple(matrix.shape)
        self.dtype = matrix.dtype
        self.load_balance = load_balance
        self.tflops = ratings
        self.tflops_source = "provided" if ratings is not None else None
        self._lock = Lock()
        self._shards = []
        self._inputs = []
        self._outputs = []
        self._streams = []
        self._rectangles = []
        self.devices = []
        self.tile_ranges = []
        self.storage_numel = []
        self.estimated_flops = []
        # Context and cuBLAS-handle allocations precede the free-memory query.
        streams = [torch.cuda.Stream(device=device) for device in selected]
        self._symv = CublasSymv(self.dtype, selected[0], streams[0])
        free = [torch.cuda.mem_get_info(device)[0] for device in selected]
        capacities = _storage_capacities(self.shape[0], matrix.element_size(), free,
                                         memory_fraction, reserve_bytes)
        if load_balance == "compute" and ratings is None:
            ratings = estimate_tflops(selected, streams, self.dtype, self.shape[0],
                                      block_size, capacities)
            self.tflops = ratings
            self.tflops_source = "benchmark"
            # Calibration may initialize BLAS workspaces and allocator caches.
            # Re-query free memory rather than relying on the earlier snapshot.
            free = [torch.cuda.mem_get_info(device)[0] for device in selected]
            capacities = _storage_capacities(self.shape[0], matrix.element_size(), free,
                                             memory_fraction, reserve_bytes)
            # Devices skipped during calibration stay excluded, even if another
            # process releases memory while the benchmark is running.
            capacities = [capacity if rating > 0 else 0
                          for capacity, rating in zip(capacities, ratings)]
        blocks, assignments, counts = plan_tiles(self.shape[0], block_size, capacities,
                                                  load_balance, ratings)
        self.diagonal_blocks = tuple(blocks)
        if matrix.is_cuda:
            torch.cuda.current_stream(matrix.device).synchronize()
        for index, (device, stream, tiles, count) in enumerate(zip(
                selected, streams, assignments, counts)):
            if not count:
                continue
            with torch.cuda.device(device):
                shard = torch.empty(count, dtype=self.dtype, device=device)
                offset = 0
                if index == 0:
                    self._diagonal_specs, offset = pack_diagonal_pairs(matrix.detach(), blocks, shard)
                rectangles = pack_rectangles(matrix.detach(), tiles, shard, offset)
                inputs = torch.empty(self.shape[0], dtype=self.dtype, device=device)
                outputs = torch.empty_like(inputs)
                torch.cuda.current_stream(device).synchronize()
            self._shards.append(shard)
            self._inputs.append(inputs)
            self._outputs.append(outputs)
            self._streams.append(stream)
            self._rectangles.append(rectangles)
            self.devices.append(device)
            self.tile_ranges.append(tuple(tiles))
            self.storage_numel.append(count)
            flops = 4 * (count - offset)
            if index == 0:
                flops += 2 * sum((stop - start)**2 for start, stop in blocks)
            self.estimated_flops.append(flops * (4 if matrix.is_complex() else 1))
        self.devices = tuple(self.devices)
        self.tile_ranges = tuple(self.tile_ranges)
        self.storage_numel = tuple(self.storage_numel)
        self.estimated_flops = tuple(self.estimated_flops)

    @torch.no_grad()
    def matvec(self, vector):
        """Multiply using cuBLAS SYMV and rectangular GEMV products."""
        validate_vector(vector, self.shape[0], self.dtype)
        with self._lock:
            source = vector.detach()
            if source.is_cuda:
                torch.cuda.current_stream(source.device).synchronize()
                # A vector is small compared with the resident matrix. One
                # host copy makes broadcast independent of CUDA peer links.
                source = source.cpu()
            try:
                for inputs, stream in zip(self._inputs, self._streams):
                    with torch.cuda.stream(stream):
                        inputs.copy_(source)
                for index, (shard, tiles, inputs, outputs, stream) in enumerate(zip(
                        self._shards, self._rectangles, self._inputs, self._outputs, self._streams)):
                    with torch.cuda.stream(stream):
                        outputs.zero_()
                        if index == 0:
                            self._symv.accumulate(shard, self._diagonal_specs, inputs, outputs)
                        accumulate_rectangles(shard, tiles, inputs, outputs)
            finally:
                for stream in self._streams:
                    stream.synchronize()
            result = torch.zeros(self.shape[0], dtype=self.dtype, device="cpu")
            partial = torch.empty_like(result)
            for outputs in self._outputs:
                partial.copy_(outputs)
                result.add_(partial)
            return result
