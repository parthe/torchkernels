"""Packed symmetric matrix storage and single-device linear operators."""

from threading import Lock

import torch

from ._cublas_symv import CublasSymv
from ._multigpu_validation import validate_matrix, validate_options, validate_vector
from ._triangle_tiles import accumulate_rectangles, pack_diagonal_pairs, pack_rectangles, plan_tiles


def _accumulate_diagonals(packed, specs, vector, output):
    """CPU/MPS triangle products without reconstructing a dense matrix.

    Each stored off-diagonal entry contributes to two output positions; the
    diagonal contributes once. Views touch only the selected triangle.
    """
    for start, size, offset, lda, upper in specs:
        for column in range(size):
            base = offset + column * lda
            if upper:
                values = packed[base:base + column + 1]
                output[start:start + column + 1].addcmul_(values, vector[start + column])
                output[start + column].add_(torch.dot(values[:-1], vector[start:start + column]))
            else:
                values = packed[base + column:base + size]
                output[start + column:start + size].addcmul_(values, vector[start + column])
                output[start + column].add_(torch.dot(values[1:], vector[start + column + 1:start + size]))


class SymmetricLinearOperator:
    """A packed, symmetric snapshot on one CPU, MPS, or CUDA device.

    ``SymmetricLinearOperator(K, device=...)`` retains exactly n*(n+1)/2
    matrix values. Device defaults to K.device. Only K's upper triangle is
    read; complex symmetry uses transpose without conjugation. Paired diagonal
    blocks and rectangles share one flat allocation, with no dense matrix
    reconstruction. CUDA uses cuBLAS SYMV for paired diagonal blocks; CPU/MPS
    accumulate directly from triangle views. Rectangles use ordinary GEMV.

    ``operator @ x``, ``operator(x)``, and ``operator.matvec(x)`` return fresh
    CPU vectors. Inputs must have shape (n,) and matching dtype. Autograd is
    not supported. Products are serialized for safe reuse of two vector buffers.
    ``storage_numel`` reports the matrix storage as a one-element tuple.

    ``block_size`` bounds block width. On CUDA, ``memory_fraction`` and
    ``reserve_bytes`` leave space for other allocations and library workspaces,
    in addition to explicitly budgeting the two full vector buffers. There
    is no dense scratch matrix. On CPU/MPS these CUDA budgeting options do not
    apply. This class never distributes data or benchmarks device throughput.
    """

    @torch.no_grad()
    def __init__(self, matrix, *, device=None, block_size=1024,
                 memory_fraction=0.9, reserve_bytes=64 * 1024**2):
        validate_matrix(matrix, memory_fraction, reserve_bytes, allow_mps=True)
        validate_options(block_size, 'memory', None)
        self.device = matrix.device if device is None else torch.device(device)
        if self.device.type not in ('cpu', 'cuda', 'mps'):
            raise ValueError('device must be CPU, CUDA, or MPS')
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.shape = tuple(matrix.shape)
        self.dtype = matrix.dtype
        self.devices = (self.device,)
        self._lock = Lock()
        self._stream = None
        self._symv = None
        size = self.shape[0]
        required = size * (size + 1) // 2
        if self.device.type == 'cuda':
            self._stream = torch.cuda.Stream(device=self.device)
            self._symv = CublasSymv(self.dtype, self.device, self._stream)
            free = torch.cuda.mem_get_info(self.device)[0]
            available = int(free * memory_fraction) - reserve_bytes - 2 * size * matrix.element_size()
            if required * matrix.element_size() > available:
                raise MemoryError('Insufficient CUDA memory for packed matrix and vector buffers')
        blocks, assignments, _ = plan_tiles(size, block_size, [required])
        self.diagonal_blocks = tuple(blocks)
        self.tile_ranges = (tuple(assignments[0]),)
        self.storage_numel = (required,)
        self._packed = torch.empty(required, dtype=self.dtype, device=self.device)
        self._inputs = torch.empty(size, dtype=self.dtype, device=self.device)
        self._outputs = torch.empty_like(self._inputs)
        if matrix.is_cuda:
            torch.cuda.current_stream(matrix.device).synchronize()
        self._diagonal_specs, offset = pack_diagonal_pairs(matrix.detach(), blocks, self._packed)
        self._rectangles = pack_rectangles(matrix.detach(), assignments[0], self._packed, offset)
        if self.device.type == 'cuda':
            torch.cuda.current_stream(self.device).synchronize()

    @torch.no_grad()
    def matvec(self, vector):
        validate_vector(vector, self.shape[0], self.dtype, allow_mps=True)
        with self._lock:
            source = vector.detach()
            if source.is_cuda:
                torch.cuda.current_stream(source.device).synchronize()
                source = source.cpu()
            if self.device.type == 'cuda':
                try:
                    with torch.cuda.stream(self._stream):
                        self._inputs.copy_(source)
                        self._outputs.zero_()
                        self._symv.accumulate(self._packed, self._diagonal_specs,
                                              self._inputs, self._outputs)
                        accumulate_rectangles(self._packed, self._rectangles,
                                              self._inputs, self._outputs)
                finally:
                    self._stream.synchronize()
            else:
                self._inputs.copy_(source)
                self._outputs.zero_()
                _accumulate_diagonals(self._packed, self._diagonal_specs,
                                      self._inputs, self._outputs)
                accumulate_rectangles(self._packed, self._rectangles,
                                      self._inputs, self._outputs)
            return self._outputs.to(device='cpu', copy=True)

    def __matmul__(self, vector):
        return self.matvec(vector)

    def __call__(self, vector):
        return self.matvec(vector)
