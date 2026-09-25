"""Automatic single-/multi-device kernel linear operators."""

from abc import ABCMeta

import torch

from ..utils.packed import SymmetricLinearOperator


class _KernelOperatorMeta(ABCMeta):
    """Dispatch construction exactly once; subclasses construct normally."""

    def __call__(cls, *args, **kwargs):
        if cls is KernelLinearOperator:
            from ..utils._multigpu_validation import cuda_devices
            requested = kwargs.get('devices')
            if requested is None:
                selected = tuple(torch.device('cuda', index)
                                 for index in range(torch.cuda.device_count()))
            else:
                selected = tuple(cuda_devices(requested))
            # Normalize once, including generators, and honor CUDA visibility.
            kwargs['devices'] = selected
            if len(selected) > 1:
                from ..utils.multigpu_matvec import MultiGpuKernelLinearOperator
                return MultiGpuKernelLinearOperator(*args, **kwargs)
        return super().__call__(*args, **kwargs)


class KernelLinearOperator(SymmetricLinearOperator, metaclass=_KernelOperatorMeta):
    """Construct a symmetric kernel operator and multiply with ``Kmat @ x``.

    With multiple visible/selected CUDA GPUs, construction returns a
    MultiGpuKernelLinearOperator. With one selected GPU, this class stores a
    packed symmetric snapshot there. Without CUDA it preserves the source
    device (CPU or MPS). ``devices=[0]`` explicitly selects one CUDA GPU even
    when more are visible. CUDA_VISIBLE_DEVICES is honored by PyTorch.

    Every backend treats the upper triangle as authoritative, including
    complex symmetric matrices without conjugation. Products take a vector
    of shape (n,) and matching dtype, and return a fresh CPU tensor without
    autograd. ``operator(x)`` and ``operator.matvec(x)`` are equivalent to @.
    These operators are not Tensor subclasses and do not support torch.matmul
    or matrix right-hand sides.

    Memory balancing remains the default. Balancing and calibration apply to
    the multi-GPU implementation; a single-device operator does not distribute
    data or benchmark. Every backend retains exactly n*(n+1)/2 matrix values. Subclasses can override matvec
    and construction without being automatically dispatched.
    """

    @torch.no_grad()
    def __init__(self, matrix, *, devices=None, memory_fraction=0.9,
                 reserve_bytes=64 * 1024**2, block_size=1024,
                 load_balance='memory', tflops=None):
        from ..utils._multigpu_validation import cuda_devices, validate_options

        ratings = validate_options(block_size, load_balance, tflops)
        selected = tuple(cuda_devices(devices)) if devices else ()
        if len(selected) > 1:
            raise ValueError('Single-device kernel initialization expects at most one CUDA GPU')
        if ratings is not None and len(ratings) != 1:
            raise ValueError('tflops must have one rating for a single-device operator')
        self.load_balance = load_balance
        self.tflops = ratings
        self.tflops_source = 'provided' if ratings is not None else None
        super().__init__(matrix, device=selected[0] if selected else None,
                         block_size=block_size, memory_fraction=memory_fraction,
                         reserve_bytes=reserve_bytes)
