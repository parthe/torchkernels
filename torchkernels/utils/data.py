"""Save kernel matrices with optional packed triangular storage."""

import numpy as np
import torch
from typing import Optional

_DTYPES = {
    str(dtype): dtype
    for dtype in (
        torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32,
        torch.int64, torch.float16, torch.bfloat16, torch.float32,
        torch.float64, torch.complex64, torch.complex128,
    )
}


def _validate_kmat(kmat, triangle):
    if not isinstance(kmat, torch.Tensor):
        raise TypeError("kmat must be a torch.Tensor")
    if kmat.layout != torch.strided or kmat.device.type == "meta":
        raise ValueError("kmat must be a dense tensor with stored values")
    if kmat.ndim != 2:
        raise ValueError("kmat must be a matrix")
    if triangle is not None and kmat.shape[0] != kmat.shape[1]:
        raise ValueError("Triangular storage requires a square matrix")
    if str(kmat.dtype) not in _DTYPES:
        raise ValueError(f"Unsupported dtype: {kmat.dtype}")
    if triangle not in (None, "upper", "lower"):
        raise ValueError("triangle must be None, 'upper', or 'lower'")


def _validate_archive(archive):
    required = {"version", "size", "triangle", "dtype", "values"}
    if not required.issubset(archive.files):
        raise ValueError("Not a packed kernel-matrix archive")
    for key in required - {"values"}:
        if archive[key].shape != ():
            raise ValueError(f"Invalid scalar metadata: {key}")
    if archive["version"].item() != 1:
        raise ValueError("Unsupported kernel-matrix archive version")
    size = archive["size"].item()
    triangle = archive["triangle"].item()
    dtype_name = archive["dtype"].item()
    if type(size) is not int or size < 0:
        raise ValueError("Invalid matrix size")
    if triangle not in ("none", "upper", "lower"):
        raise ValueError("Invalid stored triangle")
    shape = (size, size)
    count = size * (size + 1) // 2
    if triangle == "none":
        if "shape" not in archive.files:
            raise ValueError("Missing matrix shape")
        stored_shape = archive["shape"]
        if (stored_shape.shape != (2,) or stored_shape.dtype.kind not in "iu"
                or np.any(stored_shape < 0) or stored_shape[0] != size):
            raise ValueError("Invalid matrix shape")
        shape = tuple(int(dimension) for dimension in stored_shape)
        count = shape[0] * shape[1]
    if dtype_name not in _DTYPES:
        raise ValueError("Unsupported stored dtype")
    dtype = _DTYPES[dtype_name]
    values = archive["values"]
    itemsize = torch.empty((), dtype=dtype, device="cpu").element_size()
    if (values.dtype != np.uint8 or values.ndim != 1
            or values.size != count * itemsize):
        raise ValueError("Packed data does not match the matrix size and dtype")
    return shape, triangle, dtype, values


def save_kmat(kmat: torch.Tensor, path, *, triangle=None, compress=False):
    """Save a matrix as-is, or optionally pack one triangle.

    ``kmat`` may be on any device with stored values. ``triangle=None``
    stores the full matrix, including rectangular or asymmetric matrices.
    With ``triangle='upper'`` or ``'lower'``, only that triangle is read;
    symmetry is assumed and is not checked. Values retain their exact dtype
    and precision, including bfloat16. Complex matrices are treated as
    symmetric, not Hermitian.

    Triangular storage contains n*(n+1)/2 values, about half a dense matrix.
    An equally sized CPU buffer is used during saving, without allocating
    a full matrix or triangular index arrays on the GPU. Compression uses
    ``numpy.savez_compressed``; otherwise ``numpy.savez`` is used.
    ``path`` is a filesystem path used exactly as supplied (no extension is
    appended). The file format is an NPZ archive. Autograd history is omitted.
    """
    _validate_kmat(kmat, triangle)
    size = kmat.shape[0]
    source = kmat.detach()
    if triangle is None:
        # Copy directly into contiguous CPU storage even for strided GPU inputs.
        packed = torch.empty(kmat.numel(), dtype=kmat.dtype, device="cpu")
        packed.reshape(kmat.shape).copy_(source)
    else:
        packed = torch.empty(size * (size + 1) // 2, dtype=kmat.dtype, device="cpu")
        offset = 0
        for row in range(size):
            values = source[row, row:] if triangle == "upper" else source[row, :row + 1]
            count = values.numel()
            packed[offset:offset + count].copy_(values)
            offset += count

    # Bytes preserve every supported torch dtype, including NumPy-less bfloat16.
    writer = np.savez_compressed if compress else np.savez
    with open(path, "wb") as file:
        writer(file, version=np.array(1), size=np.array(size),
               shape=np.array(kmat.shape, dtype=np.int64),
               triangle=np.array("none" if triangle is None else triangle),
               dtype=np.array(str(kmat.dtype)),
               values=packed.view(torch.uint8).numpy())


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_kmat(path, *, device: Optional[torch.device] = None) -> torch.Tensor:
    """Load a matrix archive into a torch tensor on ``device``.

    The original dtype is preserved. Pass a ``torch.device`` to select the
    destination explicitly. By default, select CUDA if available, then MPS,
    then CPU. Availability is checked at each call. The destination must
    support the stored dtype; use ``torch.device('cpu')`` for dtypes that
    your accelerator does not support. Full storage restores the matrix
    as-is. Triangular storage is mirrored without conjugation and requires
    the packed CPU buffer plus the full output on the requested device.
    """
    with np.load(path, allow_pickle=False) as archive:
        shape, triangle, dtype, values = _validate_archive(archive)
    if device is None:
        device = _default_device()
    if values.size == 0:
        return torch.empty(shape, dtype=dtype, device=device)
    packed = torch.from_numpy(values).view(dtype)
    if triangle == "none":
        return packed.reshape(shape).to(device=device)
    size = shape[0]
    kmat = torch.empty(shape, dtype=dtype, device=device)
    offset = 0
    for row in range(size):
        count = size - row if triangle == "upper" else row + 1
        values = packed[offset:offset + count]
        if triangle == "upper":
            kmat[row, row:].copy_(values)
            kmat[row + 1:, row].copy_(kmat[row, row + 1:])
        else:
            kmat[row, :row + 1].copy_(values)
            kmat[:row, row].copy_(kmat[row, :row])
        offset += count
    return kmat
