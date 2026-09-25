"""Small lazy cuBLAS SYMV binding; uses PyTorch CUDA storage and streams.

No compiler, CuPy, or import-time CUDA initialization is required. The handle
is private to one operator, so stream and pointer-mode changes cannot affect
PyTorch's own cuBLAS handles.
"""

import ctypes
import ctypes.util
from pathlib import Path
import sys
import weakref

import torch


_TYPES = {
    torch.float32: ("S", ctypes.c_float, False),
    torch.float64: ("D", ctypes.c_double, False),
    torch.complex64: ("C", ctypes.c_float, True),
    torch.complex128: ("Z", ctypes.c_double, True),
}


def _symbol(library, name, arguments):
    function = getattr(library, name + "_v2", None)
    if function is None:
        function = getattr(library, name)
    function.argtypes = arguments
    function.restype = ctypes.c_int
    return function


def _check(status, operation):
    if status:
        raise RuntimeError(f"{operation} failed with cuBLAS status {status}")


def _load_cublas():
    major = (torch.version.cuda or "").split('.')[0]
    names = ([f"cublas64_{major}.dll", "cublas64_11.dll"] if sys.platform == "win32"
             else [f"libcublas.so.{major}", "libcublas.so"])
    located = ctypes.util.find_library("cublas")
    if located:
        names.append(located)
    roots = [Path(torch.__file__).parent / 'lib']
    for directory in sys.path:
        if directory:
            roots.extend([Path(directory) / 'nvidia' / 'cublas' / 'lib',
                          Path(directory) / 'nvidia' / 'cu13' / 'lib'])
    pattern = 'cublas64_*.dll' if sys.platform == 'win32' else 'libcublas.so*'
    for root in roots:
        names.extend(str(path) for path in sorted(root.glob(pattern)))
    # Try already-loaded global symbols, then loader names and wheel locations.
    for name in dict.fromkeys([None] + names):
        try:
            library = ctypes.CDLL(name)
            getattr(library, 'cublasCreate_v2')
            return library
        except (OSError, AttributeError):
            continue
    raise RuntimeError("Cannot load cuBLAS. Install a CUDA-enabled PyTorch build "
                       "and ensure its cuBLAS shared library is available.")


def _destroy_handle(destroy, handle, device):
    # Finalizers may also run during interpreter shutdown.
    try:
        with torch.cuda.device(device):
            destroy(handle)
    except Exception:
        pass


class CublasSymv:
    """Accumulate symmetric products from offset column-major triangle views."""

    def __init__(self, dtype, device, stream):
        if dtype not in _TYPES:
            raise TypeError("cuBLAS SYMV requires float32, float64, complex64, or complex128")
        self.dtype = dtype
        self.device = device
        self.stream = stream
        self.library = _load_cublas()
        pointer = ctypes.c_void_p
        integer = ctypes.c_int
        create = _symbol(self.library, 'cublasCreate', [ctypes.POINTER(pointer)])
        destroy = _symbol(self.library, 'cublasDestroy', [pointer])
        set_stream = _symbol(self.library, 'cublasSetStream', [pointer, pointer])
        set_pointer_mode = _symbol(self.library, 'cublasSetPointerMode', [pointer, integer])
        prefix, scalar_type, complex_type = _TYPES[dtype]
        self._symv = _symbol(self.library, f'cublas{prefix}symv',
                             [pointer, integer, integer, pointer, pointer, integer,
                              pointer, integer, pointer, pointer, integer])
        self._one = (scalar_type * 2)(1, 0) if complex_type else scalar_type(1)
        self.handle = pointer()
        with torch.cuda.device(device):
            _check(create(ctypes.byref(self.handle)), 'cublasCreate')
            self._finalizer = weakref.finalize(self, _destroy_handle, destroy,
                                               self.handle, device)
            _check(set_stream(self.handle, pointer(stream.cuda_stream)), 'cublasSetStream')
            _check(set_pointer_mode(self.handle, 0), 'cublasSetPointerMode')

    def accumulate(self, packed, specs, vector, output):
        # All tensors are owned, contiguous buffers on the handle's device.
        # Operator calls serialize access and synchronize the private stream.
        itemsize = packed.element_size()
        one = ctypes.byref(self._one)
        for start, size, offset, lda, upper in specs:
            _check(self._symv(
                self.handle, int(upper), size, one,
                packed.data_ptr() + offset * itemsize, lda,
                vector.data_ptr() + start * itemsize, 1, one,
                output.data_ptr() + start * itemsize, 1), 'cublasSymv')

    def close(self):
        self._finalizer()
