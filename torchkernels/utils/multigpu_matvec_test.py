"""Run with python -m unittest torchkernels.utils.multigpu_matvec_test -v."""

from contextlib import nullcontext
import ctypes
import unittest
from unittest.mock import MagicMock, patch

import torch

from ..linalg import KernelLinearOperator
from . import MultiGpuKernelLinearOperator
from ._cublas_symv import CublasSymv
from ._compute_rating import _measure_device, estimate_tflops
from ._multigpu_validation import cuda_devices, validate_vector
from .multigpu_matvec import _storage_capacities
from ._triangle_tiles import _weighted_quotas, plan_tiles, pack_diagonal_pairs, pack_rectangles, accumulate_rectangles


class CpuCublasFunction:
    """Emulate cuBLAS at its pointer boundary, without replacing the operator."""

    def __init__(self, name):
        self.name = name
        self.calls = []
        self.status = 0

    def __call__(self, *args):
        self.calls.append(args)
        if self.status:
            return self.status
        if self.name == 'cublasCreate':
            args[0]._obj.value = 123
        if self.name.endswith('symv'):
            emulate_symv(self.name, *args)
        return 0


class CpuCublasLibrary:
    def __init__(self):
        for name in ('cublasCreate', 'cublasDestroy', 'cublasSetStream',
                     'cublasSetPointerMode', 'cublasSsymv', 'cublasDsymv',
                     'cublasCsymv', 'cublasZsymv'):
            setattr(self, name + '_v2', CpuCublasFunction(name))


def pointer_tensor(address, count, dtype):
    byte_count = count * torch.empty((), dtype=dtype).element_size()
    buffer = (ctypes.c_byte * byte_count).from_address(address)
    return torch.frombuffer(buffer, dtype=dtype)


def emulate_symv(name, handle, upper, size, alpha, matrix, lda, x, incx, beta, y, incy):
    dtype = {'S': torch.float32, 'D': torch.float64,
             'C': torch.complex64, 'Z': torch.complex128}[name[6]]
    assert incx == incy == 1
    assert upper in (0, 1)
    assert lda >= size
    flat = pointer_tensor(matrix, (size - 1) * lda + size, dtype)
    view = flat.as_strided((size, size), (1, lda))
    triangle = view.triu() if upper else view.tril()
    symmetric = triangle + triangle.T - torch.diag(triangle.diagonal())
    alpha_value = pointer_tensor(ctypes.addressof(alpha._obj), 1, dtype)[0]
    beta_value = pointer_tensor(ctypes.addressof(beta._obj), 1, dtype)[0]
    output = pointer_tensor(y, size, dtype)
    output.mul_(beta_value).add_(alpha_value * (symmetric @ pointer_tensor(x, size, dtype)))


def emulated_product(matrix, vector, capacities, block_size, mode='memory', ratings=None):
    blocks, groups, counts = plan_tiles(len(vector), block_size, capacities, mode, ratings)
    result = torch.zeros_like(vector)
    library = CpuCublasLibrary()
    with patch('torchkernels.utils._cublas_symv._load_cublas', return_value=library), \
            patch('torch.cuda.device', side_effect=nullcontext):
        backend = CublasSymv(matrix.dtype, torch.device('cpu'), MagicMock(cuda_stream=456))
        for index, (tiles, count) in enumerate(zip(groups, counts)):
            if not count:
                continue
            packed = torch.empty(count, dtype=matrix.dtype)
            output = torch.zeros_like(vector)
            offset = 0
            if index == 0:
                specs, offset = pack_diagonal_pairs(matrix, blocks, packed)
            rectangles = pack_rectangles(matrix, tiles, packed, offset)
            assert torch.isfinite(packed).all()
            if index == 0:
                backend.accumulate(packed, specs, vector, output)
            accumulate_rectangles(packed, rectangles, vector, output)
            result.add_(output)
        backend.close()
    return result


class ComputeRatingTests(unittest.TestCase):
    def test_event_timing_and_flop_conversion(self):
        stream = MagicMock()
        for dtype, factor in ((torch.float64, 1), (torch.complex128, 4)):
            start, end = MagicMock(), MagicMock()
            start.elapsed_time.side_effect = [3.0, 2.0, 1.0]
            with patch('torch.cuda.device', side_effect=nullcontext), \
                    patch('torch.cuda.stream', side_effect=nullcontext), \
                    patch('torch.cuda.Event', side_effect=[start, end]):
                rating = _measure_device(torch.device('cpu'), stream, dtype, 8)
            self.assertEqual(rating, 10 * 4 * 8**2 * factor / (2.0 * 1e9))
            self.assertEqual(end.synchronize.call_count, 3)

    def test_common_width_and_zero_capacity(self):
        with patch('torchkernels.utils._compute_rating._measure_device', side_effect=[1.0, 3.0]) as measure:
            rates = estimate_tflops([0, 1, 2], ['s0', 's1', 's2'], torch.float64,
                                    100, 1024, [400, 0, 5000])
        self.assertEqual(rates, (1.0, 0.0, 3.0))
        self.assertEqual([call.args[-1] for call in measure.call_args_list], [20, 20])
        self.assertEqual([call.args[0] for call in measure.call_args_list], [0, 2])

    def test_invalid_measurement_and_capacity(self):
        with patch('torchkernels.utils._compute_rating._measure_device', return_value=float('nan')):
            with self.assertRaises(RuntimeError):
                estimate_tflops([0], ['s'], torch.float64, 2, 4, [10])
        with self.assertRaises(MemoryError):
            estimate_tflops([0], ['s'], torch.float64, 100, 4, [100])


class PlacementTests(unittest.TestCase):
    def test_memory_balance_includes_diagonal_storage(self):
        _, _, counts = plan_tiles(100, 10, [6000, 6000])
        self.assertEqual(counts, [2525, 2525])
        _, _, counts = plan_tiles(100, 10, [3000, 6000, 9000])
        self.assertEqual(sum(counts), 5050)
        for actual, target in zip(counts, (5050 / 6, 5050 / 3, 5050 / 2)):
            self.assertLessEqual(abs(actual - target), 1)

    def test_compute_balance_includes_diagonal_work(self):
        blocks, _, counts = plan_tiles(100, 10, [6000, 6000], 'compute', [1, 3])
        diagonal = sum((b - a) * (b - a + 1) // 2 for a, b in blocks)
        first_work = 2 * sum((b - a)**2 for a, b in blocks) + 4 * (counts[0] - diagonal)
        second_work = 4 * counts[1]
        self.assertLessEqual(abs(first_work - second_work / 3), 4)
        self.assertEqual(counts, [1300, 3750])

    def test_three_equal_gpus_choose_more_than_two_diagonal_blocks(self):
        blocks, _, counts = plan_tiles(100, 1024, [6000, 6000, 6000])
        self.assertGreaterEqual(len(blocks), 4)
        self.assertEqual(len(blocks) % 2, 0)
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_compute_hard_memory_cap(self):
        _, _, counts = plan_tiles(100, 10, [6000, 1000], 'compute', [1, 100])
        self.assertEqual(counts, [4050, 1000])

    def test_shrink_for_compute_share_and_minimum_diagonal_work(self):
        blocks, _, counts = plan_tiles(100, 50, [6000, 6000], 'compute', [1, 100])
        self.assertTrue(all(stop - start == 1 for start, stop in blocks))
        self.assertEqual(counts, [100, 4950])

    def test_shrink_diagonal_blocks_to_fit_first_gpu(self):
        blocks, _, counts = plan_tiles(100, 50, [100, 5000])
        self.assertTrue(all(stop - start == 1 for start, stop in blocks))
        self.assertEqual(counts, [100, 4950])

    def test_insufficient_memory(self):
        with self.assertRaises(MemoryError):
            plan_tiles(10, 4, [20, 34])
        with self.assertRaisesRegex(MemoryError, 'principal diagonal'):
            plan_tiles(10, 4, [9, 100])

    def test_coverage_sizes_caps_and_zero_capacity(self):
        for size in range(1, 26):
            required = size * (size + 1) // 2
            capacities = [required, 0, required // 2, required]
            for block_size in (1, 3, 16):
                for mode in ('memory', 'compute'):
                    blocks, groups, counts = plan_tiles(size, block_size, capacities,
                                                         mode, [1, 10, 3, 7])
                    self.assertEqual(sum(counts), required)
                    self.assertEqual(counts[1], 0)
                    self.assertEqual(groups[1], [])
                    self.assertTrue(all(c <= capacity for c, capacity in zip(counts, capacities)))
                    seen = torch.zeros(size, size, dtype=torch.int64)
                    diagonal = 0
                    for start, stop in blocks:
                        self.assertLessEqual(stop - start, block_size)
                        for row in range(start, stop):
                            seen[row, row:stop] += 1
                        diagonal += (stop - start) * (stop - start + 1) // 2
                    for i, group in enumerate(groups):
                        elements = diagonal if i == 0 else 0
                        for row, stop, column, end in group:
                            self.assertLessEqual(stop, column)
                            seen[row:stop, column:end] += 1
                            elements += (stop - row) * (end - column)
                        self.assertEqual(elements, counts[i])
                    torch.testing.assert_close(seen, torch.ones_like(seen).triu())

    def test_compute_extreme_ratings_respect_caps(self):
        self.assertEqual(_weighted_quotas(100, [80, 80], [1e-300, 1e300], [0, 0]), [20, 80])
        self.assertEqual(_weighted_quotas(100, [100, 100], [1e-300, 1e-300], [0, 0]), [50, 50])

    def test_headroom_only_vectors(self):
        self.assertEqual(_storage_capacities(10, 4, [800], .5, 64), [64])
        self.assertEqual(_storage_capacities(10, 4, [100], .5, 64), [0])


class ArithmeticTests(unittest.TestCase):
    def test_unflipped_pair_layout_and_pointer_offsets(self):
        matrix = torch.arange(36, dtype=torch.float64).view(6, 6)
        packed = torch.empty(12, dtype=matrix.dtype)
        specs, count = pack_diagonal_pairs(matrix, [(0, 3), (3, 6)], packed)
        expected = torch.tensor([[0, 1, 2], [21, 7, 8], [22, 28, 14], [23, 29, 35]],
                                dtype=matrix.dtype)
        torch.testing.assert_close(packed.view(3, 4).T, expected)
        self.assertEqual(specs, [(0, 3, 0, 4, True), (3, 3, 1, 4, False)])
        self.assertEqual(count, 12)

    def test_dtype_odd_even_strided_and_poisoned_lower_triangle(self):
        for size in (1, 2, 7, 8, 17, 18):
            for dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128):
                matrix = torch.randn(size, size, dtype=dtype).T
                matrix[torch.ones(size, size, dtype=torch.bool).tril(-1)] = float('nan')
                expected = matrix.triu() + matrix.triu(1).T
                vector = torch.randn(size, dtype=dtype)
                for block in (1, 3, 8):
                    required = size * (size + 1) // 2
                    for mode in ('memory', 'compute'):
                        actual = emulated_product(matrix, vector, [required, required],
                                                  block, mode, [1, 3])
                        torch.testing.assert_close(actual, expected @ vector)

    def test_cublas_errors_and_handle_configuration(self):
        library = CpuCublasLibrary()
        with patch('torchkernels.utils._cublas_symv._load_cublas', return_value=library), \
                patch('torch.cuda.device', side_effect=nullcontext):
            backend = CublasSymv(torch.float64, torch.device('cpu'), MagicMock(cuda_stream=456))
            self.assertEqual(library.cublasSetStream_v2.calls[0][1].value, 456)
            self.assertEqual(library.cublasSetPointerMode_v2.calls[0][1], 0)
            library.cublasDsymv_v2.status = 7
            with self.assertRaisesRegex(RuntimeError, 'cuBLAS status 7'):
                backend.accumulate(torch.ones(1, dtype=torch.float64), [(0, 1, 0, 1, True)],
                                   torch.ones(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64))
            backend.close()
            backend.close()
            self.assertEqual(len(library.cublasDestroy_v2.calls), 1)


class OperatorTests(unittest.TestCase):
    def test_base_requires_matrix(self):
        with self.assertRaises(TypeError):
            KernelLinearOperator()
        self.assertTrue(issubclass(MultiGpuKernelLinearOperator, KernelLinearOperator))

    def test_constructor_and_matmul_emulated_cuda(self):
        module = 'torchkernels.utils.multigpu_matvec'
        for mode in ('memory', 'compute', 'automatic'):
            matrix = torch.randn(17, 17, dtype=torch.float64)
            matrix = matrix + matrix.T
            expected = matrix.clone()
            library = CpuCublasLibrary()
            ratings = [1, 3] if mode == 'compute' else None
            actual_mode = 'compute' if mode == 'automatic' else mode
            with patch(module + '.cuda_devices', return_value=[torch.device('cpu')] * 2), \
                    patch('torchkernels.utils._cublas_symv._load_cublas', return_value=library), \
                    patch('torch.cuda.Stream', side_effect=[MagicMock(cuda_stream=123), MagicMock(cuda_stream=456)]), \
                    patch('torch.cuda.mem_get_info', side_effect=[(2000, 2000), (3000, 3000)] * 2) as memory_query, \
                    patch(module + '.estimate_tflops', return_value=(2.0, 4.0)) as estimate, \
                    patch('torch.cuda.device', side_effect=nullcontext), \
                    patch('torch.cuda.current_stream', return_value=MagicMock()), \
                    patch('torch.cuda.stream', side_effect=nullcontext):
                operator = MultiGpuKernelLinearOperator(matrix, block_size=3, memory_fraction=1,
                                                        reserve_bytes=0, load_balance=actual_mode, tflops=ratings)
                self.assertIsInstance(operator, KernelLinearOperator)
                if mode == 'automatic':
                    estimate.assert_called_once()
                    self.assertEqual(memory_query.call_count, 4)
                    self.assertEqual(operator.tflops, (2.0, 4.0))
                    self.assertEqual(operator.tflops_source, 'benchmark')
                else:
                    estimate.assert_not_called()
                    self.assertEqual(memory_query.call_count, 2)
                    self.assertEqual(operator.tflops_source, 'provided' if ratings else None)
                self.assertEqual(sum(operator.storage_numel), 153)
                self.assertEqual(sum(operator.estimated_flops), 2 * 17**2)
                self.assertFalse(hasattr(operator, '_scratch'))
                matrix.zero_()
                vector = torch.randn(34, dtype=matrix.dtype)[::2]
                first = operator @ vector
                torch.testing.assert_close(first, expected @ vector)
                torch.testing.assert_close(operator(vector), first)
                torch.testing.assert_close(operator.matvec(vector), first)
                torch.testing.assert_close(operator @ (2 * vector), 2 * first)
                torch.testing.assert_close(first, expected @ vector)
                with self.assertRaises(ValueError):
                    operator @ torch.ones(17, 1, dtype=matrix.dtype)
                with self.assertRaises(TypeError):
                    operator @ torch.ones(17, dtype=torch.float32)
                operator._symv.close()

    def test_invalid_arguments(self):
        for options in ({'block_size': 0}, {'block_size': 1.5}, {'block_size': True},
                        {'load_balance': 'speed'},
                        {'tflops': [1]}, {'load_balance': 'compute', 'tflops': []},
                        {'load_balance': 'compute', 'tflops': [float('nan')]},
                        {'load_balance': 'compute', 'tflops': [0]},
                        {'load_balance': 'compute', 'tflops': [-1]},
                        {'load_balance': 'compute', 'tflops': [True]}):
            with self.assertRaises(ValueError):
                MultiGpuKernelLinearOperator(torch.eye(2), **options)
        with patch('torch.cuda.device_count', return_value=2):
            with self.assertRaisesRegex(ValueError, 'one rating'):
                MultiGpuKernelLinearOperator(torch.eye(2), load_balance='compute', tflops=[1])
        for dtype in (torch.int64, torch.float16, torch.bfloat16):
            with self.assertRaises(TypeError):
                MultiGpuKernelLinearOperator(torch.ones(2, 2, dtype=dtype))
        for matrix in (torch.ones(2, 3), torch.empty(0, 0), torch.ones(2)):
            with self.assertRaises(ValueError):
                MultiGpuKernelLinearOperator(matrix)

    def test_no_cuda_and_bad_devices(self):
        with patch('torch.cuda.device_count', return_value=0):
            with self.assertRaisesRegex(RuntimeError, 'at least one CUDA'):
                MultiGpuKernelLinearOperator(torch.eye(2))
        with patch('torch.cuda.device_count', return_value=2):
            for devices in ([0, 0], ['cpu'], [2]):
                with self.assertRaises(ValueError):
                    cuda_devices(devices)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
class CUDATests(unittest.TestCase):
    def test_single_gpu_all_dtypes_odd_even_and_nondefault_stream(self):
        for size in (1, 17, 18):
            for dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128):
                with torch.cuda.stream(torch.cuda.Stream(device=0)):
                    matrix = torch.randn(size, size, dtype=dtype, device='cuda:0')
                    matrix = matrix + matrix.T
                    expected = matrix.cpu()
                    operator = MultiGpuKernelLinearOperator(matrix, devices=[0], block_size=4)
                    matrix.zero_()
                    vector = torch.randn(size, dtype=dtype, device='cuda:0')
                    actual = operator @ vector
                    torch.testing.assert_close(actual, expected @ vector.cpu())
                    torch.testing.assert_close(operator @ (2 * vector), 2 * actual)
                    self.assertEqual(sum(operator.storage_numel), size * (size + 1) // 2)
                    operator._symv.close()

    @unittest.skipUnless(torch.cuda.device_count() >= 2, 'Two CUDA GPUs are required')
    def test_two_gpu_memory_and_compute(self):
        matrix = torch.randn(33, 33, dtype=torch.float64)
        matrix = matrix + matrix.T
        for mode in ('memory', 'compute'):
            ratings = [1, 3] if mode == 'compute' else None
            operator = MultiGpuKernelLinearOperator(matrix, devices=[1, 0], block_size=4,
                                                    load_balance=mode, tflops=ratings)
            self.assertEqual(operator.devices[0], torch.device('cuda:1'))
            self.assertEqual(len(operator.devices), 2)
            for device in ('cpu', 'cuda:0', 'cuda:1'):
                vector = torch.randn(33, dtype=torch.float64, device=device)
                torch.testing.assert_close(operator @ vector, matrix @ vector.cpu())
            operator._symv.close()

    @unittest.skipUnless(torch.cuda.device_count() >= 2, 'Two CUDA GPUs are required')
    def test_cross_device_matrix_and_vector(self):
        for source in (0, 1):
            matrix = torch.randn(17, 17, device=f'cuda:{source}', dtype=torch.float64)
            matrix = matrix + matrix.T
            expected = matrix.cpu()
            operator = MultiGpuKernelLinearOperator(matrix, devices=[1, 0], block_size=4)
            vector = torch.randn(17, device=f'cuda:{source}', dtype=torch.float64)
            torch.testing.assert_close(operator @ vector, expected @ vector.cpu())
            operator._symv.close()

    @unittest.skipUnless(torch.cuda.device_count() >= 3, 'Three CUDA GPUs are required')
    def test_three_gpu_compute(self):
        matrix = torch.randn(64, 64)
        matrix = matrix + matrix.T
        operator = MultiGpuKernelLinearOperator(matrix, devices=[0, 1, 2], block_size=4,
                                                load_balance='compute')
        vector = torch.randn(64)
        for source in ('cpu', 'cuda:0', 'cuda:1', 'cuda:2'):
            torch.testing.assert_close(operator @ vector.to(source), matrix @ vector)
        self.assertEqual(len(operator.devices), 3)
        self.assertEqual(operator.tflops_source, 'benchmark')
        self.assertTrue(all(rating > 0 for rating in operator.tflops))
        self.assertEqual(sum(operator.storage_numel), 64 * 65 // 2)
        operator._symv.close()
