"""Packed symmetric storage on one device, independent of CUDA dispatch."""

import unittest
from unittest.mock import patch

import torch

from .packed import SymmetricLinearOperator
from . import KernelLinearOperator, MultiGpuKernelLinearOperator


class PackedTests(unittest.TestCase):
    def test_hierarchy(self):
        self.assertTrue(issubclass(KernelLinearOperator, SymmetricLinearOperator))
        self.assertTrue(issubclass(MultiGpuKernelLinearOperator, KernelLinearOperator))
        self.assertTrue(issubclass(MultiGpuKernelLinearOperator, SymmetricLinearOperator))

    def test_exact_storage_and_symmetric_products(self):
        for size in (1, 2, 7, 18):
            for dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128):
                for block_size in (1, 3, 1024):
                    matrix = torch.randn(size, size, dtype=dtype).T
                    matrix[torch.ones(size, size, dtype=torch.bool).tril(-1)] = float('nan')
                    expected = matrix.triu() + matrix.triu(1).T
                    operator = SymmetricLinearOperator(matrix, block_size=block_size)
                    matrix.zero_()
                    self.assertEqual(operator._packed.numel(), size * (size + 1) // 2)
                    self.assertEqual(operator.storage_numel, (size * (size + 1) // 2,))
                    self.assertFalse(hasattr(operator, '_matrix'))
                    self.assertFalse(hasattr(operator, '_scratch'))
                    vector = torch.randn(2 * size, dtype=dtype)[::2]
                    actual = operator @ vector
                    torch.testing.assert_close(actual, expected @ vector)
                    torch.testing.assert_close(operator(2 * vector), 2 * actual)
                    torch.testing.assert_close(operator.matvec(vector), actual)
                    torch.testing.assert_close(actual, expected @ vector)

    def test_explicit_cpu_does_not_dispatch_or_initialize_cuda(self):
        with patch('torch.cuda.device_count', return_value=3), \
                patch('torch.cuda.Stream') as stream:
            operator = SymmetricLinearOperator(torch.eye(3), device='cpu')
        self.assertIs(type(operator), SymmetricLinearOperator)
        stream.assert_not_called()
        torch.testing.assert_close(operator @ torch.ones(3), torch.ones(3))

    def test_validation_and_no_autograd(self):
        with self.assertRaises(ValueError):
            SymmetricLinearOperator(torch.ones(2, 3))
        with self.assertRaises(ValueError):
            SymmetricLinearOperator(torch.eye(2), block_size=0)
        operator = SymmetricLinearOperator(torch.eye(2, requires_grad=True))
        with self.assertRaises(ValueError):
            operator @ torch.ones(2, 1)
        with self.assertRaises(TypeError):
            operator @ torch.ones(2, dtype=torch.float64)
        result = operator @ torch.ones(2, requires_grad=True)
        self.assertFalse(result.requires_grad)
        self.assertFalse(operator._packed.requires_grad)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
class PackedCUDATests(unittest.TestCase):
    def test_single_device_packing_all_dtypes(self):
        for dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128):
            matrix = torch.randn(17, 17, dtype=dtype)
            matrix = matrix + matrix.T
            for index in range(torch.cuda.device_count()):
                with torch.cuda.stream(torch.cuda.Stream(device=index)):
                    operator = SymmetricLinearOperator(matrix, device=f'cuda:{index}', block_size=4)
                    self.assertEqual(operator._packed.numel(), 17 * 18 // 2)
                    vector = torch.randn(17, dtype=dtype, device=f'cuda:{index}')
                    expected = matrix @ vector.cpu()
                    first = operator @ vector
                    torch.testing.assert_close(first, expected)
                    torch.testing.assert_close(operator @ (2 * vector), 2 * expected)
                    torch.testing.assert_close(first, expected)
                    operator._symv.close()

    @unittest.skipUnless(torch.cuda.device_count() >= 2, 'Two CUDA GPUs are required')
    def test_cross_device_source_and_vector(self):
        matrix = torch.randn(13, 13, dtype=torch.float64, device='cuda:0')
        matrix = matrix + matrix.T
        operator = SymmetricLinearOperator(matrix, device='cuda:1', block_size=3)
        vector = torch.randn(13, dtype=matrix.dtype, device='cuda:0')
        torch.testing.assert_close(operator @ vector, matrix.cpu() @ vector.cpu())
        operator._symv.close()
