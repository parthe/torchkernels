"""Construction dispatch and single-device kernel products."""

import unittest
from unittest.mock import patch

import torch

from .linear_operator import KernelLinearOperator
from ..utils import MultiGpuKernelLinearOperator


class CustomOperator(KernelLinearOperator):
    def __init__(self, scale):
        self.scale = scale

    def matvec(self, vector):
        return self.scale * vector


class ConstructionTests(unittest.TestCase):
    def test_cpu_snapshot_upper_triangle_and_dtype(self):
        for dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128):
            matrix = torch.randn(7, 7, dtype=dtype).T
            matrix[torch.ones(7, 7, dtype=torch.bool).tril(-1)] = float('nan')
            expected = matrix.triu() + matrix.triu(1).T
            with patch('torch.cuda.device_count', return_value=0):
                operator = KernelLinearOperator(matrix)
            self.assertIs(type(operator), KernelLinearOperator)
            self.assertEqual(operator.device, torch.device('cpu'))
            self.assertEqual(operator.shape, (7, 7))
            self.assertEqual(operator._packed.numel(), 28)
            self.assertEqual(operator.load_balance, 'memory')
            matrix.zero_()
            vector = torch.randn(14, dtype=dtype)[::2]
            first = operator @ vector
            torch.testing.assert_close(first, expected @ vector)
            torch.testing.assert_close(operator(vector), first)
            torch.testing.assert_close(operator.matvec(2 * vector), 2 * first)
            torch.testing.assert_close(first, expected @ vector)
            self.assertFalse(first.requires_grad)

    def test_multiple_visible_cuda_devices_dispatch_once(self):
        matrix = torch.eye(2)
        with patch('torch.cuda.device_count', return_value=3), \
                patch.object(MultiGpuKernelLinearOperator, '__init__', return_value=None) as init:
            operator = KernelLinearOperator(matrix, load_balance='compute', block_size=4)
        self.assertIs(type(operator), MultiGpuKernelLinearOperator)
        self.assertIsInstance(operator, KernelLinearOperator)
        init.assert_called_once_with(matrix, devices=tuple(torch.device('cuda', i) for i in range(3)),
                                     load_balance='compute', block_size=4)

    def test_explicit_selection_and_generator(self):
        matrix = torch.eye(2)
        with patch('torch.cuda.device_count', return_value=3), \
                patch.object(MultiGpuKernelLinearOperator, '__init__', return_value=None) as init:
            operator = KernelLinearOperator(matrix, devices=iter([2, 0]))
        self.assertIs(type(operator), MultiGpuKernelLinearOperator)
        init.assert_called_once_with(matrix, devices=(torch.device('cuda:2'), torch.device('cuda:0')))

    def test_one_visible_or_selected_gpu_uses_base(self):
        matrix = torch.eye(2)
        for count, kwargs, expected in ((1, {}, 0), (3, {'devices': [2]}, 2)):
            with patch('torch.cuda.device_count', return_value=count), \
                    patch.object(KernelLinearOperator, '__init__', return_value=None) as init:
                operator = KernelLinearOperator(matrix, **kwargs)
            self.assertIs(type(operator), KernelLinearOperator)
            init.assert_called_once_with(matrix, devices=(torch.device('cuda', expected),))

    def test_subclass_not_redispatched(self):
        with patch('torch.cuda.device_count', return_value=3):
            operator = CustomOperator(2)
        self.assertIs(type(operator), CustomOperator)
        torch.testing.assert_close(operator @ torch.ones(2), torch.full((2,), 2.0))

    def test_single_device_validation(self):
        with patch('torch.cuda.device_count', return_value=0):
            with self.assertRaises(ValueError):
                KernelLinearOperator(torch.ones(2, 3))
            with self.assertRaises(ValueError):
                KernelLinearOperator(torch.eye(2), load_balance='invalid')
            operator = KernelLinearOperator(torch.eye(2), load_balance='compute')
            self.assertIsNone(operator.tflops_source)
        with self.assertRaises(ValueError):
            operator @ torch.ones(2, 1)
        with self.assertRaises(TypeError):
            operator @ torch.ones(2, dtype=torch.float64)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
class CUDADispatchTests(unittest.TestCase):
    def test_single_device_override(self):
        matrix = torch.randn(17, 17, dtype=torch.float64)
        matrix = matrix + matrix.T
        operator = KernelLinearOperator(matrix, devices=[0])
        self.assertIs(type(operator), KernelLinearOperator)
        self.assertEqual(operator.device, torch.device('cuda:0'))
        self.assertEqual(operator._packed.numel(), 17 * 18 // 2)
        for device in ('cpu', 'cuda:0'):
            vector = torch.randn(17, dtype=matrix.dtype, device=device)
            torch.testing.assert_close(operator @ vector, matrix @ vector.cpu())

    @unittest.skipUnless(torch.cuda.device_count() >= 2, 'Multiple CUDA GPUs are required')
    def test_automatic_dispatch_real_cuda(self):
        matrix = torch.randn(33, 33, dtype=torch.float64)
        matrix = matrix + matrix.T
        for mode in ('memory', 'compute'):
            operator = KernelLinearOperator(matrix, load_balance=mode, block_size=4)
            self.assertIs(type(operator), MultiGpuKernelLinearOperator)
            self.assertEqual(len(operator.devices), torch.cuda.device_count())
            vector = torch.randn(33, dtype=matrix.dtype)
            torch.testing.assert_close(operator @ vector, matrix @ vector)
            operator._symv.close()
