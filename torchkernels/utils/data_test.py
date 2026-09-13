import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np
import torch

from torchkernels.utils.data import load_kmat, save_kmat, _DTYPES as DTYPES, _default_device


class KmatTest(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix"
            for dtype in DTYPES.values():
                for size in (0, 1, 7):
                    for triangle in (None, "upper", "lower"):
                        for compress in (False, True):
                            with self.subTest(dtype=dtype, size=size,
                                              triangle=triangle, compress=compress):
                                base = torch.arange(size * size).reshape(size, size)
                                matrix = (base + base.T).to(dtype).T
                                save_kmat(matrix, path, triangle=triangle,
                                          compress=compress)
                                restored = load_kmat(path, device=torch.device("cpu"))
                                torch.testing.assert_close(restored, matrix,
                                                           rtol=0, atol=0)
                                with np.load(path, allow_pickle=False) as archive:
                                    count = size * size if triangle is None else size * (size + 1) // 2
                                    self.assertEqual(archive["values"].nbytes,
                                                     count * matrix.element_size())
                                with ZipFile(path) as archive:
                                    expected = ZIP_DEFLATED if compress else ZIP_STORED
                                    self.assertTrue(all(info.compress_type == expected
                                                        for info in archive.infolist()))

    def test_selected_triangle_and_autograd(self):
        matrix = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                              requires_grad=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.npz"
            for triangle in ("upper", "lower"):
                save_kmat(matrix, path, triangle=triangle)
                half = matrix.triu() if triangle == "upper" else matrix.tril()
                expected = half + half.T - matrix.diag().diag()
                restored = load_kmat(path, device=torch.device("cpu"))
                torch.testing.assert_close(restored, expected, rtol=0, atol=0)
                self.assertFalse(restored.requires_grad)

    def test_complex_conjugate_view(self):
        matrix = torch.tensor([[1 + 2j, 3 + 4j], [3 + 4j, 5 + 6j]]).conj()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.npz"
            save_kmat(matrix, path)
            torch.testing.assert_close(load_kmat(path, device=torch.device("cpu")), matrix, rtol=0, atol=0)

    def test_invalid_input_and_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.npz"
            for matrix in (torch.zeros(3), torch.zeros(2, 2, 2)):
                with self.assertRaises(ValueError):
                    save_kmat(matrix, path)
            with self.assertRaises(ValueError):
                save_kmat(torch.zeros(2, 3), path, triangle="upper")
            with self.assertRaises(ValueError):
                save_kmat(torch.eye(2), path, triangle="diagonal")
            with self.assertRaises(TypeError):
                save_kmat(np.eye(2), path)
            np.savez(path, version=1, size=3, triangle="upper",
                     dtype="torch.float32", values=np.zeros(4, dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "Packed data"):
                load_kmat(path, device=torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_roundtrip(self):
        base = torch.randn(17, 17, device="cuda")
        matrix = (base + base.T).T
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.npz"
            for triangle in (None, "upper", "lower"):
                for compress in (False, True):
                    save_kmat(matrix, path, triangle=triangle, compress=compress)
                    restored = load_kmat(path, device=matrix.device)
                    torch.testing.assert_close(restored, matrix, rtol=0, atol=0)

    def test_full_storage_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.npz"
            for shape in ((3, 3), (2, 5), (0, 3), (3, 0)):
                matrix = torch.arange(shape[0] * shape[1], dtype=torch.float32)
                matrix = matrix.reshape(shape).T.requires_grad_()
                save_kmat(matrix, path)
                restored = load_kmat(path, device=torch.device("cpu"))
                torch.testing.assert_close(restored, matrix, rtol=0, atol=0)
                self.assertFalse(restored.requires_grad)
                with ZipFile(path) as archive:
                    self.assertTrue(all(info.compress_type == ZIP_STORED
                                        for info in archive.infolist()))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_mps_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.npz"
            matrix = torch.arange(25, dtype=torch.float32, device="mps").reshape(5, 5)
            matrix = matrix + matrix.T
            for triangle in (None, "upper", "lower"):
                save_kmat(matrix, path, triangle=triangle)
                restored = load_kmat(path, device=torch.device("mps"))
                torch.testing.assert_close(restored, matrix, rtol=0, atol=0)

    def test_device_selection(self):
        for cuda, mps, expected in ((True, True, "cuda"), (True, False, "cuda"),
                                    (False, True, "mps"), (False, False, "cpu")):
            with self.subTest(cuda=cuda, mps=mps):
                with patch("torch.cuda.is_available", return_value=cuda), patch(
                        "torch.backends.mps.is_available", return_value=mps):
                    self.assertEqual(_default_device(), torch.device(expected))

    def test_automatic_device_and_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.npz"
            for triangle in (None, "upper", "lower"):
                matrix = torch.eye(3)
                save_kmat(matrix, path, triangle=triangle)
                with patch("torchkernels.utils.data._default_device",
                           return_value=torch.device("cpu")) as select:
                    torch.testing.assert_close(load_kmat(path), matrix)
                    select.assert_called_once_with()
                with patch("torchkernels.utils.data._default_device") as select:
                    restored = load_kmat(path, device=torch.device("cpu"))
                    torch.testing.assert_close(restored, matrix)
                    select.assert_not_called()

    def test_available_default_device_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.npz"
            for triangle in (None, "upper", "lower"):
                matrix = torch.eye(3, device=_default_device())
                save_kmat(matrix, path, triangle=triangle)
                torch.testing.assert_close(load_kmat(path), matrix)
