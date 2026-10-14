import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from triton_lab.vector_add import vector_add  # noqa: E402


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA GPU")
class VectorAddTests(unittest.TestCase):
    def test_non_multiple_of_block_size(self):
        # 1003 elements checks that the final partial block is handled safely.
        x = torch.randn(1003, device="cuda", dtype=torch.float32)
        y = torch.randn_like(x)
        actual = vector_add(x, y, block_size=256)
        expected = x + y
        self.assertTrue(torch.allclose(actual, expected, rtol=1e-5, atol=1e-6))

    def test_empty_input(self):
        x = torch.empty(0, device="cuda", dtype=torch.float32)
        y = torch.empty_like(x)
        actual = vector_add(x, y)
        self.assertEqual(actual.numel(), 0)


if __name__ == "__main__":
    unittest.main()
