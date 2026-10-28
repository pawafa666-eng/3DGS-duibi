import sys
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runtime'))
from scene.gaussian_model import Codebook, GaussianModel


def model():
    torch.manual_seed(9)
    n = 12
    g = GaussianModel(3)
    g._xyz = torch.linspace(-12, 12, n * 3).reshape(n, 3)
    g._degrees = torch.arange(n).remainder(4).reshape(n, 1)
    names = ['features_dc'] + [f'features_rest_{i}' for i in range(15)] + ['opacity', 'scaling', 'rotation_re', 'rotation_im']
    dims = [(n, 1, 3)] + [(n, 3)] * 15 + [(n, 1), (n, 3), (n, 1), (n, 3)]
    g._codebook_dict = OrderedDict((name, Codebook(torch.randint(0, 256, shape, dtype=torch.uint8), torch.linspace(-3.25, 3.75, 256).reshape(-1, 1) + i / 100)) for i, (name, shape) in enumerate(zip(names, dims)))
    return g


class HalfSaveTest(unittest.TestCase):
    def test_quantised_half_roundtrip(self):
        g = model()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'half.ply'
            g.save_ply(str(p), quantised=True, half_float=True)
            ply = PlyData.read(str(p))
            for i in range(4):
                element = ply[f'vertex_{i}']
                expected = g._xyz[(g._degrees == i).squeeze()].numpy().astype(np.float16).view(np.int16)
                for j, name in enumerate(('x', 'y', 'z')):
                    np.testing.assert_array_equal(element[name], expected[:, j])
            for name, cb in g._codebook_dict.items():
                np.testing.assert_array_equal(ply['codebook_centers'][name], cb.centers.numpy().astype(np.float16).view(np.int16).ravel())

class RecoveryTest(unittest.TestCase):
    def test_recovered_equals_runtime_half_save(self):
        from recover_quantised_half import recover
        g = model()
        with tempfile.TemporaryDirectory() as tmp:
            source, expected, recovered = (Path(tmp) / x for x in ('quant.ply', 'expected.ply', 'recovered.ply'))
            g.save_ply(str(source), quantised=True)
            g.save_ply(str(expected), quantised=True, half_float=True)
            recover(source, recovered)
            a, b = PlyData.read(str(expected)), PlyData.read(str(recovered))
            self.assertEqual([x.name for x in a.elements], [x.name for x in b.elements])
            for x, y in zip(a.elements, b.elements):
                self.assertEqual(x.data.dtype, y.data.dtype)
                np.testing.assert_array_equal(x.data, y.data)

    def test_refuses_overwrite_and_original_output(self):
        from recover_quantised_half import recover
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source.ply'
            model().save_ply(str(source), quantised=True)
            destination = Path(tmp) / 'existing.ply'
            destination.write_bytes(b'keep')
            with self.assertRaises(FileExistsError):
                recover(source, destination)
            self.assertEqual(destination.read_bytes(), b'keep')
            with self.assertRaises(ValueError):
                recover(source, ROOT.parent / 'output' / '09_test_forbidden.ply')

    def test_rejects_invalid_input(self):
        from recover_quantised_half import recover
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'wrong.ply'
            model().save_ply(str(source), quantised=True, half_float=True)
            with self.assertRaises(ValueError):
                recover(source, Path(tmp) / 'out.ply')
            self.assertFalse((Path(tmp) / 'out.ply').exists())

    def test_rejects_overflow_before_creating_output(self):
        from recover_quantised_half import recover
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source.ply'
            g = model()
            g._xyz[0, 0] = 70000
            g.save_ply(str(source), quantised=True)
            target = Path(tmp) / 'half.ply'
            with self.assertRaises(ValueError):
                recover(source, target)
            self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
