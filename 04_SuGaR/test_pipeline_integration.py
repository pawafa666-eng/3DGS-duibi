import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


RUNTIME = Path(__file__).resolve().parent / 'runtime'


class PipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='sugar test 空格 ')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / 'runtime with space'
        self.repo.mkdir()
        shutil.copy(RUNTIME / 'train_full_pipeline.py', self.repo)
        shutil.copy(RUNTIME / 'train.py', self.repo)
        (self.repo / 'sugar_utils').mkdir()
        (self.repo / 'sugar_utils' / 'general_utils.py').write_text(
            "def str2bool(v): return str(v).lower() in ('true','1')\n", encoding='utf-8')
        (self.repo / 'gaussian_splatting').mkdir()
        (self.repo / 'gaussian_splatting' / 'train.py').write_text('''
import json, os, pathlib, sys
a = sys.argv[1:]
pathlib.Path(os.environ['CALLS']).write_text(json.dumps(a), encoding='utf-8')
if os.environ.get('FAIL_VANILLA'): sys.exit(23)
p = pathlib.Path(a[a.index('-m')+1]) / 'point_cloud/iteration_7000/point_cloud.ply'
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text('model')
''', encoding='utf-8')
        for package in ('sugar_trainers', 'sugar_extractors'):
            (self.repo / package).mkdir()
        coarse = '''
import os
from pathlib import Path
def run(args):
    assert args.checkpoint_path.endswith(os.sep), 'SuGaR camera loader requires trailing separator'
    if os.environ.get('FAIL_COARSE'): raise RuntimeError('coarse failed')
    p=Path(args.output_dir)/'coarse'/ '15000.pt'; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('coarse'); return str(p)
'''
        for module, symbol in [('coarse_density', 'coarse_training_with_density_regularization'),
                               ('coarse_sdf', 'coarse_training_with_sdf_regularization'),
                               ('coarse_density_and_dn_consistency', 'coarse_training_with_density_regularization_and_dn_consistency')]:
            (self.repo / 'sugar_trainers' / (module + '.py')).write_text(coarse + symbol + '=run\n', encoding='utf-8')
        (self.repo / 'sugar_extractors' / 'coarse_mesh.py').write_text('''
from pathlib import Path
def extract_mesh_from_coarse_sugar(args):
    p=Path(args.mesh_output_dir)/'mesh.ply'; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('mesh'); return [str(p)]
''', encoding='utf-8')
        (self.repo / 'sugar_trainers' / 'refine.py').write_text('''
import os
from pathlib import Path
def refined_training(args):
    assert args.eval is False
    if os.environ.get('FAIL_REFINE'): raise RuntimeError('refine failed')
    if os.environ.get('MISSING_REFINE'): return str(Path(args.output_dir)/'missing.pt')
    p=Path(args.output_dir)/'fine'/f'{args.refinement_iterations}.pt'; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('fine')
    if args.export_ply:
        q=Path(args.refined_ply_dir)/'fine.ply'; q.parent.mkdir(parents=True,exist_ok=True); q.write_text('ply')
    return str(p)
''', encoding='utf-8')
        (self.repo / 'sugar_extractors' / 'refined_mesh.py').write_text('''
from pathlib import Path
def extract_mesh_and_texture_from_refined_sugar(args):
    assert Path(args.coarse_mesh_path).is_file()
    p=Path(args.mesh_output_dir)/'mesh.obj'; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('obj'); return str(p)
''', encoding='utf-8')
        self.data = self.root / 'scene data'
        self.data.mkdir()
        self.output = self.root / 'result folder'
        self.calls = self.root / 'calls.json'

    def run_pipeline(self, **env_overrides):
        env = os.environ.copy()
        env.update(CALLS=str(self.calls), **env_overrides)
        return subprocess.run([sys.executable, str(self.repo/'train_full_pipeline.py'), '-s', str(self.data),
            '-r', 'dn_consistency', '--output_dir', str(self.output), '--train_only', '--eval', 'False',
            '--low_poly', 'True', '--refinement_time', 'short', '--export_ply', 'True', '--export_obj', 'True'],
            cwd=self.repo, env=env, capture_output=True, text=True)

    def test_complete_pipeline_and_arguments_with_spaces(self):
        result = self.run_pipeline()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(self.calls.read_text(encoding='utf-8'))
        self.assertEqual(args[args.index('-s')+1], str(self.data))
        self.assertEqual(args[args.index('--test_iterations')+1], '-1')
        self.assertEqual(args[args.index('--iterations')+1], '7000')
        manifests = list(self.output.rglob('pipeline_results.json'))
        self.assertEqual(len(manifests), 1)
        manifest = json.loads(manifests[0].read_text(encoding='utf-8'))
        for stage in ('vanilla', 'coarse', 'coarse_mesh', 'refined', 'refined_ply', 'refined_mesh'):
            self.assertTrue(Path(manifest['artifacts'][stage]).is_file(), stage)
        self.assertEqual(manifest['refinement_iterations'], 2000)

    def test_vanilla_failure_stops_and_is_propagated(self):
        result = self.run_pipeline(FAIL_VANILLA='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('23', result.stderr)
        self.assertFalse(list(self.output.rglob('pipeline_results.json')))

    def test_refinement_failure_is_not_success(self):
        result = self.run_pipeline(FAIL_REFINE='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('refine failed', result.stderr)
        self.assertFalse(list(self.output.rglob('pipeline_results.json')))

    def test_missing_final_model_is_not_success(self):
        result = self.run_pipeline(MISSING_REFINE='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('refined artifact missing', result.stderr)
        self.assertFalse(list(self.output.rglob('pipeline_results.json')))

    def test_reused_vanilla_checkpoint_skips_initial_training(self):
        checkpoint = self.root / 'existing model' / 'point_cloud' / 'iteration_7000' / 'point_cloud.ply'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_text('old model', encoding='utf-8')
        env = os.environ.copy()
        env['CALLS'] = str(self.calls)
        result = subprocess.run([sys.executable, str(self.repo/'train_full_pipeline.py'), '-s', str(self.data),
            '-r', 'density', '--gs_output_dir', str(checkpoint.parents[2]), '--output_dir', str(self.output),
            '--train_only', '--refinement_time', 'short', '--export_ply', 'True', '--export_obj', 'False'],
            cwd=self.repo, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.calls.exists())
        manifest = json.loads(next(self.output.rglob('pipeline_results.json')).read_text(encoding='utf-8'))
        self.assertEqual(Path(manifest['artifacts']['vanilla']), checkpoint)
        self.assertNotIn('refined_mesh', manifest['artifacts'])

    def test_experiment_uses_controller_output_and_no_eval(self):
        config = json.loads((RUNTIME.parent / 'experiment.json').read_text(encoding='utf-8'))
        command = config['train']['commands'][0]
        self.assertEqual(command[command.index('--output_dir')+1], '{train_output}')
        self.assertIn('--train_only', command)
        self.assertEqual(command[command.index('--eval')+1], 'False')
        self.assertNotIn('--iterations', command)


if __name__ == '__main__':
    unittest.main()
