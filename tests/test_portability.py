import subprocess
import sys
import unittest
from unittest.mock import patch


class PortabilityTests(unittest.TestCase):
    def test_import_and_help_without_mlx(self):
        script = '''
import sys
class NoMLX:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('mlx', 'mlx_lm'):
            raise ImportError('MLX deliberately unavailable')
sys.meta_path.insert(0, NoMLX())
import jev_pakkio
from jev_pakkio.cli import main
sys.argv = ['jev_pakkio', 'serve', '--help']
main()
'''
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--backend', result.stdout)
        self.assertIn('--device', result.stdout)

    def test_serve_cli_forwards_device(self):
        from jev_pakkio.cli import main
        with patch.object(sys, 'argv', ['jev_pakkio', 'serve', '--backend', 'torch', '--device', 'cpu']), patch('jev_pakkio.server.serve') as serve:
            main()
        self.assertEqual(serve.call_args.kwargs['backend'], 'torch')
        self.assertEqual(serve.call_args.kwargs['device'], 'cpu')

    def test_server_forwards_backend(self):
        from fastapi.testclient import TestClient
        from jev_pakkio.server import create_app
        with patch('jev_pakkio.server._get_scorer_class') as get_scorer:
            scorer = get_scorer.return_value
            scorer.return_value.backend = 'torch'
            scorer.return_value.device = 'cpu'
            with TestClient(create_app(model_path='test-model', backend='torch', device='cpu')) as client:
                self.assertEqual(client.get('/health').status_code, 200)
            self.assertEqual(get_scorer.call_args.args[0], 'torch')
            self.assertEqual(scorer.call_args.kwargs['device'], 'cpu')


if __name__ == '__main__':
    unittest.main()
