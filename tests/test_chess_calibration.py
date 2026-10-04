import math
import tempfile
import unittest
from pathlib import Path

from chess_finetune.calibration import fit_temperature, log_probs, metrics
from chess_finetune.data import audit, read_split


class CalibrationTests(unittest.TestCase):
    def test_known_distribution(self):
        rows = [dict(scores=[math.log(.8), math.log(.2)], label=i) for i in [0]*8 + [1]*2]
        m = metrics(rows)
        self.assertAlmostEqual(m['accuracy'], .8)
        self.assertAlmostEqual(m['ece'], 0.)
        self.assertAlmostEqual(m['brier'], .32)
        self.assertAlmostEqual(m['nll'], -.8*math.log(.8)-.2*math.log(.2))
        self.assertAlmostEqual(sum(math.exp(p) for p in log_probs([10000, 9999])), 1.)

    def test_overconfidence_fit_preserves_decision(self):
        rows = [dict(scores=[math.log(.99), math.log(.01)], label=i) for i in [0]*7 + [1]*3]
        t = fit_temperature(rows)
        self.assertAlmostEqual(t, math.log(99)/math.log(7/3), places=4)
        self.assertLess(metrics(rows, t)['nll'], metrics(rows)['nll'])
        self.assertEqual(metrics(rows, t)['accuracy'], metrics(rows)['accuracy'])
        self.assertAlmostEqual(metrics(rows, t)['ece'], 0., places=6)

    def test_uniform_and_forced(self):
        rows = [dict(scores=[0, 0], label=0), dict(scores=[0], label=0)]
        m = metrics(rows)
        self.assertEqual(m['forced_positions'], 1)
        self.assertTrue(math.isfinite(fit_temperature(rows)))
        self.assertEqual(metrics([dict(scores=[0, 0], label=0)]*4)['selective'][0]['coverage'], 1.)

    def test_invalid_inputs(self):
        for scores in ([], [float('nan')], [float('inf')]):
            with self.assertRaises(ValueError):
                log_probs(scores)
        with self.assertRaises(ValueError):
            metrics([])
        with self.assertRaises(ValueError):
            metrics([dict(scores=[0], label=-1)])
        with self.assertRaises(ValueError):
            log_probs([0], 0)

    def test_overlap(self):
        result = audit({'train': [dict(position='x', puzzle_id='a')],
                        'test': [dict(position='x', puzzle_id='b')]})
        self.assertEqual(result['overlaps'][0]['field'], 'position')
        self.assertFalse(result['game_metadata_complete'])

    def test_pipeline_fits_only_validation(self):
        import json
        import sys
        import types
        from unittest.mock import patch
        from chess_finetune.run import main
        class FakeScorer:
            def __init__(self, *args, **kwargs):
                pass
            def score(self, prompt, candidates, norm):
                return [types.SimpleNamespace(score=x) for x in (math.log(.9), math.log(.1))]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for split, turn, targets in [('train', 'w', ['a']), ('valid', 'b', ['a', 'b']), ('test', 'w', ['b'])]:
                fen = f'8/8/8/8/8/8/8/8 {turn} {"K" if split == "test" else "-"} - 0 1'
                rows = [dict(prompt=f'FEN: {fen}\nLegal moves: a b', completion=t) for t in targets]
                (root/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            argv = ['run', '--data', d, '--base-only', '--model', d, '--out', str(root/'out')]
            with patch.object(sys, 'argv', argv), patch.dict(sys.modules, {'jev_pakkio.scorer': types.SimpleNamespace(OptionScorer=FakeScorer)}):
                main()
            fit = json.loads((root/'out/base-calibrator.json').read_text())
            report = json.loads((root/'out/metrics.json').read_text())
            self.assertEqual(fit['fit_n'], 2)
            self.assertEqual(fit['fit_split'], 'valid')
            self.assertEqual(report['base']['raw']['n'], 1)
            self.assertTrue((root/'out/report.md').exists())

    def test_bad_data_has_line_context(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'test.jsonl'
            path.write_text('{}\n')
            with self.assertRaisesRegex(ValueError, 'test.jsonl:1'):
                read_split(path)


if __name__ == '__main__':
    unittest.main()
