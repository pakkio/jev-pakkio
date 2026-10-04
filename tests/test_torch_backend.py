"""Offline numerical checks using random, tiny Gemma weights (no downloads)."""
import math
import unittest

import torch
from transformers import Gemma3TextConfig, Gemma3ForCausalLM

from jev_pakkio.scorer import OptionScorer
from jev_pakkio.torch_backend import TorchBackend


class Tokenizer:
    bos_token_id = 1
    pad_token_id = 0

    def encode(self, text, add_special_tokens=True):
        return ([1] if add_special_tokens else []) + [2 + ord(c) % 29 for c in text]

    def apply_chat_template(self, messages, **kwargs):
        return 'user:' + messages[0]['content'] + '\nassistant:'


class TorchScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        config = Gemma3TextConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                                 num_hidden_layers=2, num_attention_heads=4,
                                 num_key_value_heads=2, head_dim=8, sliding_window=8,
                                 max_position_embeddings=256,
                                 layer_types=['sliding_attention', 'full_attention'])
        engine = TorchBackend.__new__(TorchBackend)
        engine.torch = torch
        engine.device = torch.device('cpu')
        engine.model = Gemma3ForCausalLM(config).eval()
        engine.tok = Tokenizer()
        cls.engine = engine

    def scorer(self, batch_size=2):
        s = OptionScorer.__new__(OptionScorer)
        s._engine = self.engine
        s.model, s.tok = self.engine.model, self.engine.tok
        s.backend, s.device = 'torch', 'cpu'
        s.batch_size, s.chat, s.sep = batch_size, False, ''
        s.pad_id, s.bos_id, s.last_timing = 0, 1, {}
        return s

    def test_cached_matches_naive_and_batch_sizes(self):
        options = ['a', 'longer option', 'xyz', 'q']
        for context in ['hi', 'context longer than the sliding window']:
            expected = self.scorer().score_naive(context, options)
            for batch_size in [1, 2, 8]:
                with self.subTest(context=context, batch_size=batch_size):
                    actual = self.scorer(batch_size).score(context, options, norm='sum')
                    torch.testing.assert_close(torch.tensor([x.logprob_sum for x in actual]),
                                               torch.tensor(expected), atol=2e-5, rtol=1e-5)
                    self.assertAlmostEqual(sum(x.probability for x in actual), 1)

    def test_normalization_and_repeated_calls(self):
        s = self.scorer()
        options = ['x', 'long option']
        for norm in ['sum', 'mean', 'pmi']:
            a = s.score('context', options, norm=norm)
            b = s.score('context', options, norm=norm)
            self.assertEqual(a, b)
            for r in a:
                expected = r.logprob_sum if norm == 'sum' else (
                    r.logprob_mean if norm == 'mean' else r.logprob_sum - r.logprob_uncond)
                self.assertAlmostEqual(r.score, expected)
                self.assertTrue(math.isfinite(r.score))

    def test_chat_and_separator(self):
        s = self.scorer()
        for chat in [False, True]:
            s.chat, s.sep = chat, '\nAnswer: '
            actual = s.score('hello', ['a', 'bbb'])
            expected = s.score_naive('hello', ['a', 'bbb'])
            for a, b in zip(actual, expected):
                self.assertAlmostEqual(a.logprob_sum, b, places=4)

    def test_load_multimodal_checkpoint(self):
        import tempfile
        from unittest.mock import patch
        from transformers import Gemma3Config, Gemma3ForConditionalGeneration, SiglipVisionConfig
        config = Gemma3Config(
            text_config=self.engine.model.config.to_dict(),
            vision_config=SiglipVisionConfig(hidden_size=32, intermediate_size=64,
                num_hidden_layers=1, num_attention_heads=4, image_size=16, patch_size=8).to_dict(),
            mm_tokens_per_image=4,
        )
        original = Gemma3ForConditionalGeneration(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            original.save_pretrained(directory)
            with patch('transformers.AutoTokenizer.from_pretrained', return_value=Tokenizer()):
                loaded = TorchBackend(directory, 'cpu', None)
            ids = self.engine._tensor([[1, 4, 5]])
            with torch.inference_mode():
                expected = original(input_ids=ids).logits
                actual = loaded.model(input_ids=ids).logits
            torch.testing.assert_close(actual, expected)
            prefix, last = loaded.prefill([1, 4, 5])
            cached = loaded.score_with_prefix(prefix, last, [[6], [7, 8, 9]], 2, 0)
            naive = loaded.score_naive([1, 4, 5], [[6], [7, 8, 9]])
            torch.testing.assert_close(torch.tensor(cached), torch.tensor(naive))

    def test_validation(self):
        s = self.scorer()
        for options, norm in [(['a'], 'mean'), (['a', ''], 'mean'), (['a', 'b'], 'unknown')]:
            with self.assertRaises(ValueError):
                s.score('hello', options, norm=norm)
        s.bos_id = None
        with self.assertRaisesRegex(ValueError, 'BOS'):
            s.score('hello', ['a', 'b'], norm='pmi')


if __name__ == '__main__':
    unittest.main()
