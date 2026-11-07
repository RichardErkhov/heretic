from unittest import TestCase, main
from unittest.mock import patch

import torch
from transformers import BatchEncoding, LlamaConfig, LlamaForCausalLM

from heretic.config import Settings
from heretic.model import Model
from heretic.utils import Prompt


class StubTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, chats, add_generation_prompt, tokenize, strftime_now):
        return [" ".join(msg["content"] for msg in chat) for chat in chats]

    def __call__(self, texts, return_tensors, padding, return_token_type_ids):
        # Tiny deterministic character-level encoding, left-padded (heretic's convention).
        ids = [[min(ord(c), 10) + 1 for c in t[:8]] for t in texts]
        width = max(len(row) for row in ids)
        return BatchEncoding(
            {
                "input_ids": torch.tensor([[0] * (width - len(r)) + r for r in ids]),
                "attention_mask": torch.tensor(
                    [[0] * (width - len(r)) + [1] * len(r) for r in ids]
                ),
            }
        )


def make_model(stream: bool) -> Model:
    model = Model.__new__(Model)
    with patch("sys.argv", ["heretic"]):
        model.settings = Settings(model="stub", stream_layers=stream)
    model.supports_system_role = True
    model.tokenizer = StubTokenizer()
    model.stream_fallback_reason = None
    config = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=16,
        max_position_embeddings=32,
    )
    torch.manual_seed(0)
    model.model = LlamaForCausalLM(config)
    model.model.eval()
    return model


class StreamedLayersTests(TestCase):
    prompts = [
        Prompt(system="", user="hello"),
        Prompt(system="", user="a somewhat longer prompt"),
    ]

    def test_streamed_residuals_match_eager(self) -> None:
        eager = make_model(False).get_residuals(self.prompts)
        streamed = make_model(True).get_residuals(self.prompts)
        self.assertEqual(eager.shape, streamed.shape)
        self.assertTrue(torch.allclose(eager, streamed, atol=1e-4, rtol=1e-4))

    def test_streamed_logits_match_eager(self) -> None:
        eager = make_model(False).get_logits(self.prompts)
        streamed = make_model(True).get_logits(self.prompts)
        self.assertTrue(torch.allclose(eager, streamed, atol=1e-4, rtol=1e-4))
        # Greedy decoding only sees the argmax, so small drift that allclose tolerates
        # must still never flip the chosen token.
        self.assertTrue(torch.equal(eager.argmax(-1), streamed.argmax(-1)))


if __name__ == "__main__":
    main()
