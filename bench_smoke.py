# Throwaway smoke bench: streamed logits on the cached Qwen3.8-27B fixture set.
# Run from repo root: CUDA_VISIBLE_DEVICES=-1 ./.venv/python.exe bench_smoke.py
import os
import time

os.environ.setdefault("HF_HOME", "D:/heretic/hf-cache")
os.environ.setdefault("HF_HUB_CACHE", "D:/heretic/hf-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from heretic.config import Settings
from heretic.model import Model
from heretic.utils import Prompt

SENT = "Summarize the environmental impact of large solar farms in a few sentences. "
PROMPTS = [Prompt(system="You are a helpful assistant.", user=SENT * k) for k in (8, 24, 48, 8, 24, 48)]

settings = Settings(model="Qwen/Qwen3.5-9B", stream_layers=True, batch_size=2)
model = Model(settings)

model.get_logits_batched(PROMPTS[:2])  # warm: pages in weights, inits caches
t0 = time.perf_counter()
out = model.get_logits_batched(PROMPTS)
dt = time.perf_counter() - t0
print(f"BENCH prompts={len(PROMPTS)} batch=2 rounds=3 time={dt:.2f}s shape={tuple(out.shape)}")
