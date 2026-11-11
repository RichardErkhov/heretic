# Throwaway profile of the timed streaming pass on Qwen/Qwen3.5-9B (CPU).
# Run from repo root: CUDA_VISIBLE_DEVICES=-1 ./.venv/python.exe bench_profile.py
import os
import time
from collections import defaultdict

os.environ.setdefault("HF_HOME", "D:/heretic/hf-cache")
os.environ.setdefault("HF_HUB_CACHE", "D:/heretic/hf-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import re
from torch.profiler import ProfilerActivity, profile

from heretic.config import Settings
from heretic.model import Model
from heretic.utils import Prompt

SENT = "Summarize the environmental impact of large solar farms in a few sentences. "
PROMPTS = [Prompt(system="You are a helpful assistant.", user=SENT * k) for k in (8, 24, 48, 8, 24, 48)]

settings = Settings(model="Qwen/Qwen3.5-9B", stream_layers=True, batch_size=2)
model = Model(settings)
model.get_logits_batched(PROMPTS[:2])  # warm: same as bench_smoke.py

# Phase buckets via forward hooks on the inner transformers module.
# Only decoder blocks, embeddings, final norm and lm_head are hooked, so nested
# modules don't double-count; anything left over is "outside" (stream/evict/logits).
cat = defaultdict(float)
block_re = re.compile(r"(^|\.)layers\.\d+$")

def pre(module, inputs):
    module._t0 = time.perf_counter()



def post(module, inputs, outputs):
    cat[module._bucket] += time.perf_counter() - module._t0


for name, m in model.model.named_modules():
    if block_re.search(name):
        bucket = "layers"
    elif name.endswith("embed_tokens"):
        bucket = "embed"
    elif name.endswith(".norm") or name == "norm":
        bucket = "final_norm"
    elif name.endswith("lm_head"):
        bucket = "lm_head"
    else:
        continue
    m._bucket = bucket
    m.register_forward_pre_hook(pre)
    m.register_forward_hook(post)

t0 = time.perf_counter()
with profile(activities=[ProfilerActivity.CPU]) as prof:
    out = model.get_logits_batched(PROMPTS)
wall = time.perf_counter() - t0

inside = sum(cat.values())
print(f"WALL {wall:.1f}s  inside-hooks {inside:.1f}s  outside {wall - inside:.1f}s")
for k, v in sorted(cat.items(), key=lambda kv: -kv[1]):
    print(f"  {k:12s} {v:7.1f}s")
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12))
