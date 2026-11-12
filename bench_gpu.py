# Throwaway GPU comparison: streamed vs eager logits on the real-size fixture model.
# Run TWICE from repo root (each mode isolated in its own process), GPU visible:
#   ./.venv/python.exe bench_gpu.py streamed   |   ./.venv/python.exe bench_gpu.py eager
import os
import re
import sys
import time
from collections import defaultdict

os.environ.setdefault("HF_HOME", "D:/heretic/hf-cache")
os.environ.setdefault("HF_HUB_CACHE", "D:/heretic/hf-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from psutil import Process

from heretic.config import Settings
from heretic.model import Model
from heretic.utils import Prompt

SENT = "Summarize the environmental impact of large solar farms in a few sentences. "
PROMPTS = [Prompt(system="You are a helpful assistant.", user=SENT * k) for k in (8, 24, 48, 8, 24, 48)]

streamed = sys.argv[1] == "streamed"
sys.argv = [sys.argv[0]] + sys.argv[2:]  # remaining flags (e.g. --quantization) go to Settings
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name} {p.total_memory / 1024**3:.1f} GB")

cat = defaultdict(float)
block_re = re.compile(r"(^|\.)layers\.\d+$")


def pre(module, inputs):
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # module-boundary sync so bucket times reflect GPU completion
    module._t0 = time.perf_counter()


def post(module, inputs, outputs):
    cat[module._bucket] += time.perf_counter() - module._t0


t_load0 = time.perf_counter()
model = Model(Settings(model="Qwen/Qwen3.5-9B", stream_layers=streamed, batch_size=2))
t_load = time.perf_counter() - t_load0
rss_after_load = Process().memory_info().rss / 1024**3

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

model.get_logits_batched(PROMPTS[:2])  # warm
if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
cat.clear()  # keep warm-pass hook time out of the buckets
t0 = time.perf_counter()
out = model.get_logits_batched(PROMPTS)
if torch.cuda.is_available():
    torch.cuda.synchronize()
wall = time.perf_counter() - t0

rss = max(rss_after_load, Process().memory_info().rss / 1024**3)
vram = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
inside = sum(cat.values())
buckets = {k: round(v, 2) for k, v in sorted(cat.items(), key=lambda kv: -kv[1])}
print(
    f"{'streamed' if streamed else 'eager'}: load={t_load:.1f}s wall={wall:.2f}s "
    f"rss={rss:.2f}GB vram={vram:.2f}GB inside={inside:.2f}s outside={wall - inside:.2f}s "
    f"buckets={buckets} shape={tuple(out.shape)}"
)
