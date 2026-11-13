"""Reference for crates/cz-mlx/examples/llm_check: same prompt through mlx-lm (greedy)."""
import sys, time, glob
sys.path.insert(0, __import__("os").path.dirname(__file__) + "/../..")
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache
from chengzhang import prompts

d = sys.argv[1]
model, tok = load(d)
for text in sys.argv[2:]:
    ids = tok.apply_chat_template(prompts.messages(text, [], None), add_generation_prompt=True,
                                  enable_thinking=False, tokenize=True)
    out, t0, first = "", time.perf_counter(), None
    for r in stream_generate(model, tok, prompt=ids, max_tokens=200, prompt_cache=make_prompt_cache(model)):
        first = first or time.perf_counter() - t0
        out += r.text
    print(f"tokens {len(ids)} first {ids[:6]} last {ids[-6:]}")
    print(f"prefill+first {first:.3f}s gen {r.generation_tps:.1f} tok/s")
    print(out)
