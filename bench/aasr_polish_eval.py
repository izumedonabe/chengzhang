"""Chinese polishing eval on AASR-Bench (text only — no audio).

Input  = the benchmark's `oral` text (what a perfect ASR would output).
Output = each polisher's rewrite.
Scores (automatic, string matching — an approximation of the paper's LLM judge):
  content   does the output still contain the key words/phrases?      (+1 / 0)
  format    was "A" normalized to "B" (e.g. 二十二点五 → 22.5)?      (+1 / 0 / -1)
  filter    are the fillers / repetitions gone?                        (+2 / 0)
  rephrase  was the self-correction "A" resolved to "B"?               (+2 / 0 / -1)
  overall   = earned / max, as in the paper (clipped to 0..1)
Also CER against the reference `clean` text, and latency.

usage: python bench/aasr_polish_eval.py NAME [--limit N]
  NAME: raw | refiner | qwen35-4b | qwen35-2b
"""
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REFINER_SYSTEM = ("你是 ASR 文本纠错助手。保留原意，最小修改：去口癖/重复，修错字，补必要标点，"
                  "规范数字、日期、术语和代码符号，处理自我修正。不要总结、扩写或解释。"
                  "重要易错实体在末尾追加 <KEY>[词1、词2]；没有则不加。")
MODELS = {
    "refiner": str(ROOT / "models/agenticasr-refiner/mlx-int4"),
    "qwen35-4b": "mlx-community/Qwen3.5-4B-MLX-4bit",
    "qwen35-2b": "mlx-community/Qwen3.5-2B-MLX-4bit",
}


def squash(s: str) -> str:
    """for matching: NFKC, lowercase, drop spaces and punctuation"""
    s = unicodedata.normalize("NFKC", s).lower()
    return "".join(ch for ch in s if unicodedata.category(ch)[0] in "LNS" or ch in "%.:/+-_")


def cer(ref: str, hyp: str) -> tuple[int, int]:
    r, h = squash(ref), squash(hyp)
    prev = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        cur = [i]
        for j, b in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b)))
        prev = cur
    return prev[-1], len(r)


QUOTED = re.compile(r"“([^”]*)”")


def opt_scores(q):
    out = []
    for o in q["options"]:
        if isinstance(o, dict):
            out.append(int(o["score"]))
        else:
            m = re.search(r"\((-?\d+)\)\s*$", str(o))
            out.append(int(m.group(1)) if m else 0)
    return out


def score_question(q, output: str):
    """returns (earned, max) or None when the question can't be scored by matching"""
    out = squash(output)
    terms = [squash(t) for t in QUOTED.findall(q["question"])]
    scores = opt_scores(q)
    best = max(scores)
    has = lambda t: bool(t) and t in out
    cat = q["category"]
    if cat == "content":
        if not terms:
            return None
        return (1 if any(has(t) for t in terms) else 0), best
    if cat == "format" and len(terms) >= 2:
        src, dst = terms[0], terms[1:]
        if any(has(t) for t in dst):
            return 1, best
        return (0 if has(src) else -1), best
    if cat == "rephrase" and len(terms) >= 2:
        src, dst = terms[0], terms[1:]
        if has(src):
            return 0, best                      # the self-correction is still there
        if any(has(t) for t in dst):
            return 2, best                      # resolved to the final intent
        return 0, best                          # can't tell by matching: neutral
    if cat == "filter":
        ev = q.get("evidence", "")
        m = re.search(r"oral[:：]\s*(.*?)(?:;|；)\s*clean[:：]\s*(.*)$", ev)
        if not m:
            return None
        oral_bit, clean_bit = squash(m.group(1)), squash(m.group(2))
        if oral_bit and oral_bit != clean_bit and oral_bit in out:
            return 0, best                      # the filler / repetition survived
        return 2, best
    return None


def make_polisher(name):
    from chengzhang import prompts
    from chengzhang.engine import Config, Polisher

    if name == "refiner":
        prompts.messages = lambda text, dictionary, translate=None: [
            {"role": "system", "content": REFINER_SYSTEM}, {"role": "user", "content": text}]
    cfg = Config(llm_model=MODELS[name], dictionary=[], fast_path=False, max_tokens=512)
    return Polisher(cfg)


def clean_output(text: str) -> str:
    return re.sub(r"\s*<KEY>\[.*?\]\s*$", "", text, flags=re.S).strip()


def main():
    name = sys.argv[1]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    rows = [json.loads(l) for l in open(ROOT / "data/aasr/data/rubric.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["language"] == "zh" and r["status"] == "success"][:limit]

    pol = None if name == "raw" else make_polisher(name)
    if pol:
        pol("嗯那个我们明天开会吧")  # warmup

    out_path = ROOT / f"bench/aasr_out_{name}.jsonl"
    totals = {c: [0, 0] for c in ("content", "format", "filter", "rephrase")}
    agg = [0, 0]
    errs = refs = 0
    lat, ttft, unscored = [], [], 0
    by_scene = {}
    with out_path.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            if pol:
                res = pol(r["oral"])
                text = clean_output(res["text"])
                lat.append(res["total_s"])
                if res["ttft_s"]:
                    ttft.append(res["ttft_s"])
            else:
                text = r["oral"]
            e, n = cer(r["clean"], text)
            errs, refs = errs + e, refs + n
            earned = mx = 0
            for q in r["qa"]:
                s = score_question(q, text)
                if s is None:
                    unscored += 1
                    continue
                totals[q["category"]][0] += s[0]
                totals[q["category"]][1] += s[1]
                earned, mx = earned + s[0], mx + s[1]
            agg[0] += earned
            agg[1] += mx
            sc = by_scene.setdefault(r["scene"], [0, 0])
            sc[0] += earned
            sc[1] += mx
            f.write(json.dumps({"scene": r["scene"], "oral": r["oral"], "clean": r["clean"], "output": text,
                                "score": [earned, mx]}, ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                print(f"  {name}: {i + 1}/{len(rows)}", flush=True)

    pct = lambda a: round(100 * max(0, a[0]) / a[1], 1) if a[1] else None
    lat.sort()
    summary = {
        "name": name, "n": len(rows),
        "overall": pct(agg), **{c: pct(v) for c, v in totals.items()},
        "cer_vs_clean": round(100 * errs / refs, 2),
        "lat_median_s": round(lat[len(lat) // 2], 3) if lat else 0,
        "lat_p90_s": round(lat[int(len(lat) * 0.9)], 3) if lat else 0,
        "ttft_median_s": round(sorted(ttft)[len(ttft) // 2], 3) if ttft else 0,
        "unscored_questions": unscored,
        "by_scene": {k: pct(v) for k, v in sorted(by_scene.items())},
    }
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    with open(ROOT / "bench/results_aasr_polish.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
