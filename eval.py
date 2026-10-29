"""检索质量评测：内置主榜 + 真实坏例回流，自动跑检索并打分。

用法：
  python eval.py                    # 有向量走 hybrid，没向量自动退 bm25
  python eval.py --mode bm25        # 指定检索模式
  python eval.py --hyde             # HyDE 查询扩展（实验性，默认关）
  python eval.py --topk 5

题库在 eval_questions.json：
- curated：人工标注期望的主榜（计分）
- real：ask.py --rate <id> bad 回流的真实坏例（补上 expect 后开始计分）
判分规则：top-k 里只要有一条命中「期望列表」任一关键词（出现在文件路径
或标题路径里）即算通过。每次改切分/换 embedding/调权重后跑一遍，分数不应下降。
"""
import argparse
import json
from pathlib import Path

import llm
from retrieve import Retriever
from store import connect

EVAL_FILE = Path(__file__).resolve().parent / "eval_questions.json"

# 内置题库：仅在 eval_questions.json 不存在时用于首建
CURATED_SEED = [
    ("口语很差怎么练", ["5-speaking"]),
    ("背了就忘，词汇量上不去怎么办", ["2-vocabulary", "rhythm"]),
    ("CEFR 自测怎么定目标", ["0-cefr"]),
    ("怎么用 AI 练英语", ["7-ai"]),
    ("90 天计划怎么制定", ["90-day-plan"]),
    ("每周复盘要看哪些东西", ["weekly-review"]),
    ("注意力涣散学不进去", ["3-attention-and-judgment"]),
    ("听力听不懂语速快的材料", ["3-listening"]),
    ("情绪低落焦虑影响学习怎么办", ["depression-anxiety"]),
    ("语法要不要专门学", ["grammar"]),
    ("证据卡是什么", ["证据卡"]),
    ("写作怎么找真实读者", ["6-writing"]),
    ("三天打鱼两天晒网没有节奏", ["rhythm", "discipline", "consistency"]),
    ("程序员应该背哪些单词表", ["词表", "2-vocabulary"]),
    ("怎么核对 AI 给的翻译对不对", ["7-ai"]),
    ("原版书读不下去怎么办", ["4-reading"]),
]


def load_questions() -> dict:
    if not EVAL_FILE.exists():
        save_questions({"curated": [{"q": q, "expect": e} for q, e in CURATED_SEED],
                        "real": []})
    return json.loads(EVAL_FILE.read_text(encoding="utf-8"))


def save_questions(data: dict):
    EVAL_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                         encoding="utf-8")


def add_real_case(question: str, note: str = "") -> bool:
    """坏例回流：重复问题不重复入榜。"""
    data = load_questions()
    if any(x["q"] == question for x in data["real"]):
        return False
    data["real"].append({"q": question, "expect": [], "note": note or "真实坏例"})
    save_questions(data)
    return True


def run_set(r: Retriever, entries: list[dict], top_k: int, mode: str,
            hyde: bool, label: str, verbose: bool = True) -> tuple:
    """跑一组题，返回 (passed, p1, rrr_sum, misses)。"""
    passed = p1 = 0
    rrr_sum = 0.0
    misses = []
    for item in entries:
        q, expect = item["q"], item["expect"]
        q_search = llm.hyde(q) if hyde else q
        hits = r.search(q_search, top_k=top_k, mode=mode, embed_fn=llm.embed)
        rank = None  # 首个正确命中的名次
        for pos, h in enumerate(hits, 1):
            if any(e in h.source or e in h.heading for e in expect):
                rank = pos
                break
        passed += rank is not None
        p1 += rank == 1
        rrr_sum += 0.0 if rank is None else 1.0 / rank
        if rank is None:
            misses.append(q)
        if verbose:
            print(f"[{'Y' if rank else 'N'}] {q}"
                  + (f"（首个命中第 {rank} 名）" if rank else ""))
            for h in hits:
                hit_ok = any(e in h.source or e in h.heading for e in expect)
                print(f"   {'y' if hit_ok else '.'} {h.heading}  ({h.source})")
    n = len(entries)
    if verbose and n:
        print(f"{label}：pass@{top_k} = {passed}/{n}，pass@1 = {p1}/{n}，"
              f"MRR = {rrr_sum / n:.3f}")
    return passed, p1, rrr_sum, misses


def main():
    ap = argparse.ArgumentParser(description="检索评测")
    ap.add_argument("--mode", default=None, choices=["hybrid", "vector", "bm25"])
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--hyde", action="store_true",
                    help="查询先经 HyDE 扩展（实验性，默认关）")
    args = ap.parse_args()

    con = connect()
    r = Retriever(con)
    mode = args.mode or ("hybrid" if r.matrix is not None else "bm25")
    data = load_questions()
    curated, real = data["curated"], data.get("real", [])
    print(f"模式 {mode}，top-{args.topk}，HyDE={'开' if args.hyde else '关'}，"
          f"主榜 {len(curated)} 题\n")

    passed, p1, rrr, misses = run_set(r, curated, args.topk, mode, args.hyde, "主榜")
    n = len(curated)
    print(f"\n主榜结果：pass@{args.topk} = {passed}/{n}，pass@1 = {p1}/{n}，"
          f"MRR = {rrr / n:.3f}   （MRR 越接近 1 说明正确结果排得越靠前）")
    if misses:
        print("主榜未通过：", "；".join(misses))

    if real:
        annotated = [x for x in real if x.get("expect")]
        pending = [x for x in real if not x.get("expect")]
        print(f"\n== 真实坏例区（{len(annotated)} 条计分 / {len(pending)} 条待标注）==")
        if annotated:
            rp, _, _, rmiss = run_set(r, annotated, args.topk, mode, args.hyde,
                                      "真实坏例", verbose=False)
            print(f"真实坏例：pass@{args.topk} = {rp}/{len(annotated)}")
            for m in rmiss:
                print(f"   [N] {m}")
        for x in pending:
            print(f"   [待标注] {x['q']}"
                  + (f"  （{x.get('note', '')}）" if x.get("note") else "")
                  + "  ← 在 eval_questions.json 补 expect 后开始计分")


if __name__ == "__main__":
    main()
