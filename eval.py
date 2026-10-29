"""检索质量评测：一组真实问题 + 期望命中，自动跑检索并打分。

用法：
  python eval.py                # 有向量走 hybrid，没向量自动退 bm25
  python eval.py --mode bm25    # 指定检索模式
  python eval.py --topk 5

判分规则：top-k 里只要有一条命中「期望列表」中任一关键词（出现在
文件路径或标题路径里），该题算通过。每次改切分参数或换 embedding
后都跑一遍，分数不应下降。
"""
import argparse

import llm
from retrieve import Retriever
from store import connect

# (问题, [期望命中的文件名片段或标题关键词，命中任一即可])
QUESTIONS = [
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


def main():
    ap = argparse.ArgumentParser(description="检索评测")
    ap.add_argument("--mode", default=None, choices=["hybrid", "vector", "bm25"])
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--hyde", action="store_true",
                    help="查询先经 HyDE 扩展（生成假设性答案段落再检索）")
    args = ap.parse_args()

    con = connect()
    r = Retriever(con)
    mode = args.mode or ("hybrid" if r.matrix is not None else "bm25")
    print(f"模式 {mode}，top-{args.topk}，HyDE={'开' if args.hyde else '关'}，共 {len(QUESTIONS)} 题\n")

    passed = p1 = 0
    rrr_sum = 0.0
    misses = []
    for q, expect in QUESTIONS:
        q_search = llm.hyde(q) if args.hyde else q
        hits = r.search(q_search, top_k=args.topk, mode=mode, embed_fn=llm.embed)
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
        print(f"[{'Y' if rank else 'N'}] {q}"
              + (f"（首个命中第 {rank} 名）" if rank else ""))
        for h in hits:
            hit_ok = any(e in h.source or e in h.heading for e in expect)
            print(f"   {'y' if hit_ok else '.'} {h.heading}  ({h.source})")
        print()

    n = len(QUESTIONS)
    print(f"结果：pass@{args.topk} = {passed}/{n}，pass@1 = {p1}/{n}，"
          f"MRR = {rrr_sum / n:.3f}   （MRR 越接近 1 说明正确结果排得越靠前）")
    if misses:
        print("未通过：", "；".join(misses))


if __name__ == "__main__":
    main()
