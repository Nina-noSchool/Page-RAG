"""命令行问答：检索 -> 拼 prompt -> LLM 回答并标注出处。每次问答自动留痕。

用法：
  python ask.py 口语很差怎么练          # 单次提问（自动留痕）
  python ask.py                        # 进入交互循环（支持多轮追问）
  python ask.py --rewrite              # 追问先用对话模型改写成独立查询再检索（需 key）
  python ask.py --no-llm 证据卡        # 只看检索命中，不调用任何模型（不留痕）
  python ask.py --mode bm25 证据卡     # vector / bm25 / hybrid 三种检索模式
  python ask.py --log 5                # 回看最近 5 条问答留痕
  python ask.py --rate 12 bad --note "命中对了但答案太泛"   # 打标，bad 自动回流进评测题库
"""
import argparse
import json

import config
import llm
from eval import add_real_case, load_questions
from retrieve import Retriever
from store import (add_trace, connect, get_trace, recent_traces,
                   set_trace_rating)

SYSTEM = (
    "你是「书童」(Page)——基于《人生进阶指南》(github.com/byoungd/up) 的学习伴侣。"
    "书童的职守有三：替主人背书（只依据提供的资料回答，资料里没有的就直说没有，不要编造）、"
    "记日程、管笔墨。回答用中文，简洁务实，"
    "提到的方法注明来自哪条资料，格式如【1】【2】。"
)


def answer_once(con, retriever: Retriever, question: str, top_k: int, mode: str,
                use_llm: bool, rewrite: bool = False, history: list | None = None):
    q_search = question
    if rewrite and use_llm:
        q_search = llm.rewrite_query(question, history or [])
        if q_search != question:
            print(f"（查询改写：{q_search}）")

    hits = retriever.search(q_search, top_k=top_k, mode=mode,
                            embed_fn=llm.embed if use_llm else None)
    print(f"\n== 检索命中 {len(hits)} 条 ==")
    for i, h in enumerate(hits, 1):
        print(f"【{i}】{h.heading}  ({h.source})")
    if not use_llm:
        for i, h in enumerate(hits, 1):
            print(f"\n--- 【{i}】{h.heading} ---\n{h.text[:400]}")
        return

    context = "\n\n".join(f"【{i}】{h.heading}（{h.source}）\n{h.text}"
                          for i, h in enumerate(hits, 1))
    answer = llm.chat([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"资料：\n{context}\n\n问题：{question}"},
    ])
    print(f"\n{answer}")

    hits_json = json.dumps(
        [{"source": h.source, "heading": h.heading, "score": round(h.score, 4)}
         for h in hits], ensure_ascii=False)
    tid = add_trace(con, "ask", question, mode, hits_json, answer)
    print(f"\n（留痕 #{tid}；不满意可执行 python ask.py --rate {tid} bad）")

    if history is not None:  # 保留最近 3 轮，供 --rewrite 改写追问用
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        del history[:-6]


def show_traces(con, n: int):
    rows = recent_traces(con, n)
    if not rows:
        print("还没有留痕记录。")
        return
    for tid, created, query, mode, rating, answer in rows:
        print(f"#{tid} [{rating or '未评'}] {created}  {query}")
        print(f"   答案：{(answer or '')[:100]}...")
    print("\n打标：python ask.py --rate <编号> good|bad [--note \"备注\"]")


def rate_trace(con, tid: int, rating: str, note: str):
    if rating not in ("good", "bad"):
        print("rating 只能是 good 或 bad。")
        return
    t = get_trace(con, tid)
    if not t:
        print(f"没有 #{tid} 这条留痕（用 --log 查看编号）。")
        return
    set_trace_rating(con, tid, rating)
    print(f"#{tid} 已标记为 {rating}。")
    if rating == "bad":
        added = add_real_case(t[2], note)
        if added:
            print("已回流进 eval_questions.json 的「真实坏例区」"
                  "（待标注状态，补上 expect 关键词后开始计分）。")
        else:
            print("该问题已在真实坏例区，未重复添加。")
    n_real = len(load_questions().get("real", []))
    if n_real:
        print(f"当前真实坏例共 {n_real} 条。")


def main():
    ap = argparse.ArgumentParser(description="学习指南问答（书童）")
    ap.add_argument("question", nargs="*", help="不传则进入交互循环")
    ap.add_argument("--mode", choices=["hybrid", "vector", "bm25"], default="hybrid")
    ap.add_argument("--topk", type=int, default=config.TOP_K)
    ap.add_argument("--no-llm", action="store_true", help="只展示检索结果，不调用对话模型")
    ap.add_argument("--rewrite", action="store_true", help="追问先改写成独立查询再检索（需 key）")
    ap.add_argument("--log", nargs="?", const=5, type=int, metavar="N",
                    help="回看最近 N 条问答留痕（默认 5）")
    ap.add_argument("--rate", nargs=2, metavar=("ID", "good|bad"),
                    help="给某条留痕打标；bad 会回流进评测题库")
    ap.add_argument("--note", default="", help="随 --rate bad 提交的备注")
    args = ap.parse_args()

    con = connect()

    if args.log is not None:
        show_traces(con, args.log)
        return
    if args.rate:
        rate_trace(con, int(args.rate[0]), args.rate[1], args.note)
        return

    retriever = Retriever(con)
    use_llm = not args.no_llm
    history: list = []

    if args.question:
        answer_once(con, retriever, " ".join(args.question), args.topk,
                    args.mode, use_llm, args.rewrite, history)
        return
    print("书童在此。输入问题回车提问，Ctrl+C / Ctrl+Z 退出。")
    while True:
        try:
            q = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q:
            answer_once(con, retriever, q, args.topk, args.mode,
                        use_llm, args.rewrite, history)


if __name__ == "__main__":
    main()
