"""命令行问答：检索 -> 拼 prompt -> LLM 回答并标注出处。

用法：
  python ask.py 口语很差怎么练          # 单次提问
  python ask.py                        # 进入交互循环（支持多轮追问）
  python ask.py --rewrite              # 追问先用对话模型改写成独立查询再检索（需 key）
  python ask.py --no-llm 证据卡        # 只看检索命中，不调用任何模型
  python ask.py --mode bm25 证据卡     # vector / bm25 / hybrid 三种检索模式
"""
import argparse

import config
import llm
from retrieve import Retriever
from store import connect

SYSTEM = (
    "你是「书童」(Page)——基于《人生进阶指南》(github.com/byoungd/up) 的学习伴侣。"
    "书童的职守有三：替主人背书（只依据提供的资料回答，资料里没有的就直说没有，不要编造）、"
    "记日程、管笔墨。回答用中文，简洁务实，"
    "提到的方法注明来自哪条资料，格式如【1】【2】。"
)


def answer_once(retriever: Retriever, question: str, top_k: int, mode: str,
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

    if history is not None:  # 保留最近 3 轮，供 --rewrite 改写追问用
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        del history[:-6]


def main():
    ap = argparse.ArgumentParser(description="学习指南问答")
    ap.add_argument("question", nargs="*", help="不传则进入交互循环")
    ap.add_argument("--mode", choices=["hybrid", "vector", "bm25"], default="hybrid")
    ap.add_argument("--topk", type=int, default=config.TOP_K)
    ap.add_argument("--no-llm", action="store_true", help="只展示检索结果，不调用对话模型")
    ap.add_argument("--rewrite", action="store_true", help="追问先改写成独立查询再检索（需 key）")
    args = ap.parse_args()

    con = connect()
    retriever = Retriever(con)
    use_llm = not args.no_llm
    history: list = []

    if args.question:
        answer_once(retriever, " ".join(args.question), args.topk, args.mode,
                    use_llm, args.rewrite, history)
        return
    print("书童在此。输入问题回车提问，Ctrl+C / Ctrl+Z 退出。")
    while True:
        try:
            q = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q:
            answer_once(retriever, q, args.topk, args.mode, use_llm, args.rewrite, history)


if __name__ == "__main__":
    main()
