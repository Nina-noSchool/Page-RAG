"""每周复盘：只依据证据，输出「下周只改变的一个变量」。

用法：
  python review.py          # 复盘最近 7 天（没有证据会拒绝——书的方法论如此）
  python review.py --show   # 回看最近一次复盘
"""
import argparse
import sys
from datetime import date, timedelta

import llm
from retrieve import Retriever
from store import connect, latest_review, recent_evidence, recent_tasks, save_review

REVIEW_SYSTEM = (
    "你是《人生进阶指南》方法论的复盘教练。只依据证据复盘，规则："
    "1) 本周完成了什么——逐条引用具体证据；"
    "2) 证据说明了什么、不能说明什么（书的原则：证据不含糊，不写过脑子的结论）；"
    "3) 下周只改变的一个变量——只选一个，说明为什么选它；"
    "4) 下周一的任务建议（25-45 分钟，可验证）。"
    "没有证据支持的话不要写。不判分、不鸡汤。输出 markdown，正文不超过 500 字。"
)


def review_week(con) -> str:
    evidence = recent_evidence(con, 7)
    tasks = recent_tasks(con, 7)
    if not evidence:
        sys.exit("最近 7 天没有证据记录，无法复盘。\n"
                 "书的规则：先看证据，再解释原因。先跑几天 python daily.py 并 --done 打卡。")

    ev_text = "\n".join(f"- [{d}] {c}" for d, c in evidence)
    tk_text = "\n".join(f"- [{d}][{st}] {c[:80]}" for d, s, c, st in tasks) or "（无）"
    hits = Retriever(con).search("每周复盘 只改变一个变量 证据", top_k=3,
                                 mode="hybrid", embed_fn=llm.embed)
    method = "\n".join(f"- {h.heading}：{h.text[:250]}" for h in hits)

    prompt = (f"本周（{date.today() - timedelta(days=7)} ~ {date.today()}）证据：\n{ev_text}\n\n"
              f"本周任务记录：\n{tk_text}\n\n"
              f"书中复盘方法（依据）：\n{method}\n\n请输出本周复盘（markdown）。")
    content = llm.chat([
        {"role": "system", "content": REVIEW_SYSTEM},
        {"role": "user", "content": prompt},
    ]).strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("markdown"):
            content = content[8:]
    content = content.strip()

    week = date.today().isocalendar()
    save_review(con, f"{week[0]}-W{week[1]:02d}", content)
    return content


def main():
    ap = argparse.ArgumentParser(description="每周复盘")
    ap.add_argument("--show", action="store_true", help="回看最近一次复盘")
    args = ap.parse_args()

    con = connect()
    if args.show:
        row = latest_review(con)
        if not row:
            sys.exit("还没有复盘记录：先运行 python review.py。")
        print(f"复盘于 {row[0]}（{row[1]}）\n\n{row[2]}")
        return

    print("书童翻检一周笔墨（只认证据，不认感觉）...\n")
    print(review_week(con))
    print("\n（复盘已存档。下周 daily.py 派发任务时会自动参考『只改变的一个变量』。）")


if __name__ == "__main__":
    main()
