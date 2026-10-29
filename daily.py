"""每日陪跑：派发今天的 25-45 分钟最小任务，回收证据。

用法：
  python daily.py                       # 派发/查看今天的任务
  python daily.py --done --note "..."   # 完成打卡：任务标记 done，note 存为证据
  python daily.py --skip                # 今天做不了，跳过（会留痕，复盘时可见）
"""
import argparse
import difflib
import json
import sys
from datetime import date as _date

import llm
from store import (add_evidence, add_task, connect, latest_diagnoses,
                   latest_plan, latest_review, recent_evidence, recent_tasks,
                   set_task_status, task_on)

DAILY_SYSTEM = (
    "你是学习陪跑教练。根据 90 天计划、最近诊断、已有证据和最近复盘，"
    "给出今天（仅今天）的一个最小任务。权威优先级：90 天计划 > 最近诊断 > 最近复盘 > 历史证据；"
    "若旧复盘或旧任务的主题与当前计划主线冲突，一律以计划为准，不要延续计划外的旧主题。"
    "任务要求：25-45 分钟、可验证、与计划当前阶段一致；"
    "不要与最近 3 天已派发的任务重复，在主线内做递进变化；"
    "若最近复盘给出了『只改变的一个变量』且不与计划冲突，今天的任务要体现它；"
    "任务描述具体到能直接开始做，不说空话。"
    '严格输出 JSON：{"skill": "听力/口语/阅读/写作/语法 之一", "content": "任务描述"}。'
)


def derive_task(con) -> tuple[str, str]:
    """从计划+诊断+证据+复盘推导今天的任务，返回 (skill, content)。
    带机械去重：与最近任务过于相似时带着批评重问模型，最多两次。"""
    plan = latest_plan(con)
    if not plan:
        sys.exit("还没有 90 天计划：先跑 python diagnose.py 和 python plan.py。")
    diag = "\n".join(f"- {s}: {lv}，障碍 {b}" for s, lv, b, _ in latest_diagnoses(con))
    evidence = "\n".join(f"- [{d}] {c[:150]}" for d, c in recent_evidence(con, 7)) or "（暂无）"
    recent = recent_tasks(con, 3)
    tasks = "\n".join(f"- [{d}][{st}] {c[:100]}" for d, s, c, st in recent_tasks(con, 7)) or "（暂无）"
    review_row = latest_review(con)
    review = (review_row[2][:600] if review_row else "（暂无复盘）")

    prompt = (f"今天是 {_date.today().isoformat()}（星期{'一二三四五六日'[_date.today().weekday()]}）。\n"
              f"90 天计划（节选）：\n{plan[2][:1500]}\n\n最近诊断：\n{diag}\n\n"
              f"最近 7 天证据：\n{evidence}\n\n最近 7 天任务：\n{tasks}\n\n"
              f"最近一次复盘：\n{review}\n\n请给出今天的任务 JSON。")

    def ask(extra: str) -> tuple[str, str]:
        raw = llm.chat([
            {"role": "system", "content": DAILY_SYSTEM},
            {"role": "user", "content": prompt + extra},
        ]).strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
        try:
            obj = json.loads(raw)
            return str(obj.get("skill", "")), str(obj.get("content", "")).strip()
        except Exception:
            sys.exit(f"任务 JSON 解析失败，模型原始输出：\n{raw[:400]}")

    def too_similar(content: str) -> bool:
        pool = [t[2] for t in recent]
        return any(difflib.SequenceMatcher(None, content[:80], p[:80]).ratio() > 0.6
                   for p in pool)

    skill, content = ask("")
    for _ in range(2):
        if not too_similar(content):
            break
        skill, content = ask(
            f"\n\n注意：你刚生成的任务「{content[:60]}」与最近已派发的任务重复了。"
            "必须换同一技能下不同的练习维度（保存基线 / 限时调用 / 真实受众 / 迁移），禁止重复。")
    return skill, content


def main():
    ap = argparse.ArgumentParser(description="每日陪跑")
    ap.add_argument("--done", action="store_true", help="今天完成打卡")
    ap.add_argument("--note", default="", help="随 --done 提交的证据说明")
    ap.add_argument("--skip", action="store_true", help="今天跳过")
    args = ap.parse_args()

    con = connect()
    today = _date.today().isoformat()
    row = task_on(con, today)

    if args.done or args.skip:
        if not row:
            sys.exit("今天还没有派发任务：先运行 python daily.py。")
        status = "done" if args.done else "skipped"
        set_task_status(con, row[0], status)
        if args.done:
            note = args.note.strip() or input("简单写下证据（做了什么/产出在哪）: ").strip()
            if note:
                add_evidence(con, row[0], note)
                print(f"书童记下了（证据已存档）：{note}")
        print(f"今天任务已标记 {status}：{row[2][:80]}")
        return

    if row:
        print(f"今天（{today}）的任务已派发：\n\n  [{row[1] or '综合'}] {row[2]}\n\n状态：{row[3]}")
        if row[3] == "pending":
            print("完成后运行：python daily.py --done --note \"做了什么/产出在哪\"")
        return

    print("书童研墨铺纸，派发今日任务 ...\n")
    skill, content = derive_task(con)
    add_task(con, today, skill, content, "plan")
    print(f"今日任务（{today}）：\n\n  [{skill or '综合'}] {content}\n")
    print("完成后运行：python daily.py --done --note \"做了什么/产出在哪\"")


if __name__ == "__main__":
    main()
