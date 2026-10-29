"""模拟一周真实使用：诊断 → 计划 → 6 天任务+混合质量证据 → 周复盘 → 新任务派发。

用法：
  python simulate_week.py answers.sample4.json

目的：检验陪跑闭环在"真实节奏"下的表现——用户不会每天完美打卡，
证据有好有坏、偶尔跳过，复盘必须消化这些。产物全部落库。
"""
import json
import sys
from datetime import date, datetime, timedelta

from retrieve import Retriever
import diagnose as diag_mod
import daily as daily_mod
import plan as plan_mod
import review as review_mod
from store import (add_diagnoses, add_evidence, add_task, connect,
                   latest_review, reset_journey, set_task_status)

# 6 天的证据风格轮换：真实用户不会每天完美
STYLES = ["good", "vague", "skip", "good", "vague", "good"]


def main():
    persona = sys.argv[1] if len(sys.argv) > 1 else "answers.sample4.json"
    fresh = "--fresh" in sys.argv
    con = connect()
    if fresh:
        reset_journey(con)
        print("（--fresh：已清空任务/证据/复盘，模拟新用户；诊断与计划历史保留）\n")
    answers = json.loads(open(persona, encoding="utf-8").read())
    retriever = Retriever(con)

    print("== 1. 诊断 ==")
    results = diag_mod.diagnose(answers, retriever)
    created = datetime.now().isoformat(timespec="seconds")
    add_diagnoses(con, [(created, str(r.get("skill", "")), str(r.get("level", "未知")),
                         str(r.get("blocker", "")), str(r.get("detail", "")))
                        for r in results])
    for r in results:
        print(f"  {r.get('skill')}: {r.get('level')}  障碍: {r.get('blocker')}")

    print("\n== 2. 计划 ==")
    plan_mod.build_plan(con)
    print("  已生成并存档（plan.py --show 查看）")

    print("\n== 3. 模拟过去 6 天 ==")
    today = date.today()
    for i, offset in enumerate(range(6, 0, -1)):
        d = (today - timedelta(days=offset)).isoformat()
        skill, content = daily_mod.derive_task(con)
        tid = add_task(con, d, skill, content, "plan")
        style = STYLES[i % len(STYLES)]
        if style == "skip":
            set_task_status(con, tid, "skipped")
            print(f"  [{d}] {skill}: {content[:40]}... → 跳过")
            continue
        if style == "good":
            ev = (f"按任务做了：{content[:50]}。完整完成，产出笔记和录音存在手机，"
                  "中间走神两次，最后 10 分钟状态最好")
        else:
            ev = f"按计划练了一会儿（{content[:30]}...），感觉还行，没做记录"
        set_task_status(con, tid, "done")
        add_evidence(con, tid, ev, date=d)
        print(f"  [{d}] {skill}: {content[:40]}... → {style} 证据")

    print("\n== 4. 周复盘 ==")
    print(review_mod.review_week(con))

    print("\n== 5. 复盘后派发新任务 ==")
    skill, content = daily_mod.derive_task(con)
    add_task(con, today.isoformat(), skill, content, "review")
    print(f"  [{skill}] {content}")
    lv = latest_review(con)
    if lv:
        print(f"\n（复盘要点回看：{lv[2][:150]}...）")


if __name__ == "__main__":
    main()
