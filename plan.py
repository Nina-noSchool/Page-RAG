"""生成 90 天学习计划：读最近诊断 -> 检索对应章节 -> 生成并存档。

用法：
  python plan.py          # 基于最近一次诊断生成（需先跑过 diagnose.py）
  python plan.py --show   # 只显示最近一次存的计划
"""
import argparse
import sys
from datetime import datetime

import llm
from retrieve import Retriever
from store import connect, latest_diagnoses, latest_plan, save_plan

PLAN_SYSTEM = (
    "你是《人生进阶指南》方法论的学习计划教练。规则："
    "1) 从诊断结果里只选 1-2 个「对真实任务影响最大」的技能作为本期主线并给出选择理由；"
    "其余技能明确写「本期不安排专项，仅作为日常输入材料」；"
    "2) 每个任务必须落在用户自述过的真实场景里（自述提到开会就用会议练，"
    "提到给孩子读绘本就用绘本练），禁止使用与用户生活无关的泛化练习；"
    "3) 诊断若含情绪/心理类障碍（怕丢人、焦虑、不敢开口），计划必须安排对应的"
    "恢复动作（参照书中《善待自己》：把内在批评改写成 事实/影响/责任/支持 四句）；"
    "4) 12 周结果写成可验证的行为变化（例：站会上被追问两轮仍能推进讨论），"
    "禁止承诺 CEFR 等级或考试分数；诊断多为「未知」时，阶段一以采集基线证据为主；"
    "5) 每天 25-45 分钟一个最小闭环；每周只改变一个变量；三阶段递进"
    "（保存基线→限时调用与真实受众→换主题/模态/听众迁移），禁止阶段间复制任务；"
    "6) 用证据卡验收：无提示回忆、限时调用、真实受众结果，不看答案后的熟悉感。"
    "禁止输出书中任何填空模板或代码块（Learning State、状态版本、维护人等字段一律不要），"
    "输出直接从「## 本期主线」开始。"
    "正文不超过 900 字。输出 markdown。"
)

RETRIEVE_QUERIES = ["90 天计划 三个阶段 里程碑", "先定义任务 真实场景 完成标准",
                    "证据卡 延迟复测 真实受众", "每周复盘 只改变一个变量",
                    "善待自己 内在批评 恢复"]


def build_plan(con) -> str:
    rows = latest_diagnoses(con)
    if not rows:
        sys.exit("还没有诊断记录：先运行 python diagnose.py。")
    retriever = Retriever(con)

    contexts, seen = [], set()
    for q in RETRIEVE_QUERIES:
        for h in retriever.search(q, top_k=2, mode="hybrid", embed_fn=llm.embed):
            key = (h.source, h.heading)
            if key not in seen:
                seen.add(key)
                contexts.append(f"【{h.heading}】({h.source})\n{h.text[:600]}")

    diag_text = "\n".join(
        f"- {skill}：估计 {level or '未知'}，首要障碍：{blocker}。{detail}"
        for skill, level, blocker, detail in rows)
    prompt = ("最近一次诊断结果（detail 里引用了用户自述的真实场景——"
              "所有任务必须能在这些场景里执行）：\n" + diag_text +
              "\n\n书中相关章节（作为方法论依据）：\n" + "\n\n".join(contexts) +
              "\n\n请先选出本期主线（1-2 项，说明为什么是它们），"
              "再生成 90 天个人学习计划（markdown）。")

    content = llm.chat([
        {"role": "system", "content": PLAN_SYSTEM},
        {"role": "user", "content": prompt},
    ]).strip()
    if content.startswith("```"):  # 模型偶尔把整份计划包进代码块，剥掉
        content = content.strip("`")
        if content.lower().startswith("markdown"):
            content = content[8:]
    content = content.strip()
    focus = "、".join(f"{r[0]}({r[1]})" for r in rows[:2])
    save_plan(con, datetime.now().isoformat(timespec="seconds"), focus, content)
    return content


def main():
    ap = argparse.ArgumentParser(description="90 天计划生成")
    ap.add_argument("--show", action="store_true", help="只显示最近一次存的计划")
    args = ap.parse_args()

    con = connect()
    if args.show:
        row = latest_plan(con)
        if not row:
            sys.exit("还没有生成过计划：先运行 python plan.py。")
        print(f"生成于 {row[0]}（主线：{row[1]}）\n\n{row[2]}")
        return

    print("检索书中方法论并生成计划 ...\n")
    print(build_plan(con))
    print("\n（计划已存档，随时用 python plan.py --show 回看）")


if __name__ == "__main__":
    main()
