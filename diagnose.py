"""CEFR 自测诊断流：问你能完成什么、卡在哪，对照书中标准估级并存档。

用法：
  python diagnose.py                          # 交互问答
  python diagnose.py --answers answers.json   # 用答案文件跑（测试/复跑）

答案文件格式：
  {"听力": {"能完成的真实任务": "...", "最卡的时刻": "..."}, ...}
每项技能问两个问题：能完成的真实任务（估级依据）、最卡的时刻（定首要障碍）。
"""
import argparse
import json
import sys
from datetime import datetime

import llm
from retrieve import Retriever
from store import add_diagnoses, connect, latest_diagnoses

SKILLS = ["听力", "口语", "阅读", "写作", "语法"]

QUESTIONS = {
    "听力": "最近一次听真实语速的材料（会议/剧/播客）是什么场合？听完能复述出什么？",
    "口语": "最近一次真实开口（会议发言/面试/闲聊）说了什么？对方理解到位吗？",
    "阅读": "最近在读什么英文材料？一页下来能记住和复述多少？",
    "写作": "最近一次写英文（邮件/文档/帖子）是什么？别人读了反应如何？",
    "语法": "说或写的时候，哪些语法点让你犹豫或反复出错？",
}

BLOCKER_HINT = "哪个瞬间最常卡住？（如：词听不出/组织不起来/时态犹豫/不敢开口……随便说）"

EXTRACT_SYSTEM = (
    "你是证据提取器，只做客观取证，不做任何评级。先判断用户学习的目标语言；"
    "若目标语言不是英语，正常提取全部证据，仅在每项 evidence 末尾追加标注"
    "『【目标语言：X】本书语料为英语学习方法论，仅方法可迁移』——"
    "绝不能因为语种不同就少提或不提证据。"
    "从用户自述中为每项技能提取可验证的真实任务证据：具体场合、能复述或展示的内容、"
    "结果如何。注意：考试分数、「我全懂」「我很有创意」这类无法验证的断言不算证据；"
    "自述前后矛盾要如实写进 contradiction。同时保留用户的年龄/身份/身体条件等"
    "背景特征（如儿童、老年、视力听力限制），它们影响任务设计。"
    "每项技能输出 JSON 对象：skill、evidence(提取到的具体证据，没有就写「无具体任务证据」)、"
    "contradiction(自相矛盾之处，没有则空字符串)。严格输出 JSON 数组，不要其他内容。"
)

DIAG_SYSTEM = (
    "你是英语学习诊断助手，方法论来自《人生进阶指南》：只按真实任务证据估级，"
    "不按感觉、学历、考试分数和词汇量。输入是预先提取好的证据，逐项判断："
    "对每项技能输出 JSON 对象："
    'skill(技能名)、level(保守 CEFR 估计，如 "A2"；evidence 为「无具体任务证据」'
    '时必须写「未知」，并以下一个最小任务去采集首次证据)、'
    'blocker(首要障碍，只选一个，尽量用书中「证据卡」的维度表述；'
    '证据不足时写「基线未建立，还没有可验证的任务证据」)、'
    'detail(一两句判断依据，只引用证据内容；证据不足时要说明：自述里有哪些断言、'
    '各缺什么验证、矛盾点是什么)、next_task(25-45 分钟能完成的一个最小任务；'
    "证据不足时给「采集基线证据」型任务，必须针对该技能具体化——"
    "听力：任选 3 分钟真实语速材料做一次真实首听并记下听懂的部分；"
    "口语：录 60 秒真实开口（自我介绍或复述今天做的事）；"
    "阅读：读一页真实材料，记录生词和能复述的主旨；"
    "写作：写 5 句真实邮件或评论，不做美化；"
    "语法：从上面的真实产出里挑出 5 个拿不准的用法)。"
    "任务难度必须随 level 和用户背景缩放：未知/A1 → 感官化低门槛（跟读、分级读物、儿歌、"
    "涂鸦式拼写）；A2-B1 → 单一技能的最小真实任务；B2+ → 真实受众反馈、native 校对、"
    "高阶输入（文学、俚语、即兴幽默），禁止给高水平用户布置入门基线任务；"
    "儿童用户用儿童场景（绘本/玩具/课堂游戏）；老年用户考虑记忆与视力特点"
    "（少量多次、当晚复测、大字材料）。若证据标注了【目标语言：非英语】，"
    "detail 里要声明本书材料不适用、仅方法可迁移，任务相应替换语种。"
    "只评证据，不评价用户本人，也不用情绪化措辞。"
    "严格输出 JSON 数组，元素数与技能数一致，不要任何其他内容。"
)


def collect_interactive() -> dict:
    answers = {}
    for skill in SKILLS:
        print(f"\n=== {skill} ===")
        answers[skill] = {
            "能完成的真实任务": input(f"{QUESTIONS[skill]}\n> ").strip(),
            "最卡的时刻": input(f"{BLOCKER_HINT}\n> ").strip(),
        }
    return answers


def diagnose(answers: dict, retriever: Retriever) -> list[dict]:
    """两段式：先客观取证（防吹牛带偏），再对照书中标准定级。"""
    criteria, user_text = [], []
    for skill in SKILLS:
        a = answers.get(skill, {})
        hits = retriever.search(f"{skill} CEFR 能力自测 证据卡", top_k=2,
                                mode="hybrid", embed_fn=llm.embed)
        criteria.append(f"【{skill}】书中标准：\n" + "\n".join(
            f"- {h.heading}：{h.text[:200]}" for h in hits))
        user_text.append(
            f"【{skill}】能完成的任务：{a.get('能完成的真实任务', '（未答）') or '（未答）'}；"
            f"最卡的时刻：{a.get('最卡的时刻', '（未答）') or '（未答）'}")

    user_block = "\n".join(user_text)

    def parse_json(raw: str):
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
        return json.loads(raw)

    # 第一遍：客观取证
    raw = llm.chat([
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": user_block},
    ])
    try:
        evidence = parse_json(raw)
        assert isinstance(evidence, list) and evidence
    except Exception:
        sys.exit(f"证据提取解析失败，模型原始输出：\n{raw[:500]}")

    # 第二遍：只依据证据定级（分类器看不到原始自述）
    criteria_text = "\n\n".join(criteria)
    evidence_text = "\n".join(
        f"- {e.get('skill')}：证据={e.get('evidence', '无具体任务证据')}"
        f"（矛盾：{e.get('contradiction', '') or '无'}）" for e in evidence)
    raw = llm.chat([
        {"role": "system", "content": DIAG_SYSTEM},
        {"role": "user",
         "content": f"书中各技能的自测标准：\n{criteria_text}\n\n"
                    f"已提取的证据（只能依据这些）：\n{evidence_text}\n\n"
                    "请输出诊断 JSON 数组。"},
    ])
    try:
        arr = parse_json(raw)
        assert isinstance(arr, list) and arr
    except Exception:
        sys.exit(f"诊断结果解析失败，模型原始输出：\n{raw[:500]}")
    return arr


def main():
    ap = argparse.ArgumentParser(description="CEFR 自测诊断")
    ap.add_argument("--answers", help="答案 JSON 文件路径（不传则交互问答）")
    args = ap.parse_args()

    con = connect()
    retriever = Retriever(con)

    if args.answers:
        answers = json.loads(open(args.answers, encoding="utf-8").read())
    else:
        print("回答越具体越好：说真实发生的事，不评价自己。\n（Ctrl+C 退出）")
        answers = collect_interactive()

    results = diagnose(answers, retriever)

    created = datetime.now().isoformat(timespec="seconds")
    rows = [(created, str(r.get("skill", "")), str(r.get("level", "未知")),
             str(r.get("blocker", "")), str(r.get("detail", "")) +
             ("｜下一个最小任务：" + r["next_task"] if r.get("next_task") else ""))
            for r in results]
    add_diagnoses(con, rows)

    print("\n========== 诊断结果（已存档）==========")
    for r in results:
        print(f"\n【{r.get('skill')}】估计 {r.get('level', '未知')}")
        print(f"  首要障碍：{r.get('blocker', '')}")
        print(f"  依据：{r.get('detail', '')}")
        if r.get("next_task"):
            print(f"  下一个最小任务：{r['next_task']}")
    print("\n跑 python plan.py 可基于本次诊断生成 90 天计划。")


if __name__ == "__main__":
    main()
