"""LLM 接入：OpenAI 兼容接口的对话与 embedding 封装。"""
import json
import sys

import config

_client = None

REWRITE_SYSTEM = (
    "你是检索查询改写器。结合对话历史，把用户的追问改写成一条独立、完整、"
    "包含关键术语的中文搜索查询。只输出改写后的查询本身，不要解释。"
)

CTX_SYSTEM = (
    "你是检索索引助手。给你一份文档大纲和若干正文片段，请为每个片段写一句"
    "不超过40字的中文说明（它在讲什么、属于文档哪个部分），供向量检索补充上下文。"
    "严格输出 JSON 数组，元素与片段一一对应，不要输出任何其他内容。"
)

HYDE_SYSTEM = (
    "你是检索辅助器。假设有一份学习方法的文档能回答这个问题，"
    "写一段 120 字以内、最可能出现在该文档里的中文段落：直接陈述方法要点和步骤，"
    "不回答问题、不写前言、不复述问题。只输出该段落。"
)


def client():
    global _client
    if _client is None:
        if not config.API_KEY:
            sys.exit("未配置 LLM_API_KEY：请复制 .env.example 为 .env 并填写 key。")
        from openai import OpenAI

        # SDK 自带指数退避重试，覆盖 429/5xx/网络抖动；5 次仍失败才抛错
        _client = OpenAI(base_url=config.API_BASE, api_key=config.API_KEY,
                         timeout=60.0, max_retries=5)
    return _client


def embed_batch(texts: list[str]) -> list[list[float]]:
    """单个批次的向量化，返回与输入顺序一致的向量列表。"""
    resp = client().embeddings.create(model=config.EMBED_MODEL, input=list(texts))
    return [d.embedding for d in resp.data]


def embed(texts: list[str]) -> list[list[float]]:
    """查询用向量化（单条或少量，不做批次循环——建库的批次循环在 ingest 里）。"""
    return embed_batch(texts)


def chat(messages: list[dict]) -> str:
    resp = client().chat.completions.create(
        model=config.CHAT_MODEL, messages=messages, temperature=0.3
    )
    return resp.choices[0].message.content


def rewrite_query(question: str, history: list[dict] | None = None) -> str:
    """把多轮对话里的追问改写成独立查询；无 key 或失败时原样返回。"""
    if not config.API_KEY:
        return question
    msgs = [{"role": "system", "content": REWRITE_SYSTEM}]
    msgs.extend((history or [])[-6:])
    msgs.append({"role": "user",
                 "content": f"把这条追问改写成独立的搜索查询，只输出查询本身：\n{question}"})
    try:
        out = chat(msgs).strip()
        return out.splitlines()[0].strip().strip('"') if out else question
    except Exception:
        return question


def hyde(question: str) -> str:
    """HyDE：先生成假设性答案段落，再拿它去做检索；失败时退回原问题。"""
    if not config.API_KEY:
        return question
    try:
        out = chat([
            {"role": "system", "content": HYDE_SYSTEM},
            {"role": "user", "content": question},
        ]).strip()
        return out.splitlines()[0][:300] if out else question
    except Exception:
        return question


def contextualize(outlines: list[str], chunks: list[str]) -> list[str]:
    """为一批 chunk 各写一句上下文；失败返回空串列表（失败结果不缓存）。"""
    parts = []
    for i, (o, c) in enumerate(zip(outlines, chunks)):
        parts.append(f"<片段{i + 1}，文档大纲：{o}>\n{c[:800]}")
    try:
        raw = chat([
            {"role": "system", "content": CTX_SYSTEM},
            {"role": "user", "content": "\n\n".join(parts)},
        ]).strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
        arr = json.loads(raw)
        if not isinstance(arr, list):
            return [""] * len(chunks)
        out = []
        for i in range(len(chunks)):
            v = arr[i] if i < len(arr) else ""
            out.append(str(v).strip() if isinstance(v, str) and v.strip() else "")
        return out
    except Exception:
        return [""] * len(chunks)
