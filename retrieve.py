"""检索：向量余弦 + BM25（中文 bigram 分词），RRF 融合为混合检索。"""
import re
import sys

import numpy as np
from rank_bm25 import BM25Okapi

import config
from config import TOP_K
from store import all_chunks

_TOKEN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]+")
RRF_K = 60  # RRF 常数，平滑排名权重


def tokenize(text: str) -> list[str]:
    """英文按词、连续汉字按 bigram 切，BM25 够用的轻量分词。"""
    tokens = []
    for run in _TOKEN.findall(text.lower()):
        if run.isascii():
            tokens.append(run)
        elif len(run) > 1:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
        else:
            tokens.append(run)
    return tokens


class Hit:
    def __init__(self, id_, source, heading, text, score):
        self.id, self.source, self.heading, self.text, self.score = (
            id_, source, heading, text, score,
        )


class Retriever:
    def __init__(self, con):
        rows = all_chunks(con)
        if not rows:
            sys.exit("索引为空：先运行 python ingest.py 建库。")
        self.ids = [r[0] for r in rows]
        self.sources = [r[1] for r in rows]
        self.headings = [r[2] for r in rows]
        self.texts = [r[3] for r in rows]
        vectors = [r[4] for r in rows]
        self.matrix = None
        if vectors and all(v is not None for v in vectors):
            m = np.stack([np.frombuffer(v, dtype="<f4") for v in vectors])
            self.matrix = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-10)
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])

    def search(self, query, top_k=TOP_K, mode="hybrid", embed_fn=None) -> list[Hit]:
        """mode: hybrid=向量+BM25 融合；vector/bm25 单路。embed_fn 传 llm.embed。"""
        ranked_lists = []  # 每路检索为 (id 排序列表, RRF 权重)
        if mode in ("vector", "hybrid"):
            if self.matrix is None:
                if mode == "vector":
                    sys.exit("向量库为空：先运行 python ingest.py（需在 .env 配好 key）完成向量化。")
                print("（提示：索引还没有向量，本次只走 BM25。完整跑一次 ingest 即为混合检索。）")
            elif embed_fn is None:
                if mode == "vector":
                    sys.exit("--no-llm 模式下无法向量化 query，请用 --mode bm25。")
                print("（--no-llm：本次只走 BM25。）")
            else:
                q = np.asarray(embed_fn([query])[0], dtype="<f4")
                q = q / max(float(np.linalg.norm(q)), 1e-10)
                sims = self.matrix @ q
                ranked_lists.append(
                    ([self.ids[i] for i in np.argsort(sims)[::-1][: top_k * 3]], 1.0)
                )
        if mode in ("bm25", "hybrid"):
            scores = self.bm25.get_scores(tokenize(query))
            order = np.argsort(scores)[::-1][: top_k * 3]
            ranked_lists.append(([self.ids[i] for i in order if scores[i] > 0],
                                 config.BM25_WEIGHT))

        fused: dict[int, float] = {}
        for lst, weight in ranked_lists:
            for rank, cid in enumerate(lst):
                fused[cid] = fused.get(cid, 0.0) + weight / (RRF_K + rank)
        hits = []
        for cid in sorted(fused, key=fused.get, reverse=True)[:top_k]:
            i = self.ids.index(cid)
            hits.append(Hit(cid, self.sources[i], self.headings[i], self.texts[i], fused[cid]))
        return hits
