"""全局配置：路径与 LLM 接入参数，均可用 .env 覆盖。"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

try:  # Windows 下重定向输出默认 GBK，统一 UTF-8，避免管道里 print 崩掉
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")  # 把 .env 里的配置装进环境变量（此前漏了这行，.env 从未生效）
DATA_DIR = ROOT / "data" / "up"            # 语料：byoungd/up 仓库
INDEX_PATH = ROOT / "index" / "chunks.db"  # SQLite：chunk 正文 + 向量

# ---- LLM 接入（OpenAI 兼容接口，默认智谱：对话和 embedding 一套搞定）----
API_BASE = os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
API_KEY = os.getenv("LLM_API_KEY", "")
CHAT_MODEL = os.getenv("LLM_CHAT_MODEL", "glm-4-flash")
EMBED_MODEL = os.getenv("LLM_EMBED_MODEL", "embedding-3")
EMBED_BATCH = int(os.getenv("LLM_EMBED_BATCH", "16"))

# ---- 检索与切分参数 ----
MAX_CHARS = 1000  # 单 chunk 上限，超出按段落（表格不拆）再切
MIN_CHARS = 60    # 低于此长度的小节并入前一个 chunk
TOP_K = 5         # 每次检索返回条数
BM25_WEIGHT = float(os.getenv("RAG_BM25_WEIGHT", "0.8"))  # RRF 里 BM25 一路的权重（向量固定 1.0）；实测 0.8 最优：模板 chunk 关键词过密，BM25 满权会放大噪声
