"""本地向量库：SQLite 存 chunk 与向量，检索时一次性载入内存算余弦。

语料只有几百 KB、几百个 chunk，暴力余弦绰绰有余；
以后要换 Chroma/LanceDB，只需保持本文件这几个函数的语义。
"""
import sqlite3
from pathlib import Path

from config import INDEX_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id     INTEGER PRIMARY KEY,
    source TEXT,      -- 相对语料根的文件路径
    heading TEXT,     -- 标题层级路径，如「词汇篇 > 1. 先定义任务」
    text  TEXT,       -- chunk 正文（首行为标题行）
    ord   INTEGER,    -- 文件内顺序
    ctx   TEXT,       -- --contextual 生成的上下文句（可为空）
    vector BLOB       -- float32 小端序列化；未向量化时为 NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS ctx_cache (  -- 上下文缓存按 chunk 哈希存，重建索引不清掉
    h TEXT PRIMARY KEY,
    ctx TEXT
);
CREATE TABLE IF NOT EXISTS vec_cache (  -- 向量缓存同样按内容哈希，断点续跑/增量更新用
    h TEXT PRIMARY KEY,
    vector BLOB
);
CREATE TABLE IF NOT EXISTS diagnosis (  -- 第二阶段：每次诊断一行/技能
    id      INTEGER PRIMARY KEY,
    created TEXT,       -- ISO 时间，同一次诊断共享同一时间戳
    skill   TEXT,       -- 听力/口语/阅读/写作/语法
    level   TEXT,       -- CEFR 估计（证据不足时可为区间或「未知」）
    blocker TEXT,       -- 首要障碍（只选一个）
    detail  TEXT        -- 依据 + 下一个最小任务
);
CREATE TABLE IF NOT EXISTS plans (  -- 生成的 90 天计划存档
    id      INTEGER PRIMARY KEY,
    created TEXT,
    focus   TEXT,       -- 本期主方向
    content TEXT        -- 计划全文（markdown）
);
CREATE TABLE IF NOT EXISTS tasks (  -- 第三阶段：每日派发的任务
    id      INTEGER PRIMARY KEY,
    date    TEXT,       -- YYYY-MM-DD
    skill   TEXT,       -- 主线技能（可空）
    content TEXT,       -- 任务描述
    source  TEXT,       -- 来源：plan / review / adhoc
    status  TEXT,       -- pending / done / skipped
    created TEXT
);
CREATE TABLE IF NOT EXISTS evidence (  -- 用户交出的真实产出记录
    id      INTEGER PRIMARY KEY,
    date    TEXT,
    task_id INTEGER,    -- 可空
    content TEXT,       -- 做了什么、结果/产出在哪
    created TEXT
);
CREATE TABLE IF NOT EXISTS reviews (  -- 每周复盘存档
    id      INTEGER PRIMARY KEY,
    created TEXT,
    week    TEXT,       -- ISO 周标识，如 2026-W44
    content TEXT
);
"""


def connect(db: Path = INDEX_PATH) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    cols = [r[1] for r in con.execute("PRAGMA table_info(chunks)")]
    if cols and "ctx" not in cols:  # 旧版索引缺 ctx 列，直接重置（可重建）
        con.executescript("DROP TABLE IF EXISTS chunks; DROP TABLE IF EXISTS meta;")
        con.executescript(SCHEMA)
    return con


def reset(con: sqlite3.Connection):
    con.executescript("DROP TABLE IF EXISTS chunks; DROP TABLE IF EXISTS meta;")
    con.executescript(SCHEMA)


def add(con: sqlite3.Connection, rows):
    """rows: (source, heading, text, ord, ctx, vector_bytes|None)"""
    con.executemany(
        "INSERT INTO chunks(source, heading, text, ord, ctx, vector) VALUES (?,?,?,?,?,?)",
        rows,
    )
    con.commit()


def _cached_lookup(con: sqlite3.Connection, table: str, col: str, hashes: list[str]) -> dict:
    out = {}
    for i in range(0, len(hashes), 500):
        part = hashes[i : i + 500]
        q = f"SELECT h, {col} FROM {table} WHERE h IN ({','.join('?' * len(part))})"
        out.update(dict(con.execute(q, part).fetchall()))
    return out


def get_cached_ctx(con: sqlite3.Connection, hashes: list[str]) -> dict[str, str]:
    return _cached_lookup(con, "ctx_cache", "ctx", hashes)


def put_cached_ctx(con: sqlite3.Connection, pairs):
    con.executemany("INSERT OR REPLACE INTO ctx_cache VALUES (?,?)", pairs)
    con.commit()


def get_cached_vecs(con: sqlite3.Connection, hashes: list[str]) -> dict[str, bytes]:
    return _cached_lookup(con, "vec_cache", "vector", hashes)


def put_cached_vecs(con: sqlite3.Connection, pairs):
    con.executemany("INSERT OR REPLACE INTO vec_cache VALUES (?,?)", pairs)
    con.commit()


def add_diagnoses(con: sqlite3.Connection, rows):
    """rows: (created, skill, level, blocker, detail)"""
    con.executemany(
        "INSERT INTO diagnosis(created, skill, level, blocker, detail) VALUES (?,?,?,?,?)",
        rows,
    )
    con.commit()


def latest_diagnoses(con: sqlite3.Connection):
    """最近一次诊断的全部行（按 created 时间戳分组取最大）。"""
    latest = con.execute("SELECT MAX(created) FROM diagnosis").fetchone()[0]
    if not latest:
        return []
    return con.execute(
        "SELECT skill, level, blocker, detail FROM diagnosis WHERE created=? ORDER BY id",
        (latest,),
    ).fetchall()


def save_plan(con: sqlite3.Connection, created: str, focus: str, content: str):
    con.execute("INSERT INTO plans(created, focus, content) VALUES (?,?,?)",
                (created, focus, content))
    con.commit()


def latest_plan(con: sqlite3.Connection):
    return con.execute(
        "SELECT created, focus, content FROM plans ORDER BY id DESC LIMIT 1"
    ).fetchone()


# ---- 第三阶段：任务 / 证据 / 复盘 ----

def task_on(con: sqlite3.Connection, date: str):
    return con.execute(
        "SELECT id, skill, content, status FROM tasks WHERE date=? ORDER BY id DESC LIMIT 1",
        (date,),
    ).fetchone()


def add_task(con: sqlite3.Connection, date: str, skill: str, content: str, source: str) -> int:
    cur = con.execute(
        "INSERT INTO tasks(date, skill, content, source, status, created) "
        "VALUES (?,?,?,?, 'pending', ?)",
        (date, skill, content, source, datetime_now()),
    )
    con.commit()
    return cur.lastrowid


def set_task_status(con: sqlite3.Connection, task_id: int, status: str):
    con.execute("UPDATE tasks SET status=? WHERE id=?", (status, task_id))
    con.commit()


def recent_tasks(con: sqlite3.Connection, days: int = 7):
    return con.execute(
        "SELECT date, skill, content, status FROM tasks "
        "WHERE date >= date('now', ?) ORDER BY date DESC",
        (f"-{days} day",),
    ).fetchall()


def add_evidence(con: sqlite3.Connection, task_id, content: str, date: str | None = None):
    con.execute(
        "INSERT INTO evidence(date, task_id, content, created) VALUES (?,?,?,?)",
        (date or datetime_now()[:10], task_id, content, datetime_now()),
    )
    con.commit()


def recent_evidence(con: sqlite3.Connection, days: int = 7):
    return con.execute(
        "SELECT date, content FROM evidence "
        "WHERE date >= date('now', ?) ORDER BY id",
        (f"-{days} day",),
    ).fetchall()


def save_review(con: sqlite3.Connection, week: str, content: str):
    con.execute("INSERT INTO reviews(created, week, content) VALUES (?,?,?)",
                (datetime_now(), week, content))
    con.commit()


def latest_review(con: sqlite3.Connection):
    return con.execute(
        "SELECT created, week, content FROM reviews ORDER BY id DESC LIMIT 1"
    ).fetchone()


def reset_journey(con: sqlite3.Connection):
    """清空陪跑过程数据（任务/证据/复盘），保留诊断与计划历史。模拟新用户时用。"""
    con.executescript("DELETE FROM tasks; DELETE FROM evidence; DELETE FROM reviews;")
    con.commit()


def datetime_now() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


def set_meta(con: sqlite3.Connection, key: str, value: str):
    con.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, str(value)))
    con.commit()


def all_chunks(con: sqlite3.Connection):
    return con.execute(
        "SELECT id, source, heading, text, vector FROM chunks ORDER BY id"
    ).fetchall()
