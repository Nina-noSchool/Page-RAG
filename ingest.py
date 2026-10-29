"""建库：扫描语料 -> 按标题切分 -> （可选）上下文增强与向量化 -> 写入 SQLite。

用法：
  python ingest.py --dry-run        # 只切分，打印统计，不调用 API
  python ingest.py --chunk-only     # 切分入库但不向量化（无 key 可先测 BM25）
  python ingest.py                  # 完整建库（需要 .env 配好 key）
  python ingest.py --contextual     # 建库前先给每个 chunk 生成一句上下文（需 key）
  python ingest.py --stats          # 查看当前索引状态

--contextual 是 Anthropic Contextual Retrieval 的轻量版：用文档大纲而非全文
做上下文输入；生成的上下文按 chunk 哈希缓存，重复建库不重复调用。
写库放在最后一步：中途失败不影响现有索引。
"""
import argparse
import hashlib
import re
import struct
import sys

import config
import llm
import store

SPLIT_LEVEL = 2  # 在 # 和 ## 标题处开新 chunk，更深的标题留在当前块
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
FRONTMATTER_TITLE = re.compile(r"^title:\s*(.+)$", re.M)
CTX_BATCH = 8    # 每次让模型处理的 chunk 数


def iter_corpus_files(include_all: bool):
    """默认收录中文正文 docs/threads 与工作纸 docs/templates。"""
    docs = config.DATA_DIR / "docs"
    files = sorted((docs / "threads").rglob("*.md")) + sorted((docs / "templates").glob("*.md"))
    for f in files:
        rel = f.relative_to(config.DATA_DIR).as_posix()
        if not include_all and ("/archive/" in rel or rel.startswith("docs/en/")):
            continue
        if f.name.lower() == "readme.md":
            continue
        yield f, rel


def split_markdown(text: str):
    """去掉 frontmatter，按标题切成 (heading_path, lines) 小节列表。"""
    text = text.strip()
    title = ""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            m = FRONTMATTER_TITLE.search(text[3:end])
            if m:
                title = m.group(1).strip().strip('"')
            text = text[end + 4 :].lstrip()

    sections, stack, cur = [], [], None
    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m and len(m.group(1)) <= SPLIT_LEVEL:
            level, head = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, head))
            cur = (" > ".join(t for _, t in stack), [line])
            sections.append(cur)
        else:
            if cur is None:
                if not line.strip():
                    continue
                cur = (title or "（开头）", [])
                sections.append(cur)
            cur[1].append(line)
    return title, sections


def split_long(content: str, max_chars: int = config.MAX_CHARS):
    """超长小节按段落再切；连续 | 开头的表格行视为一个段落整体。"""
    blocks, buf, in_table = [], [], False
    for line in content.splitlines():
        is_table = line.lstrip().startswith("|")
        if in_table and not is_table:
            blocks.append("\n".join(buf))
            buf, in_table = [], False
        if is_table:
            in_table = True
        buf.append(line)
    if buf:
        blocks.append("\n".join(buf))

    pieces, size = [], 0
    for b in (x.strip("\n") for x in blocks):
        if not b.strip():
            continue
        if pieces and size + len(b) <= max_chars:
            pieces[-1] += "\n\n" + b
            size += len(b) + 2
        else:
            pieces.append(b)
            size = len(b)

    final = []  # 兜底：整张大表格等单个块仍超长时按行硬切
    for p in pieces:
        if len(p) <= max_chars * 2:
            final.append(p)
            continue
        buf = ""
        for line in p.splitlines():
            if buf and len(buf) + len(line) + 1 > max_chars:
                final.append(buf)
                buf = line
            else:
                buf = f"{buf}\n{line}" if buf else line
        if buf:
            final.append(buf)
    return final


def chunk_file(path, rel):
    """单个 md 文件 -> (outline, [(heading, text), ...])"""
    _, sections = split_markdown(path.read_text(encoding="utf-8"))
    chunks = []
    for head, lines in sections:
        content = "\n".join(lines).strip()
        if not content:
            continue
        pieces = split_long(content)
        for i, p in enumerate(pieces):
            label = head if len(pieces) == 1 else f"{head}（{i + 1}/{len(pieces)}）"
            chunks.append((label, p))

    merged = []  # 过短的小节并入前一块，避免碎片
    for head, text in chunks:
        if merged and len(text) < config.MIN_CHARS:
            merged[-1] = (merged[-1][0], merged[-1][1] + "\n\n" + text)
        else:
            merged.append((head, text))

    heads = []  # 文档大纲：给 --contextual 做上下文输入
    for head, _ in merged:
        head = re.sub(r"（\d+/\d+）$", "", head)
        if head not in heads:
            heads.append(head)
    return " / ".join(heads), merged


def build_contexts(con, all_chunks):
    """为每个 chunk 生成一句话上下文（哈希缓存），返回与 all_chunks 等长的 list。"""
    if not config.API_KEY:
        sys.exit("--contextual 需要调用对话模型：请先在 .env 配置 LLM_API_KEY，或去掉该参数。")
    texts = [t for _, _, t, _ in all_chunks]
    hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest()[:16] for t in texts]
    cached = store.get_cached_ctx(con, hashes)
    todo = [i for i, h in enumerate(hashes) if h not in cached]
    print(f"上下文生成：缓存命中 {len(hashes) - len(todo)}，需新生成 {len(todo)}")
    ctxs = [cached.get(h, "") for h in hashes]
    failed = 0
    for start in range(0, len(todo), CTX_BATCH):
        idxs = todo[start : start + CTX_BATCH]
        outs = llm.contextualize(
            [all_chunks[i][3] for i in idxs], [texts[i] for i in idxs]
        )
        for i, o in zip(idxs, outs):
            if o:
                ctxs[i] = o
        new_pairs = [(hashes[i], o) for i, o in zip(idxs, outs) if o]
        if new_pairs:
            store.put_cached_ctx(con, new_pairs)
        failed += sum(1 for o in outs if not o)
        print(f"  上下文 {min(start + CTX_BATCH, len(todo))}/{len(todo)}")
    if failed:
        print(f"提示：{failed} 个 chunk 上下文生成失败（未缓存），重跑本命令可自动补齐。")
    return ctxs


def embed_with_cache(con, inputs: list[str]) -> list[bytes]:
    """按内容哈希缓存的向量化：只算新增，逐批落库，中断重跑自动续。"""
    if not config.API_KEY:
        sys.exit("向量化需要调用 API：请先在 .env 配置 LLM_API_KEY，或用 --chunk-only。")
    hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest()[:16] for t in inputs]
    cached = store.get_cached_vecs(con, hashes)
    todo = [i for i, h in enumerate(hashes) if h not in cached]
    print(f"向量化：缓存命中 {len(hashes) - len(todo)}，需新生成 {len(todo)}")
    blobs = {i: cached[h] for i, h in enumerate(hashes) if h in cached}
    done = 0
    for start in range(0, len(todo), config.EMBED_BATCH):
        idxs = todo[start : start + config.EMBED_BATCH]
        try:
            vecs = llm.embed_batch([inputs[i] for i in idxs])
        except Exception as e:
            print(f"\nAPI 连续失败（{type(e).__name__}）：已完成批次均已缓存，"
                  f"解决后重跑 python ingest.py 即可续跑。")
            raise
        for i, v in zip(idxs, vecs):
            blobs[i] = struct.pack(f"<{len(v)}f", *v)
        store.put_cached_vecs(con, [(hashes[i], blobs[i]) for i in idxs])
        done += len(idxs)
        print(f"  向量化 {done}/{len(todo)}")
    return [blobs[i] for i in range(len(inputs))]


def show_stats():
    con = store.connect()
    files, n, nv, nc = con.execute(
        "SELECT COUNT(DISTINCT source), COUNT(*), COUNT(vector), "
        "SUM(CASE WHEN ctx != '' THEN 1 ELSE 0 END) FROM chunks"
    ).fetchone()
    print(f"索引：{files} 个文件，{n} 个 chunk（向量化 {nv}，带上下文 {nc or 0}）")
    print(f"位置：{config.INDEX_PATH}")


def main():
    ap = argparse.ArgumentParser(description="RAG 建库")
    ap.add_argument("--dry-run", action="store_true", help="只切分打印统计，不调用 API")
    ap.add_argument("--chunk-only", action="store_true", help="切分入库但不向量化（无 key 时可先测 BM25）")
    ap.add_argument("--contextual", action="store_true", help="入库前生成上下文句（需 key，有缓存）")
    ap.add_argument("--stats", action="store_true", help="只看当前索引状态")
    ap.add_argument("--all", action="store_true", help="包含英文版 docs/en 与 archive")
    args = ap.parse_args()

    if args.stats:
        show_stats()
        return
    if not config.DATA_DIR.exists():
        sys.exit(f"语料不存在：{config.DATA_DIR}\n先执行 git clone https://github.com/byoungd/up.git \"{config.DATA_DIR}\"")

    all_chunks = []  # (rel, heading, text, outline)
    for f, rel in iter_corpus_files(args.all):
        outline, merged = chunk_file(f, rel)
        for head, text in merged:
            all_chunks.append((rel, head, text, outline))
    sizes = [len(t) for _, _, t, _ in all_chunks]
    print(f"语料 {len(set(r for r, _, _, _ in all_chunks))} 个文件，"
          f"切出 {len(all_chunks)} 个 chunk，平均 {sum(sizes) // len(sizes)} 字符，最大 {max(sizes)}")
    if args.dry_run:
        for rel, head, text, _ in all_chunks[:5]:
            print("-" * 50)
            print(f"[{rel}] {head}\n{text[:150]}...")
        return

    con = store.connect()
    ctxs = build_contexts(con, all_chunks) if args.contextual else [""] * len(all_chunks)

    blobs = None
    if not args.chunk_only:
        embed_inputs = [f"{c}\n{text}" if c else text
                        for (_, _, text, _), c in zip(all_chunks, ctxs)]
        blobs = embed_with_cache(con, embed_inputs)

    # 全部准备好才动库：任何一步失败都不影响现有索引
    store.reset(con)
    store.set_meta(con, "source_repo", "https://github.com/byoungd/up")
    if blobs:
        store.set_meta(con, "embed_model", config.EMBED_MODEL)
        store.set_meta(con, "embed_dim", str(len(blobs[0]) // 4))
    rows = [
        (rel, head, text, i, ctx, blobs[i] if blobs else None)
        for i, ((rel, head, text, _), ctx) in enumerate(zip(all_chunks, ctxs))
    ]
    store.add(con, rows)
    mode = "切分入库（未向量化）" if args.chunk_only else "完整建库"
    print(f"完成（{mode}）：{len(rows)} 个 chunk 已写入 {config.INDEX_PATH}")


if __name__ == "__main__":
    main()
