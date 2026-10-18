"""
ingest.py - build / update the FAISS vector database.

Usage (run from the project folder):
    python ingest.py                          # ingest ./data  ->  ./vectordb
    python ingest.py data                     # same, explicit
    python ingest.py D:\\docs\\a.md  D:\\more_docs   # any files / folders (.md, .txt)
    python ingest.py --rebuild                # throw everything away and re-embed all
    python ingest.py --search "mock test negative marking"   # quick retrieval test

Incremental: every file is hashed (SHA-256). Unchanged files are NOT re-embedded,
changed files are re-embedded, deleted files are removed from the DB.

Output folder  ./vectordb :
    index.faiss      FAISS index (cosine similarity via normalised vectors)
    chunks.jsonl     one JSON line per chunk: text + metadata (source, title, heading, url)
    embeddings.npy   the raw vectors (lets us rebuild the index without re-embedding)
    manifest.json    embedding model, dimension, file hashes, build time
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import importlib

ROOT = Path(__file__).parent.resolve()
DB_DIR = ROOT / "vectordb"
DEFAULT_INPUT = ROOT / "data"
CACHE_DIR = ROOT / ".fastembed_cache"

# English model, small + fast (384 dims). For Urdu / Roman-Urdu questions see the README notes
# in the chat: swap to "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2".
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

CHUNK_MAX_CHARS = 1100   # ~250-300 tokens: precise retrieval, well under the model's 512-token limit
OVERLAP_LINES = 2        # lines repeated between neighbouring chunks of the same section
SUPPORTED = {".md", ".txt"}

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
HEADING_RE = re.compile(r"^(#{1,4})\s+(.*\S)")
URL_RE = re.compile(r"(?:URL|Sources?)\s*:\s*\**\s*(https?://[^\s)\]*]+)", re.I)
TABLE_SEP_RE = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")


# --------------------------------------------------------------------------- helpers
def _numpy():
    """Load NumPy when a database operation actually needs it."""
    try:
        return importlib.import_module("numpy")
    except ImportError as exc:
        raise SystemExit("NumPy is required for ingestion and search; install it with: python -m pip install numpy") from exc


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel_key(path: Path) -> str:
    """Stable id for a file: path relative to the project folder (or absolute if outside)."""
    path = path.resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def key_to_path(key: str) -> Path:
    p = Path(key)
    return p if p.is_absolute() else ROOT / p


def collect_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        p = p.resolve()
        if p.is_dir():
            files += [f for f in sorted(p.rglob("*")) if f.is_file() and f.suffix.lower() in SUPPORTED]
        elif p.is_file() and p.suffix.lower() in SUPPORTED:
            files.append(p)
        else:
            print(f"  [skip] not found / unsupported type: {p}")
    return files


# --------------------------------------------------------------------------- chunking
def _table_header(lines: list[str]) -> list[str]:
    for i in range(len(lines) - 1):
        if lines[i].lstrip().startswith("|") and TABLE_SEP_RE.match(lines[i + 1]):
            return [lines[i], lines[i + 1]]
    return []


def _pack_lines(lines: list[str], max_chars: int, overlap: int, header: list[str]) -> list[list[str]]:
    """Greedy line packing. Long tables are split with the header row repeated in every chunk."""
    chunks: list[list[str]] = []
    buf: list[str] = []
    size = 0
    for line in lines:
        add = len(line) + 1
        if buf and size + add > max_chars:
            chunks.append(buf)
            if header and line.lstrip().startswith("|"):
                tail = [x for x in buf[-overlap:] if x.lstrip().startswith("|") and x not in header]
                buf = list(header) + tail
            else:
                buf = buf[-overlap:] if overlap else []
            size = sum(len(x) + 1 for x in buf)
        buf.append(line)
        size += add
    if buf:
        chunks.append(buf)
    return chunks


def chunk_document(text: str, key: str, file_hash: str) -> list[dict]:
    """Markdown-aware chunking: split on headings first, then pack lines up to CHUNK_MAX_CHARS.
    Every chunk is prefixed with  [Document title > Heading path]  so it is self-explanatory."""
    title = Path(key).stem.replace("_", " ")
    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m and len(m.group(1)) == 1:
            title = m.group(2).strip()
            break
    m_url = URL_RE.search(text)
    doc_url = m_url.group(1).rstrip(".,;") if m_url else ""

    sections: list[tuple[str, list[str]]] = []
    path: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m:
            if any(l.strip() for l in current):
                sections.append((" > ".join(path), current))
            level, heading = len(m.group(1)), m.group(2).strip()
            # the H1 is the document title (already shown in the label), so it is not part of the path
            path = [] if level == 1 else path[: level - 2] + [heading]
            current = []
        elif line.strip() not in ("---", "***"):
            current.append(line)
    if any(l.strip() for l in current):
        sections.append((" > ".join(path), current))

    out: list[dict] = []
    for heading, lines in sections:
        while lines and not lines[0].strip():
            lines = lines[1:]
        m_sec = URL_RE.search("\n".join(lines))
        url = m_sec.group(1).rstrip(".,;") if m_sec else doc_url
        for piece in _pack_lines(lines, CHUNK_MAX_CHARS, OVERLAP_LINES, _table_header(lines)):
            body = "\n".join(piece).strip()
            if len(body) < 25:
                continue
            out.append({
                "source": key,
                "title": title,
                "heading": heading,
                "url": url,
                "file_hash": file_hash,
                "text": f"[{title} > {heading}]\n{body}" if heading else f"[{title}]\n{body}",
            })
    return out


# --------------------------------------------------------------------------- embeddings
def add_prefix(texts: list[str], model_name: str, kind: str) -> list[str]:
    name = model_name.lower()
    if "e5" in name:
        return [("query: " if kind == "query" else "passage: ") + t for t in texts]
    if "bge" in name and kind == "query":
        return [BGE_QUERY_PREFIX + t for t in texts]
    return texts


def load_embedder(model_name: str):
    from fastembed import TextEmbedding  # imported lazily so --help is instant

    CACHE_DIR.mkdir(exist_ok=True)
    return TextEmbedding(model_name=model_name, cache_dir=str(CACHE_DIR))


def embed_texts(embedder, texts: list[str], model_name: str, kind: str = "passage") -> np.ndarray:
    import faiss

    np = _numpy()
    vecs = np.array(list(embedder.embed(add_prefix(texts, model_name, kind), batch_size=32)), dtype="float32")
    faiss.normalize_L2(vecs)  # normalised vectors: inner product == cosine similarity
    return vecs


# --------------------------------------------------------------------------- DB I/O
def load_state(model_name: str, rebuild: bool) -> dict:
    """Return {source_key: {"hash": str, "chunks": [...], "emb": ndarray}} from the existing DB."""
    np = _numpy()
    mf_path = DB_DIR / "manifest.json"
    if rebuild or not (mf_path.exists() and (DB_DIR / "chunks.jsonl").exists() and (DB_DIR / "embeddings.npy").exists()):
        return {}
    manifest = json.loads(mf_path.read_text(encoding="utf-8"))
    if manifest.get("model") != model_name:
        print(f"  Embedding model changed ({manifest.get('model')} -> {model_name}): rebuilding everything.")
        return {}
    chunks = [json.loads(l) for l in (DB_DIR / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    emb = np.load(DB_DIR / "embeddings.npy")
    state: dict = {}
    for i, c in enumerate(chunks):
        s = state.setdefault(c["source"], {"hash": c["file_hash"], "chunks": [], "rows": []})
        s["chunks"].append(c)
        s["rows"].append(i)
    for s in state.values():
        s["emb"] = emb[s["rows"]]
        del s["rows"]
    return state


def save_db(state: dict, model_name: str, dim: int) -> int:
    import faiss

    np = _numpy()
    chunks: list[dict] = []
    embs: list[np.ndarray] = []
    for key in sorted(state):
        for c in state[key]["chunks"]:
            chunks.append({**c, "id": len(chunks)})
        embs.append(state[key]["emb"])
    matrix = np.vstack(embs).astype("float32")

    index = faiss.IndexFlatIP(dim)  # exact search - perfect for a few thousand chunks
    index.add(matrix)

    DB_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(DB_DIR / "index.faiss"))
    np.save(DB_DIR / "embeddings.npy", matrix)
    with open(DB_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    manifest = {
        "model": model_name,
        "dim": dim,
        "chunk_count": len(chunks),
        "chunk_max_chars": CHUNK_MAX_CHARS,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": {k: v["hash"] for k, v in sorted(state.items())},
    }
    (DB_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(chunks)


# --------------------------------------------------------------------------- commands
def ingest(paths: list[Path], model_name: str, rebuild: bool) -> None:
    files = collect_files(paths)
    if not files:
        sys.exit("No .md / .txt files found to ingest.")

    state = load_state(model_name, rebuild)
    for key in list(state):  # forget files that no longer exist on disk
        if not key_to_path(key).exists():
            print(f"  [removed] {key}")
            del state[key]

    pending: list[tuple[str, str, list[dict]]] = []
    for f in files:
        key, h = rel_key(f), sha256_file(f)
        if key in state and state[key]["hash"] == h:
            print(f"  [unchanged] {key}")
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_document(text, key, h)
        print(f"  [{'updated' if key in state else 'new'}] {key}: {len(chunks)} chunks")
        pending.append((key, h, chunks))

    dim = None
    if pending:
        all_texts = [c["text"] for _, _, chs in pending for c in chs]
        print(f"Embedding {len(all_texts)} chunks with {model_name} (first run downloads the model)...")
        vecs = embed_texts(load_embedder(model_name), all_texts, model_name)
        dim = vecs.shape[1]
        pos = 0
        for key, h, chs in pending:
            state[key] = {"hash": h, "chunks": chs, "emb": vecs[pos: pos + len(chs)]}
            pos += len(chs)
    elif state:
        dim = next(iter(state.values()))["emb"].shape[1]
    else:
        sys.exit("Nothing to store.")

    total = save_db(state, model_name, dim)
    print(f"\nDone. {len(state)} files, {total} chunks -> {DB_DIR}")
    if not pending:
        print("(No embedding was needed - everything was already up to date.)")


def search(query: str, k: int = 5) -> None:
    import faiss

    manifest = json.loads((DB_DIR / "manifest.json").read_text(encoding="utf-8"))
    index = faiss.read_index(str(DB_DIR / "index.faiss"))
    chunks = [json.loads(l) for l in (DB_DIR / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    q = embed_texts(load_embedder(manifest["model"]), [query], manifest["model"], kind="query")
    scores, ids = index.search(q, k)
    for s, i in zip(scores[0], ids[0]):
        c = chunks[i]
        print(f"\n{s:.3f}  {c['source']}  |  {c['heading']}\n{c['text'][:300]}...")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build/update the FAISS vector DB from .md/.txt files.")
    ap.add_argument("paths", nargs="*", type=Path, help="files or folders (default: ./data)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"fastembed model (default {DEFAULT_MODEL})")
    ap.add_argument("--rebuild", action="store_true", help="ignore the existing DB and re-embed everything")
    ap.add_argument("--search", metavar="QUERY", help="test retrieval against the existing DB and exit")
    args = ap.parse_args()
    if args.search:
        search(args.search)
    else:
        ingest(args.paths or [DEFAULT_INPUT], args.model, args.rebuild)


if __name__ == "__main__":
    main()
