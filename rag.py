"""Core RAG pipeline for financial PDFs. Fully local: LM Studio + ChromaDB."""
import os
import re

import chromadb
import pymupdf
from openai import OpenAI
from rank_bm25 import BM25Okapi

DB_PATH = "chroma_db"
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")
LLM_URL = os.getenv("LLM_URL", "http://localhost:1234/v1")  # LM Studio local server
client = OpenAI(base_url=LLM_URL, api_key="lm-studio")

_client = chromadb.PersistentClient(path=DB_PATH)
_col = _client.get_or_create_collection("fin_docs", metadata={"hnsw:space": "cosine"})
_cache = {}

SYSTEM = (
    "You are a careful financial analyst assistant. Answer ONLY from the provided "
    "context. Cite every fact as [document p.PAGE]. Page labels ending in L or R mean "
    "the left or right half of a two-page spread. If the context only partly answers "
    "the question, report what it does show and clearly say what is missing. Say "
    "'Not found in the provided documents.' only if nothing relevant is in the context. "
    "Never guess numbers. Quote numbers exactly with units and period, and show any "
    "calculation step by step. A block starting with COMPUTED RATIOS was calculated by "
    "code from the financial statements: for ratio questions use those values as-is, "
    "never recompute them, and mention the formula."
)


# ---------- parsing ----------
def _regions(page):
    """Annual reports are often two-page spreads. Read left and right halves separately."""
    r = page.rect
    if r.width > 1.2 * r.height:
        mid = r.width / 2
        return [("L", pymupdf.Rect(0, 0, mid, r.height)), ("R", pymupdf.Rect(mid, 0, r.width, r.height))]
    return [("", r)]


def parse_pdf(path):
    pages = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, 1):
            for side, clip in _regions(page):
                text = page.get_text("text", clip=clip, sort=True)
                tables = []
                if len(re.findall(r"\d[\d,.]*", text)) >= 20:  # only number-heavy halves can hold tables
                    try:
                        for t in page.find_tables(clip=clip).tables:
                            md = t.to_markdown()
                            if md and md.strip():
                                tables.append(md)
                    except Exception:
                        pass
                pages.append({"page": f"{i}{side}", "text": text, "tables": tables})
    return pages


def chunk_pages(pages, size=1500, overlap=200):
    chunks = []
    for p in pages:
        text = re.sub(r"[ \t]{3,}", "   ", p["text"])
        text = re.sub(r"\n{2,}", "\n", text).strip()
        start = 0
        while start < len(text):
            piece = text[start:start + size].strip()
            if piece:
                chunks.append({"page": p["page"], "kind": "text", "text": piece})
            if start + size >= len(text):
                break
            start += size - overlap
        for t in p["tables"]:
            chunks.append({"page": p["page"], "kind": "table", "text": t[:4000]})
    return chunks


# ---------- embeddings + storage ----------
def embed(texts, batch=16):
    out = []
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + batch])
        out.extend(d.embedding for d in resp.data)
    return out


def ingest_pdf(path, doc_name):
    """Index one PDF. Returns (num_chunks, looks_scanned)."""
    pages = parse_pdf(path)
    looks_scanned = sum(len(p["text"]) for p in pages) < 50 * max(len(pages), 1)
    chunks = chunk_pages(pages)
    if not chunks:
        return 0, True
    _col.delete(where={"doc": doc_name})
    _cache.clear()
    embs = embed(["search_document: " + c["text"] for c in chunks])
    _col.add(
        ids=[f"{doc_name}-{i}" for i in range(len(chunks))],
        embeddings=embs,
        documents=[c["text"] for c in chunks],
        metadatas=[{"doc": doc_name, "page": c["page"], "kind": c["kind"]} for c in chunks],
    )
    _cache.clear()
    return len(chunks), looks_scanned


def list_docs():
    metas = _col.get(include=["metadatas"])["metadatas"]
    return sorted({m["doc"] for m in metas})


def delete_doc(doc_name):
    _col.delete(where={"doc": doc_name})
    _cache.clear()


# ---------- hybrid retrieval (vector + BM25, fused with RRF) ----------
def _tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def _state():
    n = _col.count()
    if _cache.get("n") != n:
        data = _col.get(include=["documents", "metadatas"])
        _cache.update(
            n=n, ids=data["ids"], docs=data["documents"], metas=data["metadatas"],
            bm25=BM25Okapi([_tok(d) for d in data["documents"]]) if data["ids"] else None,
        )
    return _cache


def _bm25_rank(question, docs, n):
    st = _state()
    if not st.get("bm25"):
        return []
    scores = st["bm25"].get_scores(_tok(question))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out = []
    for i in order:
        if scores[i] <= 0:
            break
        if docs and st["metas"][i]["doc"] not in docs:
            continue
        out.append(st["ids"][i])
        if len(out) >= n:
            break
    return out


def retrieve(question, docs=None, k=6):
    total = _col.count()
    if total == 0:
        return []
    qe = embed(["search_query: " + question])[0]
    where = None
    if docs:
        where = {"doc": docs[0]} if len(docs) == 1 else {"doc": {"$in": docs}}
    r = _col.query(query_embeddings=[qe], n_results=min(k * 3, total), where=where)
    rankings = [r["ids"][0], _bm25_rank(question, docs, k * 3)]
    fused = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            fused[cid] = fused.get(cid, 0) + 1 / (60 + rank)
    top = sorted(fused, key=fused.get, reverse=True)[:k]
    st = _state()
    idx = {cid: i for i, cid in enumerate(st["ids"])}
    return [
        {"text": st["docs"][idx[c]], "doc": st["metas"][idx[c]]["doc"],
         "page": st["metas"][idx[c]]["page"], "kind": st["metas"][idx[c]]["kind"]}
        for c in top if c in idx
    ]


# ---------- generation ----------
def answer_stream(question, ctx, model, extra=""):
    context = (extra + "\n\n" if extra else "") + "\n\n".join(
        f"[{c['doc']} p.{c['page']}] ({c['kind']})\n{c['text']}" for c in ctx
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    stream = client.chat.completions.create(
        model=model, messages=messages, stream=True, temperature=0.1,
    )
    for part in stream:
        if part.choices and part.choices[0].delta.content:
            yield part.choices[0].delta.content


def chat_models():
    try:
        names = [m.id for m in client.models.list().data]
    except Exception:
        return []
    return [n for n in names if "embed" not in n.lower()]
