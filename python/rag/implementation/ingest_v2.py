"""
ingest_v2.py — Production RAG Ingestion
- Đọc tài liệu từ local folder (txt, md, pdf, docx)
- Chunk bằng rule-based (RecursiveCharacterTextSplitter)
- Embedding: dense (YeScale) + sparse (BM25 via FastEmbed)
- Lưu vector vào Qdrant (async) với tenant isolation
- Rate limiting trước mỗi batch embedding
- Retry policy cho 429 / 5xx / timeout
"""

import os
import re
import uuid
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timezone

from docx import Document
from openai import RateLimitError, APITimeoutError, InternalServerError, AsyncOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm import tqdm
from tenacity import (
    retry, wait_exponential, stop_after_attempt,
    retry_if_exception_type,
)
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, PayloadSchemaType,
    SparseVectorParams, SparseIndexParams, SparseVector,
    PointIdsList,
)
from fastembed import SparseTextEmbedding
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("ingest")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

load_dotenv(override=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — giống hệt answer_v2.py để đồng bộ
# ═══════════════════════════════════════════════════════════════════════════════

EMBEDDING_API_KEY    = os.getenv("RAG_EMBEDDING_API_KEY", "")
RAG_LLM_API_KEY      = os.getenv("RAG_OPEN_AI", "")
EMBEDDING_MODEL      = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_BASE_URL   = os.getenv("EMBEDDING_BASE_URL", "https://api.yescale.io/v1")
QDRANT_URL           = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY       = os.getenv("QDRANT_API_KEY")

VECTOR_DIM    = 1024
BATCH_SIZE    = int(os.getenv("INGEST_BATCH_SIZE", "30"))   
WORKERS       = int(os.getenv("INGEST_WORKERS", "8"))

# ── Rate Limiting ──────────────────────────────────────────────────────────────
RATE_LIMIT_EMBED  = int(os.getenv("RATE_LIMIT_EMBED", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

# ── Retry policies ──────────────────────────────────────────────────────────────
wait_rate  = wait_exponential(multiplier=10, min=30, max=180)
stop_rate  = stop_after_attempt(6)
wait_srv   = wait_exponential(multiplier=5,  min=10, max=60)
stop_srv   = stop_after_attempt(4)
wait_net   = wait_exponential(multiplier=2,  min=4,  max=30)
stop_net   = stop_after_attempt(5)

# ═══════════════════════════════════════════════════════════════════════════════
# CLIENTS — async để match answer_v2
# ═══════════════════════════════════════════════════════════════════════════════

openai_client = AsyncOpenAI(base_url=EMBEDDING_BASE_URL, api_key=EMBEDDING_API_KEY)

qdrant = AsyncQdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
)

sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

# ── Rate Limiter (sliding window) ─────────────────────────────────────────────
_rate_limit_times: list[float] = []
_rate_limit_lock: asyncio.Lock | None = None
async def _check_rate_limit():
    """Kiểm tra và sleep nếu vượt rate limit."""
    global _rate_limit_times, _rate_limit_lock

    if _rate_limit_lock is None:
        _rate_limit_lock = asyncio.Lock()

    async with _rate_limit_lock: 
        now = datetime.now(timezone.utc).timestamp()
        window = now - RATE_LIMIT_WINDOW
        _rate_limit_times = [t for t in _rate_limit_times if t > window]
        
        if len(_rate_limit_times) >= RATE_LIMIT_EMBED:
            oldest = min(_rate_limit_times)
            wait_sec = max(1, int(oldest + RATE_LIMIT_WINDOW - now) + 2)
            logger.warning(f"⏳ Ingest rate limit — sleeping {wait_sec}s")
            await asyncio.sleep(wait_sec) 
            _rate_limit_times = []
            
        _rate_limit_times.append(datetime.now(timezone.utc).timestamp())


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS — reuse từ answer_v2
# ═══════════════════════════════════════════════════════════════════════════════

BLOCKLISTED = {'admin', 'config', 'system', 'root', 'drop', 'delete'}


def safe_collection_name(tenant_id: str) -> str:
    """Hash + sanitize tenant_id → collection name an toàn."""
    raw = re.sub(r'[^a-zA-Z0-9_-]', '_', tenant_id.lower())[:59]
    if any(b in raw for b in BLOCKLISTED):
        raw = f"user_{raw}"
    return f"tenant_{raw}" if len(raw) >= 3 else "tenant_default"


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class Result(BaseModel):
    page_content: str
    metadata: dict


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT SPLITTING
# ═══════════════════════════════════════════════════════════════════════════════

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,     # tăng từ 800 → 1200 (phù hợp content gen, context đủ dài)
    chunk_overlap=200,   # tăng từ 150 → 200 (không mất context ở seam)
    length_function=len,
    separators=["\n\n", "\n", r"(?<=\. )", " ", ""],
)


# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENT LOADING
# ═══════════════════════════════════════════════════════════════════════════════

SUPPORTED_EXT = {".txt", ".md", ".docx", ".pdf"}


def load_documents(folder_path: str) -> list[dict]:
    """Load tất cả file hỗ trợ từ folder."""
    folder = Path(folder_path)
    if not folder.exists():
        logger.error(f"Thư mục không tồn tại: {folder_path}")
        return []

    docs = []
    for file in folder.rglob("*"):
        if file.suffix.lower() not in SUPPORTED_EXT:
            continue

        try:
            if file.suffix.lower() == ".docx":
                doc  = Document(file)
                text = "\n".join(p.text for p in doc.paragraphs)
            elif file.suffix.lower() == ".pdf":
                text = pymupdf4llm.to_markdown(str(file))
            else:
                text = file.read_text(encoding="utf-8", errors="replace")

            if text.strip():
                docs.append({
                    "source": file.name,
                    "type":   file.suffix.lstrip("."),
                    "text":   text,
                })
                logger.info(f"  ✅ Loaded: {file.name} ({len(text):,} chars)")

        except Exception as e:
            logger.warning(f"  ⚠️  Bỏ qua {file.name}: {e}")

    logger.info(f"  Total: {len(docs)} file(s)")
    return docs


# ═══════════════════════════════════════════════════════════════════════════════
# CHUNKING
# ═══════════════════════════════════════════════════════════════════════════════

def chunk_documents(documents: list[dict]) -> list[Result]:
    """Cắt documents thành chunks dùng RecursiveCharacterTextSplitter."""
    all_chunks = []
    for doc in tqdm(documents, desc="Chunking Documents"):
        chunks_text = text_splitter.split_text(doc["text"])
        for chunk in chunks_text:
            if chunk.strip():
                all_chunks.append(Result(
                    page_content=chunk.strip(),
                    metadata={"source": doc["source"], "type": doc["type"]},
                ))
    logger.info(f"  Total chunks: {len(all_chunks)}")
    return all_chunks


# ═══════════════════════════════════════════════════════════════════════════════
# QDRANT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

_collection_cache: set[str] = set()


async def ensure_collection(col: str):
    """Tạo collection nếu chưa tồn tại (dense + sparse vectors)."""
    if col in _collection_cache:
        return

    response = await qdrant.get_collections()
    existing = {c.name for c in response.collections}

    if col not in existing:
        await qdrant.create_collection(
            collection_name=col,
            vectors_config={
                "dense": VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
            },
        )
        await qdrant.create_payload_index(col, "source", PayloadSchemaType.KEYWORD)
        await qdrant.create_payload_index(col, "type",   PayloadSchemaType.KEYWORD)
        logger.info(f"  Created collection (dense+sparse): {col}")

    _collection_cache.add(col)


async def replace_collection(col: str):
    """Xóa collection cũ nếu tồn tại."""
    response = await qdrant.get_collections()
    existing = {c.name for c in response.collections}
    if col in existing:
        await qdrant.delete_collection(col)
        _collection_cache.discard(col)
        logger.info(f"  Dropped old collection: {col}")


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING + UPSERT
# ═══════════════════════════════════════════════════════════════════════════════
@retry(
    wait=wait_rate, stop=stop_rate,
    retry=retry_if_exception_type((RateLimitError,)), 
    reraise=True,
)
@retry(
    wait=wait_srv, stop=stop_srv,
    retry=retry_if_exception_type((InternalServerError,)), 
    reraise=True,
)
@retry(
    wait=wait_net, stop=stop_net,
    retry=retry_if_exception_type((APITimeoutError, asyncio.TimeoutError, OSError)),
    reraise=True,
)
async def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed 1 batch texts có rate limiting + retry."""
    await _check_rate_limit()
    resp = await openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        dimensions=VECTOR_DIM,
    )
    return [d.embedding for d in resp.data]


def _sparse_batch(texts: list[str]) -> list[dict]:
    """Tính BM25 sparse vectors."""
    results = list(sparse_model.embed(texts))
    return [
        {"indices": r.indices.tolist(), "values": r.values.tolist()}
        for r in results
    ]


async def upsert_chunks(tenant_id: str, chunks: list[Result], replace: bool = True):
    """Embed và upsert chunks vào Qdrant."""
    col = safe_collection_name(tenant_id)

    if replace:
        await replace_collection(col)

    await ensure_collection(col)

    total = len(chunks)
    sem = asyncio.Semaphore(WORKERS)

    async def process_single_batch(batch_chunks: list[Result]):
        async with sem:
            texts = [c.page_content for c in batch_chunks]

            # 1. Embed dense (gọi API)
            dense_vecs = await _embed_batch(texts)

            # 2. Embed sparse (BM25 chạy CPU local, đẩy vào thread để không block)
            sparse_vecs = await asyncio.to_thread(_sparse_batch, texts)

            # 3. Tạo payload
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense":  d_vec,
                        "sparse": SparseVector(
                            indices=s_vec["indices"],
                            values=s_vec["values"],
                        ),
                    },
                    payload={"page_content": text, **chunk.metadata},
                )
                for text, d_vec, s_vec, chunk
                in zip(texts, dense_vecs, sparse_vecs, batch_chunks)
            ]

            # 4. Đẩy lên Qdrant
            await qdrant.upsert(collection_name=col, points=points)

    logger.info(f"Đang đẩy {total} chunks lên Qdrant...")
    
    # 🟢 Gom tất cả các batch lại và chạy đồng loạt
    tasks = [
        process_single_batch(chunks[i : i + BATCH_SIZE]) 
        for i in range(0, total, BATCH_SIZE)
    ]
    
    # Dùng tqdm bọc bên ngoài gather để có thanh tiến trình cực mượt
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Upserting [{tenant_id}]"):
        await f 

    count_result = await qdrant.count(col)
    logger.info(f"  ✅ {col}: {count_result.count:,} vectors stored")


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

async def ingest_user(tenant_id: str, folder_path: str, replace: bool = True):
    """
    Load → Chunk → Embed → Upsert vào Qdrant cho 1 tenant.

    Args:
        tenant_id:   ID của user/tenant (sẽ được hash → collection name)
        folder_path: Đường dẫn thư mục chứa tài liệu
        replace:     Xóa collection cũ trước khi ingest
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Ingest started — tenant={tenant_id}  folder={folder_path}")

    t0 = datetime.now(timezone.utc)

    # Load
    docs = await asyncio.to_thread(load_documents, folder_path)
    if not docs:
        logger.warning("⚠️  Không có tài liệu nào. Dừng.")
        return

    # Chunk
    chunks = await asyncio.to_thread(chunk_documents, docs)

    # Embed + Upsert
    await upsert_chunks(tenant_id, chunks, replace=replace)

    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    logger.info(f"  ✅ Done in {elapsed:.1f}s — tenant={tenant_id}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# DELETE: xóa chunks theo source filename trong Qdrant
# ═══════════════════════════════════════════════════════════════════════════════

async def delete_chunks_by_source(tenant_id: str, filename: str) -> int:
    """
    Xóa tất cả chunks trong Qdrant collection của tenant mà có payload.source == filename.

    Trả về số chunks đã xóa.
    Nếu collection không tồn tại → trả về 0.
    """
    col = safe_collection_name(tenant_id)

    response = await qdrant.get_collections()
    existing = {c.name for c in response.collections}
    if col not in existing:
        return 0

    # Lấy tất cả points có source == filename
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    results, _ = await qdrant.scroll(
        collection_name=col,
        scroll_filter=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=filename))]
        ),
        limit=10000,
        with_payload=False,
        with_vectors=False,
    )

    if not results:
        return 0

    ids_to_delete = [pt.id for pt in results]
    await qdrant.delete(
        collection_name=col,
        points_selector=PointIdsList(points=ids_to_delete),
    )
    _collection_cache.discard(col)
    logger.info(f"  🗑️  Deleted {len(ids_to_delete)} chunks for source='{filename}' in {col}")
    return len(ids_to_delete)


async def delete_user(tenant_id: str, filename: str) -> int:
    """Async wrapper — xóa chunks khỏi Qdrant theo filename."""
    return await delete_chunks_by_source(tenant_id, filename)


def delete_user_sync(tenant_id: str, filename: str) -> int:
    """Sync wrapper cho những chỗ gọi đồng bộ."""
    return asyncio.run(delete_user(tenant_id, filename))


# ── Sync wrapper cho backward compatibility ─────────────────────────────────
def ingest_user_sync(tenant_id: str, folder_path: str, replace: bool = True):
    """Wrapper sync cho những chỗ gọi đồng bộ."""
    asyncio.run(ingest_user(tenant_id, folder_path, replace=replace))


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG Ingest CLI")
    parser.add_argument("-t", "--tenant", default="user_test_001", help="Tenant ID")
    parser.add_argument("-f", "--folder", default="./implementation/test_data", help="Folder path")
    parser.add_argument("--no-replace", action="store_true", help="Không xóa collection cũ")
    args = parser.parse_args()

    ingest_user_sync(
        tenant_id=args.tenant,
        folder_path=args.folder,
        replace=not args.no_replace,
    )
