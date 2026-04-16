"""
answer_v2.py — RAG Answer Engine
- Hybrid search (dense + sparse + RRF fusion)
- Multi-query expansion
- BGE Cross-Encoder rerank
- Redis cache
- Rate limiting (global + per-tenant) + wave processing cho batch lớn
- Smart retry: riêng cho 429 (rate limit), 500, network fail
"""

import os
import asyncio
import hashlib
import re
import math
import logging
from datetime import datetime, timezone

from dotenv import load_dotenv
from pydantic import BaseModel
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)
from sentence_transformers import CrossEncoder

from openai import AsyncOpenAI, RateLimitError, APITimeoutError
from litellm import acompletion
from qdrant_client import AsyncQdrantClient
import redis.asyncio as redis

from qdrant_client.models import (
    Filter, FieldCondition, MatchValue,
    SparseVector, Prefetch,
)
from qdrant_client.http import models as rest_models

logger = logging.getLogger("answer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

load_dotenv(override=True)

# ── Debug ──────────────────────────────────────────────────────────────────────
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
if DEBUG_MODE:
    import litellm
    litellm._turn_on_debug()
    logger.warning("⚠️  DEBUG MODE ON — API keys có thể xuất hiện trong log!")

# ── API Keys ───────────────────────────────────────────────────────────────────
EMBEDDING_API_KEY = os.getenv("RAG_EMBEDDING_API_KEY", "")
RAG_LLM_API_KEY   = os.getenv("RAG_OPEN_AI", "")

RAG_MODEL        = os.getenv("RAG_MODEL")
REWRITE_MODEL   = os.getenv("REWRITE_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL")
LLM_BASE_URL    = os.getenv("LLM_BASE_URL")

# ── Redis ───────────────────────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))

# ── Validation ─────────────────────────────────────────────────────────────────
MAX_Q_LEN   = 1000
BLOCKLISTED  = {'admin', 'config', 'system', 'root', 'drop', 'delete'}
SUSPICIOUS   = [
    r'exec\(', r'eval\(', r'__import__', r'<script>', r'javascript:',
]

# ── Batch / Workers ─────────────────────────────────────────────────────────────
BATCH_MAX_WORKERS = int(os.getenv("BATCH_MAX_WORKERS", "5"))
QUESTION_TIMEOUT  = int(os.getenv("QUESTION_TIMEOUT", "120"))

# ── Semantic Cache ──────────────────────────────────────────────────────────────
SEMANTIC_CACHE_TTL  = int(os.getenv("SEMANTIC_CACHE_TTL", str(CACHE_TTL)))
SEMANTIC_SIM_THRESHOLD = float(os.getenv("SEMANTIC_SIM_THRESHOLD", "0.92"))

# ── Rate Limiting ──────────────────────────────────────────────────────────────
# Số request MAX đi ra API (embedding + LLM) trong 1 sliding window (giây)
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))
RATE_LIMIT_WINDOW   = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

# Số câu hỏi xử lý trong 1 wave trước khi nghỉ giữa wave
WAVE_SIZE = int(os.getenv("WAVE_SIZE", "50"))
WAVE_SLEEP = float(os.getenv("WAVE_SLEEP", "5.0"))   # giây nghỉ giữa 2 wave

# ── Retrieval ──────────────────────────────────────────────────────────────────
RETRIEVAL_K      = 15      # tăng từ 20 → 25 (pool rộng hơn cho reranker)
FINAL_K          = 5       # giảm từ 10 → 5 (tiết kiệm ~45% token, top-chunks đủ relevant)
MAX_RERANK_INPUT = 15      # giảm từ 20 → 15 (BGE v2-m3 đủ chính xác ở 15)

# ── Retry policies ──────────────────────────────────────────────────────────────
# Policy A: rate limit (429) — chờ lâu, nhiều lần thử
wait_rate_limit  = wait_exponential(multiplier=10, min=30, max=180)
stop_rate_limit  = stop_after_attempt(8)

# Policy B: lỗi server (500/502/503) — chờ vừa, ít lần thử hơn
wait_server_err  = wait_exponential(multiplier=5, min=10, max=60)
stop_server_err  = stop_after_attempt(4)

# Policy C: timeout / network — chờ ít, thử nhanh
wait_timeout      = wait_exponential(multiplier=2, min=4, max=30)
stop_timeout      = stop_after_attempt(5)

# ── Clients ─────────────────────────────────────────────────────────────────────
openai_client = AsyncOpenAI(base_url=EMBEDDING_BASE_URL, api_key=EMBEDDING_API_KEY)

qdrant = AsyncQdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
)

redis_client = redis.from_url(
    REDIS_URL,
    decode_responses=True,
    health_check_interval=10,
    socket_timeout=10,
    socket_connect_timeout=5,
    retry_on_timeout=True,
)

# ── Rate Limiter Lua Script (atomic — prevents race conditions) ─────────────────
# Atomically: remove old entries → check count → add only if under limit
_RATE_LIMITER_SCRIPT = """
local key = KEYS[1]
local max_requests = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local window_start = now - window

redis.call('ZREMRANGEBYSCORE', key, 0, window_start)
local current = redis.call('ZCARD', key)

if current < max_requests then
    redis.call('ZADD', key, now, now)
    redis.call('EXPIRE', key, window + 1)
    return 0
else
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    if #oldest >= 2 then
        return math.ceil(oldest[2] + window - now) + 1
    else
        return window + 1
    end
end
"""


class TokenBucketRateLimiter:
    """
    Atomic sliding-window rate limiter using Redis + Lua script.
    - No race conditions (atomic check-and-add in single Redis operation)
    - Returns exact wait time when at limit → sleeps that exact duration
    - Gracefully degrades if Redis is unavailable
    """

    def __init__(self, requests_per_window: int, window_seconds: int):
        self.max_requests = requests_per_window
        self.window        = window_seconds
        self._redis        = redis_client
        self._redis_key    = "ratelimit:global"
        self._script_sha: str | None = None  # cached script SHA

    async def _ensure_script(self):
        """Load Lua script into Redis once, reuse via SHA (faster)."""
        if self._script_sha:
            return
        self._script_sha = await self._redis.script_load(_RATE_LIMITER_SCRIPT)

    async def _wait_for_slot(self, max_attempts: int = 3) -> bool:
        """Poll until a rate-limit slot opens (up to max_attempts). Returns True if allowed."""
        for _ in range(max_attempts):
            now = datetime.now(timezone.utc).timestamp()
            wait_sec: int = await self._redis.evalsha(
                self._script_sha, 1,
                self._redis_key,
                str(self.max_requests),
                str(self.window),
                str(now),
            )
            if wait_sec == 0:
                return True
            await asyncio.sleep(wait_sec)
        logger.warning(f"⚠️  Still at rate limit after {max_attempts} attempts — proceeding anyway")
        return False

    async def acquire(self):
        """Blocking cho đến khi được phép gọi API."""
        if not self._redis:
            return  # no Redis → skip rate limiting

        try:
            await self._ensure_script()
            await self._wait_for_slot()
        except Exception as e:
            # NOSCRIPT = Redis was restarted → reload script
            if "NOSCRIPT" in str(e):
                self._script_sha = None
                try:
                    await self._ensure_script()
                    await self._wait_for_slot()
                except Exception as e2:
                    logger.warning(f"Rate limiter retry failed: {e2}")
            else:
                logger.warning(f"Rate limiter error (continuing without limit): {e}")

    async def close(self):
        if self._redis:
            await self._redis.close()


# Singleton rate limiter
_global_limiter: TokenBucketRateLimiter | None = None


def get_rate_limiter() -> TokenBucketRateLimiter:
    global _global_limiter
    if _global_limiter is None:
        _global_limiter = TokenBucketRateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW)
    return _global_limiter


# ── System prompt ───────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a knowledgeable and helpful AI assistant.
You answer questions based ONLY on the provided context from the knowledge base or documents.

Rules:
- Use only the information from the provided context.
- Do not use outside knowledge.
- Do not make up information.
- If the answer is not in the context, say:
  "I'm sorry, but I don't have that information in the provided documents."
- If the context contains partial information, answer with what is available.
- Keep answers clear, accurate, and concise.

Context:
{context}

Answer:
"""

# ── Pydantic models ──────────────────────────────────────────────────────────────
class Result(BaseModel):
    page_content: str
    metadata: dict


# ── Cache helpers ───────────────────────────────────────────────────────────────
async def get_cached(key: str):
    if not redis_client: return None
    try: return await redis_client.get(f"rag:{key}")
    except: return None

async def set_cache(key: str, value: str):
    if not redis_client: return
    try: await redis_client.setex(f"rag:{key}", CACHE_TTL, value)
    except: pass

async def batch_get_cached(keys: list[str]):
    if not redis_client: return {}
    try:
        async with redis_client.pipeline() as pipeline:
            for key in keys:
                pipeline.get(f"rag:{key}")
            results = await pipeline.execute()
        return {key: result for key, result in zip(keys, results) if result is not None}
    except Exception as e:
        logger.warning(f"Lỗi lấy cache batch: {e}")
        return {}

async def batch_set_cache(data: dict[str, str]):
    if not redis_client or not data: return
    try:
        async with redis_client.pipeline() as pipeline:
            for key, value in data.items():
                pipeline.setex(f"rag:{key}", CACHE_TTL, value)
            await pipeline.execute()
    except Exception as e:
        logger.warning(f"Lỗi ghi cache batch: {e}")


# ── Semantic Cache (embedding-based) ──────────────────────────────────────────
def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _embed_question(text: str) -> list[float]:
    """Embed a question using the embedding API (with rate limiting)."""
    limiter = get_rate_limiter()
    await limiter.acquire()
    resp = await openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
        dimensions=1024,
    )
    return resp.data[0].embedding


async def get_cached_semantic(question: str, tenant_id: str) -> tuple[str | None, float]:
    """
    Semantic cache lookup.
    Returns (cached_answer, similarity_score) if similarity >= SEMANTIC_SIM_THRESHOLD,
    else (None, 0.0).
    """
    if not redis_client:
        return None, 0.0

    try:
        cache_key_prefix = make_cache_key(question, tenant_id)
        embed = await _embed_question(question)

        # Scan all semantic cache entries for this tenant
        pattern = f"semcache:{tenant_id}:*"
        cursor = 0
        while True:
            cursor, keys = await redis_client.scan(cursor, match=pattern, count=100)
            for key in keys:
                stored = await redis_client.hgetall(key)
                if not stored:
                    continue
                stored_embed = stored.get("embedding", "")
                if not stored_embed:
                    continue
                stored_vec = [float(x) for x in stored_embed.split(",")]
                sim = _cosine_sim(embed, stored_vec)
                if sim >= SEMANTIC_SIM_THRESHOLD:
                    answer = stored.get("answer", "")
                    logger.info(f"🎯 Semantic cache HIT — sim={sim:.3f} >= {SEMANTIC_SIM_THRESHOLD}")
                    return answer, sim
            if cursor == 0:
                break
        return None, 0.0
    except Exception as e:
        logger.warning(f"Semantic cache lookup failed: {e}")
        return None, 0.0


async def set_cached_semantic(question: str, tenant_id: str, answer: str):
    """Store question embedding + answer in semantic cache."""
    if not redis_client:
        return
    try:
        embed = await _embed_question(question)
        cache_key = make_cache_key(question, tenant_id)
        redis_key = f"semcache:{tenant_id}:{cache_key}"
        await redis_client.hset(
            redis_key,
            mapping={
                "embedding": ",".join(str(x) for x in embed),
                "answer":    answer,
            },
        )
        await redis_client.expire(redis_key, SEMANTIC_CACHE_TTL)
    except Exception as e:
        logger.warning(f"Semantic cache write failed: {e}")


# ── Validation ─────────────────────────────────────────────────────────────────
def validate_question(q: str) -> tuple[bool, str]:
    if not q or len(q.strip()) < 2:
        return False, "Câu hỏi quá ngắn."
    if len(q) > MAX_Q_LEN:
        return False, f"Câu hỏi quá dài (tối đa {MAX_Q_LEN} ký tự)."
    for p in SUSPICIOUS:
        if re.search(p, q, re.IGNORECASE):
            return False, "Câu hỏi không hợp lệ."
    return True, ""

def safe_collection_name(tenant_id: str) -> str:
    clean = re.sub(r'[^a-zA-Z0-9_-]', '_', tenant_id.lower())[:59]
    if any(b in clean for b in BLOCKLISTED):
        clean = f"user_{clean}"
    return f"tenant_{clean}" if len(clean) >= 3 else "tenant_default"

def make_cache_key(question: str, tenant_id: str) -> str:
    return hashlib.md5(f"{tenant_id}:{question.strip().lower()}".encode()).hexdigest()


# ── Qdrant helpers ─────────────────────────────────────────────────────────────
_collection_cache: set[str] = set()
_collection_lock: asyncio.Lock | None = None


async def collection_exists(tenant_id: str) -> bool:
    global _collection_lock
    if _collection_lock is None:
        _collection_lock = asyncio.Lock()

    col = safe_collection_name(tenant_id)
    if col in _collection_cache:
        return True
    async with _collection_lock:
        if col in _collection_cache:
            return True
        response = await qdrant.get_collections()
        existing = {c.name for c in response.collections}
        if col in existing:
            _collection_cache.add(col)
            return True
    return False


# ── Embedding ───────────────────────────────────────────────────────────────────
from fastembed import SparseTextEmbedding

sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")


async def embed(text: str) -> list[float]:
    limiter = get_rate_limiter()
    await limiter.acquire()

    response = await openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
        dimensions=1024,
    )
    return response.data[0].embedding


# ── Fetch context (hybrid search) ──────────────────────────────────────────────
async def fetch_context_unranked(
    tenant_id: str,
    question: str,
    doc_type_filter: str | None = None,
) -> list[Result]:
    col = safe_collection_name(tenant_id)

    dense_vec = await embed(question)

    sparse_result_list = await asyncio.to_thread(
        lambda: list(sparse_model.query_embed(question))
    )
    sparse_result = sparse_result_list[0]
    sparse_vec = SparseVector(
        indices=sparse_result.indices.tolist(),
        values=sparse_result.values.tolist(),
    )

    query_filter = None
    if doc_type_filter:
        query_filter = Filter(
            must=[FieldCondition(key="type", match=MatchValue(value=doc_type_filter))]
        )

    try:
        hits = (await qdrant.query_points(
            collection_name=col,
            prefetch=[
                Prefetch(query=dense_vec, using="dense", limit=RETRIEVAL_K),
                Prefetch(query=sparse_vec, using="sparse", limit=RETRIEVAL_K),
            ],
            query=rest_models.FusionQuery(fusion=rest_models.Fusion.RRF),
            limit=RETRIEVAL_K,
            query_filter=query_filter,
            with_payload=True,
        )).points
    except Exception as e:
        logger.warning(f"⚠️ Hybrid failed ({e}), fallback dense")
        hits = (await qdrant.query_points(
            collection_name=col,
            query=dense_vec,
            using="dense",
            limit=RETRIEVAL_K,
            query_filter=query_filter,
            with_payload=True,
        )).points

    return [
        Result(
            page_content=h.payload.get("page_content", ""),
            metadata={
                "source": h.payload.get("source", ""),
                "type":   h.payload.get("type", ""),
                "score":  h.score,
            },
        )
        for h in hits
    ]


# ── Reranker ────────────────────────────────────────────────────────────────────
logger.info("Loading CrossEncoder Reranker model...")
reranker_model = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512)
logger.info("Reranker model loaded successfully.")


def rerank(question: str, chunks: list[Result]) -> list[Result]:
    if not chunks:
        return []
    pairs = [[question, chunk.page_content] for chunk in chunks]
    scores = reranker_model.predict(pairs)
    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    if DEBUG_MODE:
        for i, (chunk, score) in enumerate(scored_chunks[:3]):
            logger.info(f"Top {i+1} Score {score:.4f}: {chunk.page_content[:50]}...")
    return [chunk for chunk, score in scored_chunks]


# ── QueryClassifierAgent ─────────────────────────────────────────────────────────
# Smart: decides when multi-query expansion is needed.
# Cost savings: simple questions (80% of traffic) skip the LLM rewrite entirely.
# vs. old generate_multi_queries: always called LLM, always did 3 extra searches.

_QUERY_CLASSIFIER_PROMPT = """Classify this question into ONE type:

- simple: keyword or short phrase (< 5 words), no complex structure
  Examples: "bảo hành", "chính sách đổi trả", "Insurellm là gì"
- general: broad topic needing multiple aspects covered
  Examples: "bảo hiểm nhân thọ hoạt động như thế nào", "các loại sản phẩm bảo hiểm"
- specific: focused question with clear entity/constraint
  Examples: "thời hạn bảo hành sản phẩm X là bao lâu", "ai được hưởng bảo hiểm khi người mua qua đời"
- ambiguous: vague, missing context, or multiple meanings
  Examples: "nó có tốt không", "chính sách mới", "liên quan đến khách hàng"

Question: {question}

Reply with exactly ONE word: simple | general | specific | ambiguous"""


_QUERY_REWRITE_PROMPT = """Given the question below, generate 2 alternative phrasings that cover DIFFERENT ASPECTS (not synonyms).

Examples of aspect expansion:
- "bảo hành" → "thời hạn bảo hành" and "quy trình bảo hành"
- "bảo hiểm nhân thọ" → "lợi ích bảo hiểm nhân thọ" and "phí bảo hiểm nhân thọ"

Question: {question}

Reply with one alternative per line (no numbers, no explanation):"""


def _retry_decorator(fn):
    """Shared retry decorators for LLM calls."""
    @retry(wait=wait_rate_limit, stop=stop_rate_limit,
           retry=retry_if_exception_type((RateLimitError,)), reraise=True)
    @retry(wait=wait_server_err, stop=stop_server_err, reraise=True)
    @retry(wait=wait_timeout, stop=stop_timeout,
           retry=retry_if_exception_type((APITimeoutError, asyncio.TimeoutError, OSError)),
           reraise=True)
    async def wrapper(*args, **kwargs):
        return await fn(*args, **kwargs)
    return wrapper


@_retry_decorator
async def _classify_question(question: str) -> str:
    """Agent: classify question type. Returns: simple | general | specific | ambiguous."""
    limiter = get_rate_limiter()
    await limiter.acquire()
    prompt = _QUERY_CLASSIFIER_PROMPT.format(question=question)
    response = await acompletion(
        model=REWRITE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        api_key=RAG_LLM_API_KEY,
        api_base=LLM_BASE_URL,
        max_retries=0,
    )
    content = response.choices[0].message.content.strip().lower()
    valid = {"simple", "general", "specific", "ambiguous"}
    return content if content in valid else "general"


@_retry_decorator
async def _rewrite_for_aspects(question: str) -> list[str]:
    """Agent: expand general/ambiguous question into aspect-based sub-queries."""
    limiter = get_rate_limiter()
    await limiter.acquire()
    prompt = _QUERY_REWRITE_PROMPT.format(question=question)
    response = await acompletion(
        model=REWRITE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        api_key=RAG_LLM_API_KEY,
        api_base=LLM_BASE_URL,
        max_retries=0,
    )
    content = response.choices[0].message.content
    queries = [q.strip() for q in content.split('\n') if q.strip()]
    return queries[:2] if queries else []


def merge_chunks(primary, secondary):
    seen   = {c.page_content for c in primary}
    merged = primary[:]
    for c in secondary:
        if c.page_content not in seen:
            merged.append(c)
            seen.add(c.page_content)
    return merged


# ── Full fetch with smart query expansion ──────────────────────────────────────
# Smart flow:
#   simple/specific → skip rewrite → 1 search
#   general/ambiguous → aspect expansion → 3 searches (parallel)
async def fetch_context(
    tenant_id: str,
    question: str,
    history: list[dict] | None = None,
    skip_rewrite: bool = False,
) -> list[Result]:
    if skip_rewrite:
        chunks = await fetch_context_unranked(tenant_id, question)
        return (await asyncio.to_thread(rerank, question, chunks[:MAX_RERANK_INPUT]))[:FINAL_K]

    # Step 1: Classify question type (1 LLM call — cheap)
    qtype = await _classify_question(question)

    if qtype in ("simple", "specific"):
        # Fast path: no rewrite needed, direct search
        chunks = await fetch_context_unranked(tenant_id, question)
        return (await asyncio.to_thread(rerank, question, chunks[:MAX_RERANK_INPUT]))[:FINAL_K]

    # Step 2: general/ambiguous → expand into aspect queries (1 LLM call, 2 sub-queries)
    sub_queries = await _rewrite_for_aspects(question)
    all_queries = [question] + sub_queries

    # Step 3: Parallel search all queries
    fetch_tasks = [fetch_context_unranked(tenant_id, q) for q in all_queries]
    results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    # Step 4: Merge and deduplicate (filter exceptions, flatten results)
    seen: set[str] = set()
    all_chunks: list[Result] = []
    for r in results:
        if isinstance(r, Exception):
            logger.warning(f"Search failed: {r}")
            continue
        for c in r:
            if c.page_content not in seen:
                seen.add(c.page_content)
                all_chunks.append(c)

    reranked = await asyncio.to_thread(rerank, question, all_chunks)
    return reranked[:FINAL_K]


def make_rag_messages(question: str, history: list[dict], chunks: list[Result]) -> list[dict]:
    # Build context string
    if chunks:
        context_parts = [
            f"Source: {chunk.metadata.get('source', 'document')}\n{chunk.page_content}"
            for chunk in chunks
        ]
        context = "\n\n".join(context_parts)
    else:
        context = "(No relevant context found)"

    # Build system prompt as plain string
    system_prompt = (
        "You are a helpful AI assistant. Answer based ONLY on the provided context.\n"
        "If the answer is not in the context, say you don't know.\n"
        "For each claim, cite the source using [Nguồn: filename] notation.\n"
        "If no relevant context, respond: 'Không có thông tin trong tài liệu.'\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )

    messages: list[dict] = [{"role": "system", "content": str(system_prompt)}]

    # Add history — ensure each message has string content
    for msg in (history or []):
        content = msg.get("content", "")
        if isinstance(content, str):
            messages.append({"role": str(msg.get("role", "user")), "content": content})
        elif isinstance(content, dict) and "text" in content:
            messages.append({"role": str(msg.get("role", "user")), "content": str(content["text"])})

    messages.append({"role": "user", "content": str(question)})
    return messages


# ── LLM Answer ─────────────────────────────────────────────────────────────────
@retry(
    wait=wait_rate_limit,
    stop=stop_rate_limit,
    retry=retry_if_exception_type((RateLimitError,)),
    reraise=True,
)
@retry(
    wait=wait_server_err,
    stop=stop_server_err,
    reraise=True,
)
@retry(
    wait=wait_timeout,
    stop=stop_timeout,
    retry=retry_if_exception_type((APITimeoutError, asyncio.TimeoutError, OSError)),
    reraise=True,
)
async def call_llm(messages: list[dict]) -> str:
    limiter = get_rate_limiter()
    await limiter.acquire()

    response = await acompletion(
        model=RAG_MODEL,
        messages=messages,
        api_key=RAG_LLM_API_KEY,
        api_base=LLM_BASE_URL,
        max_retries=0,
    )
    return response.choices[0].message.content


# ── Single question answer ───────────────────────────────────────────────────────
async def answer_question(
    question: str,
    history: list[dict] | None = None,
    tenant_id: str = "default",
) -> tuple[str, list[Result]]:
    history = history or []
    ok, err = validate_question(question)
    if not ok:
        return err, []

    if not await collection_exists(tenant_id):
        return "⚠️ Chưa có dữ liệu...", []

    # 1. Exact cache (fastest path)
    cache_key = make_cache_key(question, tenant_id)
    cached    = await get_cached(cache_key)
    if cached:
        logger.info("🎯 Exact cache HIT")
        return cached, []

    # 2. Semantic cache (embed similarity >= threshold)
    sem_cached, sim = await get_cached_semantic(question, tenant_id)
    if sem_cached:
        logger.info(f"🎯 Semantic cache HIT — sim={sim:.3f}")
        return sem_cached, []

    # 3. Full pipeline: fetch → LLM → cache
    chunks   = await fetch_context(tenant_id, question, history)
    messages = make_rag_messages(question, history, chunks)
    answer   = await call_llm(messages)

    # Write to both caches (semantic uses embedding, exact uses md5 key)
    await asyncio.gather(
        set_cache(cache_key, answer),
        set_cached_semantic(question, tenant_id, answer),
    )

    return answer, chunks


# ── Wave processor (core batch logic) ──────────────────────────────────────────
async def _execute_batch_core(
    uncached_q: list[str],
    uncached_idx: list[int],
    tenant_id: str,
    max_workers: int,
    timeout: int,
) -> list[tuple[int, tuple[str, list[Result]]]]:
    """Hàm lõi thực thi việc gọi LLM cho một nhóm câu hỏi (Bỏ qua check cache)"""
    sem = asyncio.Semaphore(max_workers)

    async def _bounded_answer(idx: int, q: str):
        async with sem:
            try:
                answer, context = await asyncio.wait_for(
                    answer_question(q, [], tenant_id),
                    timeout=timeout,
                )
                return idx, (answer, context)
            except asyncio.TimeoutError:
                logger.error(f"⏰ Timeout: câu {idx + 1}")
                return idx, ("⏰ Câu hỏi xử lý quá lâu.", [])
            except RateLimitError as e:
                logger.error(f"🚫 Rate limit: câu {idx + 1} — {e}")
                return idx, ("🚫 Bị giới hạn rate limit, thử lại sau.", [])
            except Exception as e:
                logger.error(f"❌ Lỗi câu {idx + 1}: {e}")
                return idx, (f"Lỗi: {str(e)}", [])

    tasks = [_bounded_answer(idx, q) for idx, q in zip(uncached_idx, uncached_q)]
    processed = await asyncio.gather(*tasks, return_exceptions=True)

    # Lọc bỏ exception cứng và trả về kết quả
    results_filtered: list[tuple[int, tuple[str, list[Result]]]] = []
    for item in processed:
        if isinstance(item, Exception):
            continue
        results_filtered.append(item)  # type: ignore[arg-type]
    return results_filtered


async def batch_answer_questions_waved(
    uncached_q: list[str],
    uncached_idx: list[int],
    tenant_id: str,
    max_workers: int,
    timeout: int,
    wave_size: int = WAVE_SIZE,
    wave_sleep: float = WAVE_SLEEP,
) -> list[tuple[int, tuple[str, list[Result]]]]:
    """
    Xử lý batch lớn bằng wave processing.

    Thay vì bắn 150 câu cùng lúc, chia thành các wave nhỏ hơn,
    mỗi wave nghỉ 1 khoảng để API "hồi phục" quota.
    """
    total = len(uncached_q)
    all_results: list[tuple[int, tuple[str, list[Result]]]] = []

    for i in range(0, total, wave_size):
        wave_q = uncached_q[i : i + wave_size]
        wave_idx = uncached_idx[i : i + wave_size]
        wave_num = i // wave_size + 1

        logger.info(f"  🌊 Wave {wave_num}: {len(wave_q)} câu hỏi, {max_workers} workers")

        results = await _execute_batch_core(wave_q, wave_idx, tenant_id, max_workers, timeout)
        all_results.extend(results)

        # Nghỉ giữa wave (trừ wave cuối)
        remaining = total - (i + wave_size)
        if remaining > 0:
            logger.info(f"  😴 Wave {wave_num} done — sleeping {wave_sleep}s ({remaining} câu còn lại)")
            await asyncio.sleep(wave_sleep)

    return all_results


# ── Batch answer (giữ nguyên interface cũ, bên trong dùng waved) ──────────────
async def batch_answer_questions(
    questions: list[str],
    tenant_id: str = "default",
    max_workers: int = BATCH_MAX_WORKERS,
    timeout: int = QUESTION_TIMEOUT,
    wave_size: int = WAVE_SIZE,
    wave_sleep: float = WAVE_SLEEP,
) -> list[tuple[str, list[Result]]]:
    """
    Process multiple questions concurrently.

    - Dùng wave processing nếu > WAVE_SIZE câu
    - Dùng asyncio.Semaphore để giới hạn concurrency trong mỗi wave
    - Timeout riêng cho từng câu hỏi
    - Retry riêng cho từng loại lỗi (429 / 5xx / timeout)
    - Cache hit → trả ngay không gọi API
    """
    if not questions:
        return []

    # ── Cache lookup trước ────────────────────────────────────────────────────
    cache_keys   = [make_cache_key(q, tenant_id) for q in questions]
    cached_map   = await batch_get_cached(cache_keys)

    uncached_q    = []
    uncached_idx  = []
    results: list[tuple | None] = [None] * len(questions)

    for idx, (q, ck) in enumerate(zip(questions, cache_keys)):
        if ck in cached_map:
            results[idx] = (cached_map[ck], [])
        else:
            uncached_q.append(q)
            uncached_idx.append(idx)

    if not uncached_q:
        return results

    # ── Wave processing cho batch lớn ─────────────────────────────────────────
    if len(uncached_q) > wave_size:
        logger.info(f"Batch {len(uncached_q)} câu — dùng wave processing")
        processed_items = await batch_answer_questions_waved(
            uncached_q, uncached_idx, tenant_id, max_workers, timeout,
            wave_size=wave_size, wave_sleep=wave_sleep,
        )
    else:
        processed_items = await _execute_batch_core(
            uncached_q, uncached_idx, tenant_id, max_workers, timeout
        )

    for idx, res in processed_items:
        results[idx] = res

    # ── Batch cache write (chỉ kết quả tốt) ─────────────────────────────────
    new_cache_data = {
        make_cache_key(questions[idx], tenant_id): results[idx][0]
        for idx in uncached_idx
        if results[idx] and not results[idx][0].startswith(("⚠️", "⏰", "🚫", "Lỗi"))
    }
    await batch_set_cache(new_cache_data)

    # ── Fallback cho None ──────────────────────────────────────────────────────
    for idx, r in enumerate(results):
        if r is None:
            results[idx] = ("⚠️ Câu hỏi bị hủy.", [])

    return results
