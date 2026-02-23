from typing import Any, Dict, List
import json
import logging
import re
import time

from langchain_core.runnables import RunnableLambda, RunnableSequence

try:
    from langchain.retrievers import ContextualCompressionRetriever as _CCR
except Exception:
    _CCR = None

try:
    from langchain.retrievers.document_compressors import (
        BaseDocumentCompressor as _BDC
    )
except Exception:
    _BDC = None


if _BDC is None:
    class BaseDocumentCompressor:  # type: ignore
        def compress_documents(self, documents, query=None, **kwargs):
            return documents
else:
    BaseDocumentCompressor = _BDC


if _CCR is None:
    class ContextualCompressionRetriever:  # type: ignore
        def __init__(self, *, base_retriever, base_compressor):
            self.base_retriever = base_retriever
            self.base_compressor = base_compressor

        def get_relevant_documents(self, query: str):
            docs = self.base_retriever.get_relevant_documents(query)
            return self.base_compressor.compress_documents(docs, query=query)
else:
    ContextualCompressionRetriever = _CCR

from app.pipeline.vector_store import search_experience

from .retrievers import HierarchicalRetriever
from .tools import build_tool_parallel, collect_tool_references

logger = logging.getLogger(__name__)


class NoOpCompressor(BaseDocumentCompressor):
    def compress_documents(self, documents, query=None, **kwargs):
        return documents


PRIMARY_MIN_SIM = 0.08
SECONDARY_MIN_SIM = 0.04

INTENT_DISSATISFIED = "dissatisfied_retry"
INTENT_LATEST_SOCIAL = "latest_social_updates"
INTENT_MEMORY = "memory_followup"
INTENT_CLARIFY = "clarification_needed"
INTENT_DOCS = "docs_lookup"
INTENT_WEBSITE = "website_lookup"
INTENT_LABELS = {
    INTENT_DISSATISFIED,
    INTENT_LATEST_SOCIAL,
    INTENT_MEMORY,
    INTENT_DOCS,
    INTENT_WEBSITE
}


def _avg_score(chunks: List[Dict[str, Any]]) -> float:
    if not chunks:
        return 0.0
    return sum(c.get("score", 0.0) for c in chunks) / max(len(chunks), 1)


def _append_trace(state: Dict[str, Any], step: str, start: float, detail: str | None = None):
    trace = list(state.get("trace") or [])
    entry = {
        "step": step,
        "ms": round((time.perf_counter() - start) * 1000, 2)
    }
    if detail:
        entry["detail"] = detail
    trace.append(entry)
    return trace


def _preview_text(value: Any, limit: int = 220) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = re.sub(r"\s+", " ", text)
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."


def _default_intent(reason: str = "Default document-grounded lookup path.") -> Dict[str, Any]:
    return {
        "intent_label": INTENT_DOCS,
        "intent_reason": reason,
        "intent_confidence": 0.6,
        "use_secondary_retrieval": False,
        "requires_social_search": False,
        "prefer_memory": False,
        "needs_clarification_intent": False,
        "doc_scope": None
    }


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return default


def _to_float(value: Any, default: float = 0.6) -> float:
    try:
        score = float(value)
    except Exception:
        return default
    if score < 0:
        return 0.0
    if score > 1:
        return 1.0
    return score


def _normalize_intent_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    aliases = {
        "dissatisfied": INTENT_DISSATISFIED,
        "dissatisfied_retry": INTENT_DISSATISFIED,
        "retry": INTENT_DISSATISFIED,
        "latest": INTENT_LATEST_SOCIAL,
        "latest_social_updates": INTENT_LATEST_SOCIAL,
        "social": INTENT_LATEST_SOCIAL,
        "memory": INTENT_MEMORY,
        "memory_followup": INTENT_MEMORY,
        "followup": INTENT_MEMORY,
        "website": INTENT_WEBSITE,
        "website_lookup": INTENT_WEBSITE,
        "site": INTENT_WEBSITE,
        "web": INTENT_WEBSITE,
        # Clarification intent is intentionally disabled to avoid sudden
        # clarification turns for broad company/domain questions.
        "clarification": INTENT_DOCS,
        "clarification_needed": INTENT_DOCS,
        "clarify": INTENT_DOCS,
        "docs": INTENT_DOCS,
        "docs_lookup": INTENT_DOCS,
        "document_lookup": INTENT_DOCS
    }
    normalized = aliases.get(label, label)
    if normalized in INTENT_LABELS:
        return normalized
    return INTENT_DOCS


def _extract_json_blob(raw_text: str) -> Dict[str, Any] | None:
    if not raw_text:
        return None
    text = raw_text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(text[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None

    return None


def _normalize_intent_payload(payload: Dict[str, Any] | None) -> Dict[str, Any]:
    base = _default_intent()
    if not isinstance(payload, dict):
        return base

    label = _normalize_intent_label(payload.get("intent_label"))
    reason = str(payload.get("intent_reason") or base["intent_reason"]).strip()
    confidence = _to_float(payload.get("intent_confidence"), default=base["intent_confidence"])

    use_secondary = _to_bool(payload.get("use_secondary_retrieval"), False)
    requires_social = _to_bool(payload.get("requires_social_search"), False)
    prefer_memory = _to_bool(payload.get("prefer_memory"), False)
    needs_clarify = _to_bool(payload.get("needs_clarification_intent"), False)
    doc_scope = payload.get("doc_scope")

    if label == INTENT_DISSATISFIED:
        use_secondary = True
    elif label == INTENT_LATEST_SOCIAL:
        requires_social = True
    elif label == INTENT_MEMORY:
        prefer_memory = True
    elif label == INTENT_WEBSITE:
        doc_scope = "website"
    elif label == INTENT_DOCS:
        use_secondary = False
        requires_social = False
        prefer_memory = False
        needs_clarify = False

    return {
        "intent_label": label,
        "intent_reason": reason,
        "intent_confidence": confidence,
        "use_secondary_retrieval": use_secondary,
        "requires_social_search": requires_social,
        "prefer_memory": prefer_memory,
        "needs_clarification_intent": needs_clarify,
        "doc_scope": doc_scope
    }


def _build_intent_prompt(
    query: str,
    conversation: List[Dict[str, Any]] | None = None
) -> str:
    memory = _format_conversation(conversation, max_turns=8, max_chars=1600) or "none"
    clean_query = (query or "").strip()
    return f"""
Classify user intent for a customer support retrieval system.
Return JSON only. No markdown, no prose.
Knowledge scope is strictly limited to this bot's sources:
- uploaded DOCS
- company social/web tool outputs
- conversation MEMORY
Do not apply global/world knowledge assumptions.

Allowed intent_label values:
- {INTENT_DISSATISFIED}: user is not satisfied and asks for a different/regenerated answer.
- {INTENT_LATEST_SOCIAL}: user asks for latest/current updates likely requiring social/web lookup.
- {INTENT_MEMORY}: user follow-up can be resolved using conversation memory/previous assistant response.
- {INTENT_DOCS}: default document-grounded query.
- {INTENT_WEBSITE}: user asks specifically about website pages or content (site, web page, website info).
Important: do not output "{INTENT_CLARIFY}".
For broad company/domain questions (example: "role of customer"), use "{INTENT_DOCS}".

Output JSON schema:
{{
  "intent_label": "<one allowed label>",
  "intent_reason": "<short reason>",
  "intent_confidence": <float 0..1>,
  "use_secondary_retrieval": <boolean>,
  "requires_social_search": <boolean>,
  "prefer_memory": <boolean>,
  "needs_clarification_intent": <boolean>,
  "doc_scope": "<null or website>"
}}

Routing constraints:
- If intent_label is "{INTENT_DISSATISFIED}", set use_secondary_retrieval=true.
- If intent_label is "{INTENT_LATEST_SOCIAL}", set requires_social_search=true.
- If intent_label is "{INTENT_MEMORY}", set prefer_memory=true.
- If intent_label is "{INTENT_WEBSITE}", set doc_scope="website".
- If intent_label is "{INTENT_DOCS}", set needs_clarification_intent=false.
- For this system, default to "{INTENT_DOCS}" instead of clarification.

Conversation:
{memory}

User query:
{clean_query}
""".strip()


def _is_ambiguous(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return True
    if len(q.split()) < 3:
        return True
    vague = {
        "why",
        "what",
        "how",
        "help",
        "explain",
        "details",
        "more",
        "info",
        "information"
    }
    return q in vague


def _is_website_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    patterns = [
        "website",
        "web site",
        "webpage",
        "web page",
        "homepage",
        "home page",
        "on your site",
        "on the site",
        "on your website",
        "on the website",
        "from your website",
        "from the website",
        "pricing page",
        "contact page",
        "about page"
    ]
    return any(p in q for p in patterns)


def _is_followup(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    phrases = [
        "i don't understand",
        "i dont understand",
        "didn't understand",
        "did not understand",
        "not clear",
        "confused",
        "can you clarify",
        "clarify",
        "explain again",
        "explain it again",
        "simpler",
        "in simple terms",
        "more detail",
        "elaborate",
        "what do you mean"
    ]
    return any(p in q for p in phrases)


def _last_assistant(conversation: List[Dict[str, Any]] | None) -> str | None:
    if not conversation:
        return None
    for item in reversed(conversation):
        if item.get("role") == "assistant" and item.get("content"):
            return item.get("content")
    return None


def _format_conversation(
    conversation: List[Dict[str, Any]] | None,
    max_turns: int = 6,
    max_chars: int = 1200
) -> str:
    if not conversation:
        return ""
    items = [c for c in conversation if c.get("content")]
    if not items:
        return ""
    items = items[-max_turns:]
    lines = []
    total = 0
    for item in items:
        role = (item.get("role") or "user").upper()
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        if len(content) > 400:
            content = content[:400].rstrip() + "..."
        line = f"{role}: {content}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)


def _format_tool_results(tool_results: Dict[str, Any] | None, max_items: int = 8) -> str:
    if not isinstance(tool_results, dict):
        return ""
    lines: list[str] = []
    for tool_name, value in tool_results.items():
        results = value.get("results") if isinstance(value, dict) else None
        if not results:
            continue
        for item in results[:max_items]:
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            platform = str(item.get("platform") or tool_name).strip()
            snippet = item.get("snippet") or item.get("summary") or item.get("content")
            snippet_text = _preview_text(snippet, limit=220) if snippet else ""
            if not title and not url:
                continue
            if url:
                if snippet_text:
                    lines.append(f"[{platform}] {title or url} | {url} | {snippet_text}")
                else:
                    lines.append(f"[{platform}] {title or url} | {url}")
            else:
                lines.append(f"[{platform}] {title}")
            if len(lines) >= max_items:
                return "\n".join(lines)
    return "\n".join(lines)


def _build_prompt(
    context: str,
    question: str,
    conversation: List[Dict[str, Any]] | None = None,
    followup_context: str | None = None,
    no_docs: bool = False,
    is_ambiguous: bool = False,
    tool_has_results: bool = False,
    tool_context: str = "",
    intent_label: str = INTENT_DOCS,
    intent_reason: str = ""
) -> str:
    memory_block = _format_conversation(conversation)
    followup_block = (followup_context or "").strip()
    docs_block = (context or "").strip()
    tools_block = (tool_context or "").strip()
    return f"""
You are a helpful assistant for a document QA system.
Decide whether to answer from DOCS, TOOLS, or MEMORY.

Rules:
- Use only the provided DOCS, TOOLS, and MEMORY.
- Do not use outside/world knowledge.
- If evidence is insufficient, state that you do not have enough verified information from this bot's knowledge base.
- Do not ask clarifying questions.
- Keep the answer concise and factual with no hallucinations.

Signals:
intent_label={intent_label}
intent_reason={intent_reason or "none"}
no_docs={no_docs}
is_ambiguous={is_ambiguous}
tool_has_results={tool_has_results}

MEMORY:
{memory_block or "none"}

FOLLOWUP_CONTEXT:
{followup_block or "none"}

TOOLS:
{tools_block or "none"}

DOCS:
{docs_block or "none"}

QUESTION:
{question}

Answer:
""".strip()


def _build_social_prompt(
    question: str,
    conversation: List[Dict[str, Any]] | None = None,
    tool_context: str = "",
    docs_context: str = "",
    no_docs: bool = False,
    tool_has_results: bool = False,
    intent_reason: str = ""
) -> str:
    memory_block = _format_conversation(conversation)
    tools_block = (tool_context or "").strip()
    docs_block = (docs_context or "").strip()
    return f"""
You are a friendly social updates assistant for a company.
Summarize recent/ongoing updates using ONLY the provided TOOLS (and MEMORY if relevant).
Be relatable and concise, like a helpful teammate.

Rules:
- Do not use outside/world knowledge.
- Prefer TOOLS for anything "latest" or time-sensitive.
- DOCS are optional background only; never invent dates or posts.
- If TOOLS are empty or look like placeholders/login walls, say you couldn't find verified recent updates.
- Avoid long lists; keep it tight.

Signals:
intent_reason={intent_reason or "none"}
no_docs={no_docs}
tool_has_results={tool_has_results}

MEMORY:
{memory_block or "none"}

TOOLS:
{tools_block or "none"}

DOCS (background only):
{docs_block or "none"}

QUESTION:
{question}

Answer:
""".strip()


def make_intent_classifier_node(call_intent_llm=None):
    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        query = state.get("query", "")
        conversation = state.get("conversation") or []
        logger.info(
            "IntentClassifier start | query=%s | conversation_turns=%d",
            _preview_text(query, 180),
            len(conversation)
        )
        intent = _default_intent("Intent classification unavailable.")

        if callable(call_intent_llm):
            prompt = _build_intent_prompt(query, conversation)
            logger.info(
                "IntentClassifier LLM request | prompt_chars=%d",
                len(prompt)
            )
            raw = call_intent_llm(prompt)
            if raw:
                logger.info(
                    "IntentClassifier LLM response | response_chars=%d | preview=%s",
                    len(raw),
                    _preview_text(raw)
                )
            else:
                logger.warning("IntentClassifier empty LLM response.")
            parsed = _extract_json_blob(raw or "")
            if parsed is not None:
                intent = _normalize_intent_payload(parsed)
                logger.info(
                    "IntentClassifier parse success | keys=%s",
                    ",".join(sorted(parsed.keys()))
                )
            else:
                intent = _default_intent(
                    "Could not parse LLM intent response; using docs lookup."
                )
                logger.warning(
                    "IntentClassifier parse failed | fallback_intent=%s",
                    intent["intent_label"]
                )
        else:
            logger.warning(
                "IntentClassifier callable missing | fallback_intent=%s",
                intent["intent_label"]
            )

        doc_scope = intent.get("doc_scope") or state.get("doc_scope")
        if not doc_scope and _is_website_query(query):
            doc_scope = "website"
            if intent["intent_label"] == INTENT_DOCS:
                intent["intent_label"] = INTENT_WEBSITE
                intent["intent_reason"] = (
                    "User query references the company website; narrowing retrieval scope."
                )

        detail = (
            f"intent={intent['intent_label']};"
            f"secondary={int(intent['use_secondary_retrieval'])};"
            f"social={int(intent['requires_social_search'])};"
            f"memory={int(intent['prefer_memory'])};"
            f"clarify={int(intent['needs_clarification_intent'])}"
        )
        logger.info(
            "IntentClassifier decision | intent=%s | confidence=%.2f | secondary=%s | social=%s | memory=%s | clarify=%s",
            intent["intent_label"],
            intent["intent_confidence"],
            int(intent["use_secondary_retrieval"]),
            int(intent["requires_social_search"]),
            int(intent["prefer_memory"]),
            int(intent["needs_clarification_intent"])
        )
        return {
            "intent_label": intent["intent_label"],
            "intent_reason": intent["intent_reason"],
            "intent_confidence": intent["intent_confidence"],
            "use_secondary_retrieval": intent["use_secondary_retrieval"],
            "requires_social_search": intent["requires_social_search"],
            "prefer_memory": intent["prefer_memory"],
            "needs_clarification_intent": intent["needs_clarification_intent"],
            "doc_scope": doc_scope,
            "trace": _append_trace(state, "IntentClassifier", start, detail)
        }

    return node


def make_check_feedback_node(embedder):
    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        query = state.get("query", "")
        query_embedding = embedder.encode(
            query,
            normalize_embeddings=True
        )
        experience_hit = search_experience(
            query_embedding,
            state.get("bot_id", "")
        )
        feedback_score = (
            experience_hit.get("feedback_score", 0)
            if isinstance(experience_hit, dict)
            else 0
        )
        needs_human = feedback_score < -2
        return {
            "query_embedding": query_embedding,
            "experience_hit": experience_hit,
            "needs_human": needs_human,
            "trace": _append_trace(
                state,
                "CheckFeedbackState",
                start,
                f"feedback={feedback_score}"
            )
        }

    return node


def make_semantic_memory_node():
    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        conversation = state.get("conversation") or []
        followup = bool(state.get("prefer_memory")) or _is_followup(state.get("query", ""))
        last_assistant = _last_assistant(conversation)
        followup_context = last_assistant if (followup and last_assistant) else None
        detail = "followup" if followup_context else "none"
        return {
            "followup_context": followup_context,
            "followup_intent": bool(followup_context),
            "trace": _append_trace(state, "SemanticMemoryLookup", start, detail)
        }

    return node


def _docs_to_chunks(docs):
    chunks = []
    for doc in docs:
        meta = doc.metadata or {}
        chunks.append({
            "chunk_ref": meta.get("chunk_ref"),
            "score": meta.get("score", 0.0),
            "cluster": meta.get("cluster"),
            "chunk_index": meta.get("chunk_index"),
            "text": doc.page_content,
            "source_type": meta.get("source_type"),
            "source_url": meta.get("source_url"),
            "pdf": meta.get("pdf")
        })
    return chunks


def _log_retrieval_chunks(stage: str, chunks: List[Dict[str, Any]], limit: int = 8) -> None:
    if not chunks:
        logger.info("%s retrieval chunks | count=0", stage)
        return

    logger.info("%s retrieval chunks | count=%d", stage, len(chunks))
    for idx, c in enumerate(chunks[: max(1, int(limit))], start=1):
        logger.info(
            (
                "%s chunk[%d] | ref=%s | score=%.4f | source_type=%s | "
                "cluster=%s | pdf=%s | url=%s | text='%s'"
            ),
            stage,
            idx,
            c.get("chunk_ref"),
            float(c.get("score", 0.0) or 0.0),
            c.get("source_type"),
            c.get("cluster"),
            c.get("pdf"),
            c.get("source_url"),
            _preview_text(c.get("text", ""), limit=260),
        )


def _log_tool_results(tool_results: Dict[str, Any] | None, limit_per_tool: int = 5) -> None:
    if not isinstance(tool_results, dict) or not tool_results:
        logger.info("ToolRetrieval details | no tool payload")
        return

    for tool_name, payload in tool_results.items():
        results = payload.get("results") if isinstance(payload, dict) else None
        if not results:
            logger.info("ToolRetrieval details | tool=%s | count=0", tool_name)
            continue

        logger.info(
            "ToolRetrieval details | tool=%s | count=%d",
            tool_name,
            len(results),
        )
        for idx, item in enumerate(results[: max(1, int(limit_per_tool))], start=1):
            logger.info(
                (
                    "ToolRetrieval item[%d] | tool=%s | title=%s | url=%s | "
                    "platform=%s | snippet='%s'"
                ),
                idx,
                tool_name,
                item.get("title"),
                item.get("url"),
                item.get("platform"),
                _preview_text(
                    item.get("snippet")
                    or item.get("summary")
                    or item.get("content"),
                    limit=220,
                ),
            )


def make_primary_retrieval_node(embedder):
    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        top_k = state.get("top_k", 5)
        logger.info(
            "PrimaryRetrieval request | query=%s | top_k=%s | doc_scope=%s",
            _preview_text(state.get("query", ""), 180),
            top_k,
            state.get("doc_scope"),
        )
        retriever = HierarchicalRetriever(
            embedder=embedder,
            client_id=state.get("client_id", ""),
            bot_id=state.get("bot_id", ""),
            top_clusters=2,
            top_chunks=top_k,
            precomputed_embedding=state.get("query_embedding"),
            source_filter=state.get("doc_scope"),
            enable_doc_bm25=True,
            enable_social_bm25=True
        )
        compression = ContextualCompressionRetriever(
            base_retriever=retriever,
            base_compressor=NoOpCompressor()
        )
        docs = compression.get_relevant_documents(state.get("query", ""))
        chunks = _docs_to_chunks(docs)
        _log_retrieval_chunks("PrimaryRetrieval", chunks)
        context = "\n\n".join(d.page_content for d in docs)
        followup_context = state.get("followup_context")
        if followup_context:
            if context:
                context = f"{followup_context}\n\n{context}"
            else:
                context = f"{followup_context}"
        avg_score = _avg_score(chunks)
        logger.info(
            "PrimaryRetrieval stats | chunks=%d | avg_score=%.4f | threshold=%.2f",
            len(chunks),
            avg_score,
            PRIMARY_MIN_SIM
        )
        if not chunks or avg_score < PRIMARY_MIN_SIM:
            logger.warning(
                "PrimaryRetrieval below threshold | chunks=%d | avg_score=%.4f -> escalate",
                len(chunks),
                avg_score
            )
            return {
                "chunks": [],
                "context": "",
                "regen_failed": False,
                "no_docs": True,
                "source_type": "docs",
                "trace": _append_trace(
                    state,
                    "PrimaryRetrieval",
                    start,
                    "no_docs"
                )
            }
        return {
            "chunks": chunks,
            "context": context,
            "trace": _append_trace(
                state,
                "PrimaryRetrieval",
                start,
                f"chunks={len(chunks)}"
            )
        }

    return node


def make_secondary_retrieval_node(embedder):
    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        top_k = state.get("top_k", 5)
        logger.info(
            "SecondaryRetrieval request | query=%s | top_k=%s | doc_scope=%s",
            _preview_text(state.get("query", ""), 180),
            top_k,
            state.get("doc_scope"),
        )
        exclude_set = set(state.get("exclude_chunk_refs") or [])
        if exclude_set:
            top_chunks = top_k + len(exclude_set)
        else:
            top_chunks = top_k

        retriever = HierarchicalRetriever(
            embedder=embedder,
            client_id=state.get("client_id", ""),
            bot_id=state.get("bot_id", ""),
            top_clusters=4,
            top_chunks=top_chunks,
            exclude_chunk_refs=exclude_set,
            precomputed_embedding=state.get("query_embedding"),
            source_filter=state.get("doc_scope"),
            enable_doc_bm25=True,
            enable_social_bm25=True
        )
        compression = ContextualCompressionRetriever(
            base_retriever=retriever,
            base_compressor=NoOpCompressor()
        )
        docs = compression.get_relevant_documents(state.get("query", ""))
        chunks = _docs_to_chunks(docs)
        _log_retrieval_chunks("SecondaryRetrieval", chunks)

        avg_score = _avg_score(chunks)
        logger.info(
            "SecondaryRetrieval stats | chunks=%d | avg_score=%.4f | threshold=%.2f",
            len(chunks),
            avg_score,
            SECONDARY_MIN_SIM
        )
        if not chunks or avg_score < SECONDARY_MIN_SIM:
            logger.warning(
                "SecondaryRetrieval below threshold | chunks=%d | avg_score=%.4f -> escalate",
                len(chunks),
                avg_score
            )
            return {
                "chunks": [],
                "context": "",
                "regen_failed": True,
                "no_docs": True,
                "source_type": "docs",
                "trace": _append_trace(
                    state,
                    "SecondaryRetrieval",
                    start,
                    "no_docs"
                )
            }

        context = "\n\n".join(d.page_content for d in docs)
        followup_context = state.get("followup_context")
        if followup_context:
            context = f"{followup_context}\n\n{context}"
        return {
            "chunks": chunks,
            "context": context,
            "regen_failed": False,
            "trace": _append_trace(
                state,
                "SecondaryRetrieval",
                start,
                f"chunks={len(chunks)}"
            )
        }

    return node


def make_tool_retrieval_node():
    tool_parallel = build_tool_parallel()

    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        tool_results = tool_parallel.invoke(state)
        references = collect_tool_references(tool_results)
        tool_has_results = any(
            (value.get("results") if isinstance(value, dict) else None)
            for value in (tool_results or {}).values()
        )
        _log_tool_results(tool_results)
        return {
            "tool_results": tool_results,
            "tool_attempted": True,
            "tool_has_results": tool_has_results,
            "references": references,
            "trace": _append_trace(
                state,
                "ToolRetrieval",
                start,
                "results" if tool_has_results else "empty"
            )
        }

    return node


def make_analyzer_node():
    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        chunks = state.get("chunks") or []
        confidence = _avg_score(chunks)

        experience_hit = state.get("experience_hit") or {}
        feedback_score = experience_hit.get("feedback_score", 0) or 0

        regen_failed = state.get("regen_failed", False)
        intent_label = state.get("intent_label", INTENT_DOCS)
        is_ambiguous = False
        tool_attempted = state.get("tool_attempted", False)
        tool_has_results = state.get("tool_has_results", False)
        followup_context = state.get("followup_context")

        low_confidence = confidence < PRIMARY_MIN_SIM
        feedback_block = feedback_score < -2
        no_docs = state.get("no_docs", False)

        if tool_has_results and not chunks:
            confidence = max(confidence, 0.55)
            low_confidence = False

        needs_human = (
            feedback_block
            or regen_failed
            or (tool_attempted and (not tool_has_results or low_confidence))
            or no_docs
        )

        if followup_context and confidence < PRIMARY_MIN_SIM:
            confidence = 0.35
            low_confidence = False

        if followup_context and not feedback_block and not regen_failed and not no_docs:
            needs_human = False

        if intent_label == INTENT_LATEST_SOCIAL and tool_has_results:
            needs_human = False
            logger.info(
                "AnalyzerNode social intent satisfied with tool results; not escalating."
            )

        if no_docs and needs_human:
            logger.warning(
                "AnalyzerNode escalation | reason=no_docs_or_below_threshold | intent=%s",
                intent_label
            )

        source_type = "docs"
        if needs_human:
            source_type = "human"
        elif tool_has_results:
            source_type = "web"

        return {
            "confidence": confidence,
            "is_ambiguous": is_ambiguous,
            "low_confidence": low_confidence,
            "feedback_block": feedback_block,
            "needs_human": needs_human,
            "source_type": source_type,
            "trace": _append_trace(
                state,
                "AnalyzerNode",
                start,
                (
                    "no_docs_escalated"
                    if no_docs
                    else f"intent={intent_label};confidence={confidence:.2f}"
                )
            )
        }

    return node


def make_human_in_loop_node():
    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        return {
            "answer": (
                "I do not have enough verified information in this bot's "
                "documents or company social sources for a reliable answer. "
                "I have escalated this to support."
            ),
            "confidence": 0.0,
            "references": [],
            "chunks": [],
            "source_type": "human",
            "trace": _append_trace(state, "HumanInLoopNode", start, "handoff")
        }

    return node


def make_owner_resolution_node():
    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        override = state.get("owner_override")
        if isinstance(override, dict) and override.get("answer"):
            return {
                "answer": override.get("answer"),
                "source_type": "human",
                "confidence": float(override.get("confidence", 1.0)),
                "references": override.get("references", []),
                "trace": _append_trace(state, "OwnerResolutionNode", start, "override")
            }
        return {"trace": _append_trace(state, "OwnerResolutionNode", start, "none")}

    return node


def make_finalize_node(call_llm, call_social_llm=None):
    chain = RunnableSequence(
        RunnableLambda(
            lambda x: _build_prompt(
                x.get("context", ""),
                x.get("query", ""),
                x.get("conversation"),
                x.get("followup_context"),
                x.get("no_docs", False),
                x.get("is_ambiguous", False),
                x.get("tool_has_results", False),
                x.get("tool_context", ""),
                x.get("intent_label", INTENT_DOCS),
                x.get("intent_reason", "")
            )
        ),
        RunnableLambda(call_llm)
    )

    def _call_social(prompt: str) -> str:
        if callable(call_social_llm):
            try:
                response = call_social_llm(prompt)
            except Exception:
                response = None
            if isinstance(response, str) and response.strip():
                return response.strip()
        return call_llm(prompt)

    social_chain = RunnableSequence(
        RunnableLambda(
            lambda x: _build_social_prompt(
                x.get("question", ""),
                x.get("conversation"),
                x.get("tool_context", ""),
                x.get("docs_context", ""),
                x.get("no_docs", False),
                x.get("tool_has_results", False),
                x.get("intent_reason", "")
            )
        ),
        RunnableLambda(_call_social)
    )

    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        answer = state.get("answer")
        chunks = state.get("chunks") or []
        references = state.get("references") or []
        confidence = state.get("confidence", 0.0)
        source_type = state.get("source_type", "docs")
        followup_context = state.get("followup_context")
        no_docs = state.get("no_docs", False)
        tool_results = state.get("tool_results") or {}
        tool_context = _format_tool_results(tool_results)
        intent_label = state.get("intent_label", INTENT_DOCS)

        if not answer:
            context = state.get("context", "")
            if tool_context and intent_label != INTENT_LATEST_SOCIAL:
                if context:
                    context = f"{tool_context}\n\n{context}"
                else:
                    context = tool_context
            if followup_context and not context:
                context = f"{followup_context}"

            if intent_label == INTENT_LATEST_SOCIAL:
                answer = social_chain.invoke({
                    "question": state.get("query", ""),
                    "conversation": state.get("conversation"),
                    "tool_context": tool_context,
                    "docs_context": context,
                    "no_docs": no_docs,
                    "tool_has_results": state.get("tool_has_results", False),
                    "intent_reason": state.get("intent_reason", "")
                })
            else:
                answer = chain.invoke({
                    "context": context,
                    "query": state.get("query", ""),
                    "conversation": state.get("conversation"),
                    "followup_context": followup_context,
                    "no_docs": no_docs,
                    "is_ambiguous": state.get("is_ambiguous", False),
                    "tool_has_results": state.get("tool_has_results", False),
                    "tool_context": tool_context,
                    "intent_label": intent_label,
                    "intent_reason": state.get("intent_reason", "")
                })

        if not references and chunks:
            references = [
                {
                    "chunk_ref": c.get("chunk_ref"),
                    "cluster": c.get("cluster"),
                    "score": c.get("score", 0.0),
                    "source_type": c.get("source_type"),
                    "source_url": c.get("source_url"),
                    "pdf": c.get("pdf")
                }
                for c in chunks
            ]

        if not references and followup_context and not chunks:
            references = [
                {
                    "type": "memory",
                    "note": "previous assistant response"
                }
            ]

        output_chunks = [
            {
                "chunk_ref": c.get("chunk_ref"),
                "score": c.get("score", 0.0),
                "cluster": c.get("cluster"),
                "source_type": c.get("source_type"),
                "source_url": c.get("source_url"),
                "pdf": c.get("pdf")
            }
            for c in chunks
        ]

        return {
            "answer": answer,
            "chunks": output_chunks,
            "references": references,
            "confidence": confidence,
            "source_type": source_type,
            "no_docs": no_docs,
            "trace": _append_trace(state, "FinalizeResponse", start, "done")
        }

    return node
