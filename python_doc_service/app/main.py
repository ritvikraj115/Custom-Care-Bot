from fastapi import FastAPI, UploadFile, Form, File
from typing import List
import os
import json
import re
import shutil
import time
import numpy as np
import logging
from fastapi import HTTPException
import traceback

try:
    from dotenv import load_dotenv
    _ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(_ENV_PATH)
except Exception:
    pass

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.pipeline.run_pipeline import run_pipeline
from app.pipeline.loader import load_all_indexes
from app.pipeline.model_monitoring import get_model_dashboard
from app.pipeline.dvc_auto import run_cluster_training
from app.pipeline.airflow_trigger import trigger_hdbscan_cluster_dag
from app.autocomplete_training_pipeline import get_autocomplete_manager
# LangGraph orchestration
from app.graph.build_graph import build_answer_graph
from app.graph.tools import (
    refresh_social_index_for_bot,
    get_social_index_status
)
# ðŸ”¹ STEP 5: vector memory
from app.pipeline.vector_store import (
    get_negative_count_for_bot,
    store_experience,
    search_experience,
    update_experience_feedback
)
from pydantic import BaseModel
class AnswerRequest(BaseModel):
    query: str
    bot_id: str
    client_id: str
    top_k: int = 5
    retrieval_variant: str = "primary"
    exclude_chunk_refs: list[str] | None = None
    conversation: list[dict] | None = None
    doc_scope: str | None = None
    social_links: dict | None = None
    website_url: str | None = None


class SocialRefreshRequest(BaseModel):
    bot_id: str
    social_links: dict | None = None
    website_url: str | None = None
    query_hints: list[str] | None = None
    max_results_per_platform: int = 2

# ===== STEP 5 =====
class ExperienceIndexRequest(BaseModel):
    experience_id: str
    text: str
    bot_id: str
    client_id: str
    semantic_id: str | None = None
    feedback_score: float = 0
    negative_count: int = 0
    owner_answer: bool = False
    avg_chunk_similarity: float


class ExperienceSearchRequest(BaseModel):
    query: str
    bot_id: str

class ExperienceFeedbackUpdate(BaseModel):
    experience_id: str | None = None
    semantic_id: str | None = None
    bot_id: str | None = None
    delta: float
    feedback_score: float | None = None
    negative_count: int | None = None


class BotDeleteRequest(BaseModel):
    bot_id: str
    client_id: str

class AutocompleteRecordRequest(BaseModel):
    query: str
    bot_id: str
    client_id: str


class AutocompleteSuggestRequest(BaseModel):
    query: str
    bot_id: str
    client_id: str
    max_suggestions: int = 5
    max_future_words: int = 5


class AutocompleteTrainRequest(BaseModel):
    bot_id: str
    client_id: str
    wait: bool = False
    force: bool = False


class AutocompleteStatusRequest(BaseModel):
    bot_id: str
    client_id: str


class AutocompleteTopQuestionsRequest(BaseModel):
    bot_id: str
    client_id: str
    limit: int = 5


class HdbscanTrainRequest(BaseModel):
    bot_id: str
    client_id: str
    rebuild_mode: str = "full"
    pdf_manifest: str = "data/dvc/pdf_manifest.json"


# --------------------------------------------------
# App setup
# --------------------------------------------------

logging.getLogger().setLevel(logging.INFO)

app = FastAPI(title="Document Intelligence Service")

UPLOAD_DIR = "/tmp/pdfs"
SCRAPED_PDF_DIR = os.getenv("SCRAPED_PDF_DIR", UPLOAD_DIR).strip() or UPLOAD_DIR
os.makedirs(UPLOAD_DIR, exist_ok=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNTIME_DVC_ROOT = os.path.join(PROJECT_ROOT, "data", "dvc", "runtime")
os.makedirs(RUNTIME_DVC_ROOT, exist_ok=True)

# --------------------------------------------------
# Models
# --------------------------------------------------

embedder = SentenceTransformer("all-MiniLM-L6-v2")
autocomplete_manager = get_autocomplete_manager()
if hasattr(autocomplete_manager, "set_semantic_embedder"):
    autocomplete_manager.set_semantic_embedder(embedder)


@app.on_event("startup")
def startup_load_indexes():
    try:
        stats = load_all_indexes()
        logging.info(
            "Hierarchical indexes loaded | bots=%d | chunk_partitions=%d",
            int((stats or {}).get("bots_loaded", 0)),
            int((stats or {}).get("chunk_partitions_loaded", 0))
        )
    except Exception as err:
        logging.exception("Failed to load persisted indexes: %s", err)


def _extract_json_block(text: str) -> dict | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _safe_runtime_id(raw: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(raw or "unknown")).strip("_") or "unknown"


def _safe_runtime_filename(raw: str, fallback: str) -> str:
    base = str(raw or fallback).strip()
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", base).strip("._")
    cleaned = cleaned or fallback
    if not cleaned.lower().endswith(".pdf"):
        cleaned = f"{cleaned}.pdf"
    return cleaned


def _build_runtime_pdf_manifest(
    client_id: str,
    bot_id: str,
    pdf_paths: list[str],
    pdf_metadata: dict[str, dict] | None = None,
) -> str:
    safe_client = _safe_runtime_id(client_id)
    safe_bot = _safe_runtime_id(bot_id)
    stamp = str(int(time.time() * 1000))

    upload_rel_dir = os.path.join(
        "data",
        "dvc",
        "runtime",
        "uploads",
        f"client_{safe_client}",
        f"bot_{safe_bot}",
        stamp,
    )
    upload_abs_dir = os.path.join(PROJECT_ROOT, upload_rel_dir)
    os.makedirs(upload_abs_dir, exist_ok=True)

    entries: list[dict] = []
    metadata = pdf_metadata or {}
    for idx, src in enumerate(pdf_paths or []):
        src_path = str(src or "").strip()
        if not src_path:
            continue
        if not os.path.exists(src_path):
            continue

        original_name = os.path.basename(src_path)
        safe_name = _safe_runtime_filename(original_name, fallback=f"file_{idx}.pdf")
        dst_name = f"{idx:03d}_{safe_name}"
        dst_abs = os.path.join(upload_abs_dir, dst_name)
        shutil.copy2(src_path, dst_abs)

        rel_path = os.path.join(upload_rel_dir, dst_name).replace("\\", "/")
        row: dict[str, str] = {"path": rel_path}
        meta = metadata.get(original_name, {}) if isinstance(metadata, dict) else {}
        if isinstance(meta, dict):
            source_type = str(meta.get("source_type", "")).strip()
            source_url = str(meta.get("source_url", "")).strip()
            if source_type:
                row["source_type"] = source_type
            if source_url:
                row["source_url"] = source_url
        entries.append(row)

    if not entries:
        raise RuntimeError("Unable to build runtime PDF manifest: no resolved PDF files")

    manifest_rel = os.path.join(
        "data",
        "dvc",
        "runtime",
        "manifests",
        f"client_{safe_client}",
        f"bot_{safe_bot}",
        "latest_pdf_manifest.json",
    ).replace("\\", "/")
    manifest_abs = os.path.join(PROJECT_ROOT, manifest_rel)
    os.makedirs(os.path.dirname(manifest_abs), exist_ok=True)
    payload = {
        "generated_at": int(time.time()),
        "client_id": str(client_id),
        "bot_id": str(bot_id),
        "pdfs": entries,
    }
    with open(manifest_abs, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    return manifest_rel


def _build_seed_generation_prompt(doc_samples: list[str], website_samples: list[str], target: int = 100) -> str:
    docs_block = "\n".join(f"- {row}" for row in (doc_samples or [])[:18])
    website_block = "\n".join(f"- {row}" for row in (website_samples or [])[:12])
    target = max(20, int(target))
    return f"""
You create high-quality customer chatbot questions grounded in business documentation.

Generate exactly {target} unique, realistic user questions based only on this content.

Output format rules:
1) Return strict JSON only.
2) JSON shape must be: {{"questions": ["q1", "q2", "..."]}}
3) Do not return markdown, code fences, notes, or numbering.

Quality rules:
- Keep each question natural and conversational.
- 5 to 16 words each.
- Cover mixed intent types: setup, pricing, policy, troubleshooting, account actions, billing, support.
- Avoid duplicates and near duplicates.
- Avoid invented features not implied by content.

DOCUMENT SAMPLES:
{docs_block or "- (none)"}

WEBSITE SAMPLES:
{website_block or "- (none)"}
""".strip()


def _build_seed_augmentation_prompt(base_questions: list[str], target: int = 50) -> str:
    examples = "\n".join(f"- {q}" for q in (base_questions or [])[:60])
    target = max(20, int(target))
    return f"""
You are augmenting chatbot training data.

Given the base user questions, generate exactly {target} additional realistic variants.
These should be semantically close to base questions but with different wording.

Output format rules:
1) Return strict JSON only.
2) JSON shape must be: {{"questions": ["q1", "q2", "..."]}}
3) Do not return markdown, code fences, notes, numbering, or explanations.

Quality rules:
- Keep each question natural and conversational.
- 5 to 16 words each.
- Avoid exact duplicates of base questions.
- Keep intent grounded to the same domain.

BASE QUESTIONS:
{examples or "- (none)"}
""".strip()


def _augment_bootstrap_questions_with_llm(
    bot_id: str,
    base_questions: list[str],
    target: int = 50,
) -> list[str]:
    if not base_questions:
        return []

    prompt = _build_seed_augmentation_prompt(base_questions, target=target)
    llm_text = call_gemini_llm(prompt)
    parsed = _extract_json_block(llm_text or "")
    rows: list[str] = []
    if parsed and isinstance(parsed.get("questions"), list):
        rows = [str(q) for q in parsed.get("questions", [])]

    if not rows:
        logging.warning("Gemini augmentation returned empty set | bot_id=%s", bot_id)
        return []

    base_set = {
        re.sub(r"\s+", " ", str(q)).strip().lower()
        for q in base_questions
        if str(q).strip()
    }
    cleaned = [re.sub(r"\s+", " ", str(q)).strip().lower() for q in rows]
    cleaned = [q for q in cleaned if q and q not in base_set]
    cleaned = list(dict.fromkeys(cleaned))
    return cleaned[: max(20, int(target))]


def _generate_bootstrap_questions_from_content(
    bot_id: str,
    doc_samples: list[str],
    website_samples: list[str],
    target: int = 100,
) -> list[str]:
    prompt = _build_seed_generation_prompt(doc_samples, website_samples, target=target)
    llm_text = call_gemini_llm(prompt)
    parsed = _extract_json_block(llm_text or "")
    rows: list[str] = []
    if parsed and isinstance(parsed.get("questions"), list):
        rows = [str(q) for q in parsed.get("questions", [])]

    base_questions: list[str] = []
    if rows:
        cleaned = [re.sub(r"\s+", " ", str(q)).strip().lower() for q in rows]
        cleaned = [q for q in cleaned if q]
        cleaned = list(dict.fromkeys(cleaned))
        if len(cleaned) >= 30:
            base_questions = cleaned[: max(30, int(target))]

    if not base_questions:
        logging.warning(
            "Gemini starter-question generation fallback | bot_id=%s | generated=%d",
            bot_id,
            len(rows),
        )
        base_questions = autocomplete_manager.build_fallback_seed_questions(
            context_texts=(doc_samples or []) + (website_samples or []),
            target_count=target,
        )

    aug_target = max(20, int(max(30, int(target)) * 0.5))
    augmented_questions = _augment_bootstrap_questions_with_llm(
        bot_id=bot_id,
        base_questions=base_questions[: max(40, min(len(base_questions), 120))],
        target=aug_target
    )

    merged = list(dict.fromkeys(base_questions + augmented_questions))
    if len(merged) < int(target):
        fallback = autocomplete_manager.build_fallback_seed_questions(
            context_texts=(doc_samples or []) + (website_samples or []),
            target_count=max(10, int(target) - len(merged)),
        )
        merged = list(dict.fromkeys(merged + fallback))

    return merged[: max(30, int(target) + aug_target)]


def _semantic_top_questions(rows: list[dict], limit: int = 5) -> list[dict]:
    if not rows:
        return []
    limit = max(1, min(int(limit), 10))

    merged: dict[str, dict] = {}
    for row in rows:
        q = " ".join(str(row.get("question", "")).strip().lower().split())
        if not q:
            continue
        ts = float(row.get("ts", 0.0) or 0.0)
        current = merged.get(q)
        if not current:
            merged[q] = {"question": q, "count": 1, "latest_ts": ts}
        else:
            current["count"] += 1
            if ts > float(current.get("latest_ts", 0.0) or 0.0):
                current["latest_ts"] = ts

    items = list(merged.values())
    if not items:
        return []
    if len(items) == 1:
        one = items[0]
        return [{"text": one["question"], "count": int(one["count"])}]

    questions = [item["question"] for item in items]
    embeddings = embedder.encode(questions, normalize_embeddings=True)
    sim = cosine_similarity(embeddings)

    order = sorted(
        range(len(items)),
        key=lambda idx: (
            int(items[idx].get("count", 0)),
            float(items[idx].get("latest_ts", 0.0)),
        ),
        reverse=True,
    )

    threshold = 0.74
    assigned = set()
    now_ts = float(time.time())
    clusters: list[dict] = []
    for root_idx in order:
        if root_idx in assigned:
            continue
        members = [root_idx]
        assigned.add(root_idx)
        for candidate_idx in order:
            if candidate_idx in assigned:
                continue
            if float(sim[root_idx, candidate_idx]) >= threshold:
                members.append(candidate_idx)
                assigned.add(candidate_idx)

        member_items = [items[idx] for idx in members]
        representative = sorted(
            member_items,
            key=lambda item: (int(item["count"]), float(item["latest_ts"])),
            reverse=True,
        )[0]

        total_count = int(sum(int(it["count"]) for it in member_items))
        latest_ts = max(float(it["latest_ts"]) for it in member_items)
        age_days = max(0.0, (now_ts - latest_ts) / 86400.0)
        recency_bonus = float(np.exp(-age_days / 21.0))
        cluster_score = float(total_count) + (0.85 * recency_bonus)

        clusters.append(
            {
                "text": representative["question"],
                "count": total_count,
                "cluster_size": len(member_items),
                "score": cluster_score,
            }
        )

    clusters.sort(key=lambda item: item["score"], reverse=True)
    out = []
    for cluster in clusters[:limit]:
        out.append(
            {
                "text": str(cluster["text"]),
                "count": int(cluster["count"]),
            }
        )
    return out

# --------------------------------------------------
# PDF INGESTION ENDPOINT
# --------------------------------------------------

@app.post("/process")
async def process_pdfs(
    files: List[UploadFile] = File(default=[]),
    bot_id: str = Form(...),
    client_id: str = Form(...),
    website_url: str | None = Form(None),
    rebuild_mode: str = Form("full")
):
    pdf_paths = []
    pdf_metadata: dict[str, dict] = {}

    for file in files or []:
        path = os.path.join(UPLOAD_DIR, file.filename)
        with open(path, "wb") as f:
            f.write(await file.read())
        pdf_paths.append(path)
        pdf_metadata[os.path.basename(path)] = {
            "source_type": "upload"
        }

    if website_url:
        try:
            from app.pipeline.website import make_website_pdf
            logging.info("Website ingestion requested | url=%s", website_url)
            website_result = make_website_pdf(
                website_url,
                SCRAPED_PDF_DIR,
                bot_id=bot_id
            )
            if website_result:
                website_path, _ = website_result
                logging.info("Website PDF created | path=%s", website_path)
                pdf_paths.append(website_path)
                pdf_metadata[os.path.basename(website_path)] = {
                    "source_type": "website",
                    "source_url": website_url
                }
            else:
                logging.warning("Website ingestion produced no PDF | url=%s", website_url)
        except Exception as err:
            logging.exception("Website crawl failed: %s", err)

    if not pdf_paths:
        raise HTTPException(
            status_code=400,
            detail="No PDFs or website content provided"
        )

    hdbscan_dvc = {
        "triggered": False,
        "ok": False,
        "detail": "",
        "pdf_manifest": "",
    }
    pipeline_summary = {}
    runtime_manifest = ""
    try:
        runtime_manifest = _build_runtime_pdf_manifest(
            client_id=client_id,
            bot_id=bot_id,
            pdf_paths=pdf_paths,
            pdf_metadata=pdf_metadata,
        )
        dvc_result = run_cluster_training(
            client_id=client_id,
            bot_id=bot_id,
            rebuild_mode=rebuild_mode,
            pdf_manifest=runtime_manifest,
        )
        if bool(dvc_result.get("ok", False)):
            pipeline_summary = dvc_result.get("summary", {}) or {}
            hdbscan_dvc = {
                "triggered": True,
                "ok": True,
                "detail": "cluster_train_completed",
                "pdf_manifest": str(runtime_manifest),
            }
        else:
            hdbscan_dvc = {
                "triggered": True,
                "ok": False,
                "detail": str(dvc_result.get("error", "cluster_train_failed")),
                "pdf_manifest": str(runtime_manifest),
            }
    except Exception as err:
        hdbscan_dvc = {
            "triggered": True,
            "ok": False,
            "detail": str(err),
            "pdf_manifest": str(runtime_manifest),
        }

    if not pipeline_summary:
        logging.warning(
            "DVC cluster path unavailable, using direct run_pipeline fallback | client_id=%s | bot_id=%s | detail=%s",
            client_id,
            bot_id,
            hdbscan_dvc.get("detail"),
        )
        pipeline_summary = run_pipeline(
            pdf_paths,
            bot_id,
            client_id,
            pdf_metadata=pdf_metadata,
            rebuild_mode=rebuild_mode
        )

    hdbscan_airflow = {
        "triggered": False,
        "detail": "",
        "skip_train": True,
    }
    try:
        ok, detail = trigger_hdbscan_cluster_dag(
            client_id=client_id,
            bot_id=bot_id,
            rebuild_mode=rebuild_mode,
            pdf_manifest=(runtime_manifest or "data/dvc/pdf_manifest.json"),
            reason="frontend_process_upload",
            skip_train=True,
        )
        hdbscan_airflow = {
            "triggered": bool(ok),
            "detail": str(detail),
            "skip_train": True,
        }
    except Exception as err:
        hdbscan_airflow = {
            "triggered": False,
            "detail": str(err),
            "skip_train": True,
        }

    bootstrap_info = {
        "initialized": False,
        "triggered_training": False,
        "seed_questions_used": 0
    }
    try:
        status = autocomplete_manager.get_status(client_id, bot_id)
        already_bootstrapped = bool(status.get("bootstrap_completed", False))
        if not already_bootstrapped:
            doc_samples = list(pipeline_summary.get("doc_samples", []) or [])
            website_samples = list(pipeline_summary.get("website_samples", []) or [])
            starter_questions = _generate_bootstrap_questions_from_content(
                bot_id=bot_id,
                doc_samples=doc_samples,
                website_samples=website_samples,
                target=100
            )
            bootstrap_info = autocomplete_manager.bootstrap_seed_questions(
                client_id=client_id,
                bot_id=bot_id,
                seed_questions=starter_questions,
                wait=False
            )
        else:
            bootstrap_info["reason"] = "already_bootstrapped"
    except Exception as err:
        logging.exception("Autocomplete bootstrap failed | bot_id=%s | err=%s", bot_id, err)
        bootstrap_info["error"] = str(err)

    return {
        "status": "processing_completed",
        "pdfs_processed": len(pdf_paths),
        "autocomplete_bootstrap": bootstrap_info,
        "hdbscan_dvc": hdbscan_dvc,
        "hdbscan_airflow": hdbscan_airflow,
    }

# --------------------------------------------------
# AUTOCOMPLETE (BOT-SPECIFIC)
# --------------------------------------------------


@app.post("/hdbscan/train")
def hdbscan_train(req: HdbscanTrainRequest):
    try:
        result = run_cluster_training(
            client_id=req.client_id,
            bot_id=req.bot_id,
            rebuild_mode=req.rebuild_mode,
            pdf_manifest=req.pdf_manifest,
        )
        if not bool(result.get("ok", False)):
            raise HTTPException(
                status_code=500,
                detail=str(result.get("error", "cluster_train_failed")),
            )
        return {
            "triggered": True,
            "ok": True,
            "summary": result.get("summary", {}),
            "summary_file": result.get("summary_file", ""),
        }
    except HTTPException:
        raise
    except Exception as err:
        logging.exception("HDBSCAN train trigger failed: %s", err)
        raise HTTPException(
            status_code=500,
            detail="Failed to trigger HDBSCAN training",
        )


@app.post("/autocomplete/record-question")
def record_autocomplete_question(req: AutocompleteRecordRequest):
    try:
        result = autocomplete_manager.record_question(
            client_id=req.client_id,
            bot_id=req.bot_id,
            question=req.query
        )
        return result
    except Exception as err:
        logging.exception("Autocomplete record failed: %s", err)
        raise HTTPException(
            status_code=500,
            detail="Failed to record autocomplete question"
        )


@app.post("/autocomplete/suggest")
def autocomplete_suggest(req: AutocompleteSuggestRequest):
    try:
        max_suggestions = max(1, min(int(req.max_suggestions), 3))
        max_future_words = max(1, min(int(req.max_future_words), 3))

        result = autocomplete_manager.suggest(
            client_id=req.client_id,
            bot_id=req.bot_id,
            text=req.query,
            max_suggestions=max_suggestions,
            max_future_words=max_future_words
        )
        return result
    except Exception as err:
        logging.exception("Autocomplete suggest failed: %s", err)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate autocomplete suggestions"
        )


@app.post("/autocomplete/train")
def autocomplete_train(req: AutocompleteTrainRequest):
    try:
        triggered = autocomplete_manager.trigger_training(
            client_id=req.client_id,
            bot_id=req.bot_id,
            wait=bool(req.wait),
            force=bool(req.force),
        )
        status = autocomplete_manager.get_status(req.client_id, req.bot_id)
        return {
            "triggered": bool(triggered),
            "status": status
        }
    except Exception as err:
        logging.exception("Autocomplete train trigger failed: %s", err)
        raise HTTPException(
            status_code=500,
            detail="Failed to trigger autocomplete training"
        )


@app.post("/autocomplete/status")
def autocomplete_status(req: AutocompleteStatusRequest):
    try:
        return autocomplete_manager.get_status(req.client_id, req.bot_id)
    except Exception as err:
        logging.exception("Autocomplete status failed: %s", err)
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch autocomplete status"
        )


@app.post("/autocomplete/top-questions")
def autocomplete_top_questions(req: AutocompleteTopQuestionsRequest):
    try:
        limit = max(1, min(int(req.limit), 5))
        rows = autocomplete_manager.get_question_rows(
            client_id=req.client_id,
            bot_id=req.bot_id,
            include_user=True,
            include_seed=False,
            limit=2000
        )
        use_source = "user"
        if not rows:
            rows = autocomplete_manager.get_question_rows(
                client_id=req.client_id,
                bot_id=req.bot_id,
                include_user=False,
                include_seed=True,
                limit=2000
            )
            use_source = "seed"

        top_items = _semantic_top_questions(rows, limit=limit)
        return {
            "questions": top_items,
            "source": use_source,
            "status": autocomplete_manager.get_status(req.client_id, req.bot_id)
        }
    except Exception as err:
        logging.exception("Autocomplete top questions failed: %s", err)
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch top autocomplete questions"
        )

# --------------------------------------------------
# MODEL MONITORING DASHBOARD
# --------------------------------------------------

@app.get("/monitoring/model-dashboard")
def model_dashboard():
    try:
        return get_model_dashboard()
    except Exception as err:
        logging.exception("Model dashboard fetch failed: %s", err)
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch model dashboard"
        )

# --------------------------------------------------
@app.post("/social/refresh")
def refresh_social_index(req: SocialRefreshRequest):
    try:
        summary = refresh_social_index_for_bot(
            bot_id=req.bot_id,
            social_links=req.social_links or {},
            website_url=req.website_url,
            query_hints=req.query_hints or [],
            max_results_per_platform=max(1, min(int(req.max_results_per_platform), 4))
        )
        return {
            "status": "ok",
            "summary": summary
        }
    except Exception as err:
        logging.exception("Social refresh failed: %s", err)
        raise HTTPException(
            status_code=500,
            detail="Failed to refresh social index"
        )


@app.get("/social/status/{bot_id}")
def social_status(bot_id: str):
    try:
        return get_social_index_status(bot_id)
    except Exception as err:
        logging.exception("Social status failed: %s", err)
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch social index status"
        )

# GEMINI SDK - google-genai
# --------------------------------------------------

def call_gemini_llm(prompt: str) -> str | None:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        logging.info("Gemini API key not set (GEMINI_API_KEY).")
        return None

    try:
        from google import genai
        from google.genai import types
    except Exception as err:
        logging.exception("google-genai not available: %s", err)
        return None

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.2") or 0.2)

    config = None
    try:
        config = types.GenerateContentConfig(temperature=temperature)
    except Exception:
        config = None

    try:
        client = genai.Client()
        if config is None:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
        else:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config
            )
    except Exception as err:
        logging.exception("Gemini LLM call failed: %s", err)
        return None

    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


# --------------------------------------------------
# OPENAI SDK (disabled - kept for reference)
# --------------------------------------------------

# def call_openai_llm(prompt: str) -> str | None:
#     api_key = os.getenv("OPENAI_API_KEY", "").strip()
#     if not api_key:
#         logging.info("OpenAI API key not set (OPENAI_API_KEY).")
#         return None
#
#     try:
#         from openai import OpenAI
#     except Exception as err:
#         logging.exception("openai package not available: %s", err)
#         return None
#
#     model = os.getenv("OPENAI_MODEL", "gpt-4.1").strip() or "gpt-4.1"
#     temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2") or 0.2)
#
#     try:
#         client = OpenAI(api_key=api_key)
#         response = client.chat.completions.create(
#             model=model,
#             temperature=temperature,
#             messages=[
#                 {"role": "user", "content": prompt}
#             ]
#         )
#     except Exception as err:
#         logging.exception("OpenAI LLM call failed: %s", err)
#         return None
#
#     try:
#         text = response.choices[0].message.content
#     except Exception:
#         text = None
#
#     if isinstance(text, str) and text.strip():
#         return text.strip()
#     return None


# --------------------------------------------------
# LLM CALL (Gemini + fallback)
# --------------------------------------------------

def call_llm(prompt: str) -> str:
    """
    Temporary extractive answer for testing
    """
    gemini_answer = call_gemini_llm(prompt)
    if gemini_answer:
        return gemini_answer

    # Return first 2-3 sentences from docs if available
    if "DOCS:" in prompt:
        context_block = prompt.split("DOCS:")[-1]
    elif "Context:" in prompt:
        context_block = prompt.split("Context:")[-1]
    else:
        context_block = ""

    if "QUESTION:" in context_block:
        context_block = context_block.split("QUESTION:")[0]

    lines = context_block.split("\n")
    answer_lines = [
        l.strip()
        for l in lines
        if l.strip()
    ]
    if answer_lines:
        return " ".join(answer_lines[:3])
    return (
        "I do not have enough verified information in this bot's documents "
        "or company social sources for a reliable answer. "
        "I have escalated this to support."
    )


# --------------------------------------------------
# LangGraph (Answer Orchestration)
# --------------------------------------------------

ANSWER_GRAPH = build_answer_graph(
    embedder,
    call_llm,
    call_gemini_llm,
    call_gemini_llm
)


# --------------------------------------------------
# ANSWER ENDPOINT (RAG)
# --------------------------------------------------


@app.post("/answer")
async def answer_query(req: AnswerRequest):
    try:
        retrieval_variant = req.retrieval_variant or "primary"
        logging.info(
            "Answer request | client_id=%s | bot_id=%s | top_k=%d | retrieval_variant=%s | conversation_turns=%d",
            req.client_id,
            req.bot_id,
            req.top_k,
            retrieval_variant,
            len(req.conversation or [])
        )

        state = {
            "query": req.query,
            "bot_id": req.bot_id,
            "client_id": req.client_id,
            "top_k": req.top_k,
            "retrieval_variant": retrieval_variant,
            "exclude_chunk_refs": req.exclude_chunk_refs or [],
            "conversation": req.conversation,
            "doc_scope": req.doc_scope,
            "social_links": req.social_links,
            "website_url": req.website_url
        }

        result = ANSWER_GRAPH.invoke(state)

        trace = result.get("trace", []) or []
        trace_summary = " > ".join(
            f"{item.get('step')}({item.get('ms')}ms)"
            for item in trace
            if isinstance(item, dict)
        )
        logging.info(
            "Answer result | intent=%s | intent_confidence=%.2f | source_type=%s | confidence=%.2f | no_docs=%s",
            result.get("intent_label", "unknown"),
            float(result.get("intent_confidence", 0.0) or 0.0),
            result.get("source_type", "docs"),
            float(result.get("confidence", 0.0) or 0.0),
            int(bool(result.get("no_docs", False)))
        )
        if trace_summary:
            logging.info("Answer trace | %s", trace_summary)

        return {
            "answer": result.get("answer", "No relevant information found."),
            "chunks": result.get("chunks", []),
            "confidence": result.get("confidence", 0.0),
            "references": result.get("references", []),
            "source_type": result.get("source_type", "docs"),
            "trace": result.get("trace", []),
            "no_docs": result.get("no_docs", False)
        }

    except HTTPException as he:
        raise he

    except Exception as e:
        logging.exception("Unhandled error in /answer")
        return {
            "answer": "Internal RAG error",
            "error": str(e)
        }

# ===== STEP 5: EXPERIENCE VECTOR INDEX =====
# --------------------------------------------------

@app.post("/experience/index")
def index_experience(req: ExperienceIndexRequest):
    """
    Index a past question as experience memory
    """
    try:
        embedding = embedder.encode(
            req.text,
            normalize_embeddings=True
        )

        store_experience(
            embedding,
            {
                "experience_id": req.experience_id,
                "bot_id": req.bot_id,
                "client_id": req.client_id,
                "semantic_id": req.semantic_id,
                "feedback_score": req.feedback_score,
                "negative_count": req.negative_count,
                "owner_answer": req.owner_answer,
                "avg_chunk_similarity": req.avg_chunk_similarity
            }
        )

        return {"status": "indexed"}

    except Exception:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Failed to index experience"
        )


@app.post("/experience/search")
def search_experience_api(req: ExperienceSearchRequest):
    """
    Search similar past experiences (semantic)
    """
    try:
        embedding = embedder.encode(
            req.query,
            normalize_embeddings=True
        )

        result = search_experience(
            embedding,
            req.bot_id
        )
        print("EXPERIENCE SEARCH RESULT:", result)


        return result

    except Exception:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Failed to search experience"
        )

@app.post("/experience/update-feedback")
def update_feedback(req: ExperienceFeedbackUpdate):
    updated = update_experience_feedback(
        experience_id=req.experience_id,
        delta=req.delta,
        semantic_id=req.semantic_id,
        bot_id=req.bot_id,
        feedback_score=req.feedback_score,
        negative_count=req.negative_count
    )

    if not updated:
        raise HTTPException(
            status_code=404,
            detail="Experience vector not found"
        )

    return {"status": "updated"}

@app.post("/experience/negative-count")
def negative_count(req: dict):
    bot_id = req.get("bot_id")
    return {
        "negative_count": get_negative_count_for_bot(bot_id)
    }


# --------------------------------------------------
# BOT DELETE (VECTOR CLEANUP)
# --------------------------------------------------

@app.post("/bot/delete")
def delete_bot_vectors(req: BotDeleteRequest):
    try:
        from app.pipeline.hierarchical_index import delete_bot_indexes
        from app.pipeline.vector_store import (
            delete_experiences_for_bot,
            delete_chunks_for_bot,
            delete_website_chunks_for_bot
        )
        from app.pipeline.elasticsearch_hybrid import (
            delete_bot_chunks as es_delete_bot_chunks,
            delete_bot_questions as es_delete_bot_questions,
        )
        from app.pipeline.storage import delete_bot_storage

        delete_bot_indexes(req.client_id, req.bot_id)
        delete_experiences_for_bot(req.bot_id)
        delete_chunks_for_bot(req.bot_id)
        delete_website_chunks_for_bot(req.bot_id)
        es_delete_bot_chunks(req.client_id, req.bot_id)
        es_delete_bot_questions(req.client_id, req.bot_id)
        delete_bot_storage(req.client_id, req.bot_id)

        return {"status": "vector_cleanup_complete"}
    except Exception as err:
        logging.exception("Bot vector cleanup failed: %s", err)
        raise HTTPException(
            status_code=500,
            detail="Failed to delete bot vectors"
        )



