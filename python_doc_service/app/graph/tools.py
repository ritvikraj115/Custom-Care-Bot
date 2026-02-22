from typing import Dict, Any, List
import html
import json
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime
import asyncio
import concurrent.futures
import time
import threading

try:
    from playwright.sync_api import sync_playwright
except Exception:
    sync_playwright = None

try:
    import trafilatura
except Exception:
    trafilatura = None

import fitz

from langchain_core.runnables import RunnableLambda, RunnableParallel

from app.pipeline.logger import get_logger
from app.pipeline.storage import BASE_DIR, ensure_dir

log = get_logger("tools")

_PLAYWRIGHT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2)

DEFAULT_LOG_PREVIEW = int(
    os.getenv("SOCIAL_LOG_PREVIEW_CHARS", "400")
)
SOCIAL_PDF_DIR = os.getenv("SCRAPED_PDF_DIR", "/tmp/pdfs")
SOCIAL_TEXT_LIMIT = int(os.getenv("SOCIAL_TEXT_LIMIT", "50000"))
SOCIAL_RENDER_TIMEOUT_MS = int(os.getenv("SOCIAL_RENDER_TIMEOUT_MS", "35000"))
SOCIAL_SCROLL_PASSES = int(os.getenv("SOCIAL_SCROLL_PASSES", "1"))
SOCIAL_USE_PLAYWRIGHT = os.getenv("SOCIAL_USE_PLAYWRIGHT", "true").strip().lower() in {
    "1", "true", "yes", "y"
}
SOCIAL_RENDER_WAIT_MS = int(os.getenv("SOCIAL_RENDER_WAIT_MS", "3000"))
SOCIAL_RENDER_EXTRA_WAIT_MS = int(os.getenv("SOCIAL_RENDER_EXTRA_WAIT_MS", "5000"))
SOCIAL_INTENT_DOUBTFUL_THRESHOLD = float(
    os.getenv("SOCIAL_INTENT_DOUBTFUL_THRESHOLD", "0.55") or 0.55
)
SOCIAL_INDEX_ROOT = os.path.join(BASE_DIR, "social")
SOCIAL_INDEX_TTL_SEC = int(os.getenv("SOCIAL_INDEX_TTL_SEC", "21600"))
SOCIAL_REFRESH_TIMEOUT_SEC = int(os.getenv("SOCIAL_REFRESH_TIMEOUT_SEC", "15"))
SOCIAL_TOOL_LIVE_FALLBACK = os.getenv("SOCIAL_TOOL_LIVE_FALLBACK", "false").strip().lower() in {
    "1", "true", "yes", "y"
}

_SOCIAL_INDEX: Dict[str, Dict[str, Any]] = {}
_SOCIAL_LOCK = threading.Lock()
_SOCIAL_INDEX_LOADED = False


def _preview_text(text: str, limit: int | None = None) -> str:
    if not text:
        return ""
    limit = limit or DEFAULT_LOG_PREVIEW
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "..."


def _write_text_pdf(text: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc = fitz.open()
    page_width = 612
    page_height = 792
    margin = 36
    font_size = 10
    max_chars = 3200

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        start = end

    for chunk in chunks:
        page = doc.new_page(width=page_width, height=page_height)
        rect = fitz.Rect(margin, margin, page_width - margin, page_height - margin)
        page.insert_textbox(rect, chunk, fontsize=font_size)

    doc.save(output_path)
    doc.close()


def _fetch_rendered_html(
    url: str,
    timeout_ms: int,
    scroll_passes: int
) -> tuple[str | None, str | None]:
    if sync_playwright is None:
        return None, None

    def _render_sync() -> tuple[str | None, str | None]:
        browser = None
        context = None
        html_text = None
        visible_text = None
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=["--disable-blink-features=AutomationControlled"]
                )
                context = browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                    viewport={"width": 1280, "height": 720}
                )
                page = context.new_page()
                page.add_init_script(
                    "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
                )
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                page.wait_for_timeout(SOCIAL_RENDER_WAIT_MS)
                try:
                    page.wait_for_load_state("networkidle", timeout=5000)
                except Exception:
                    pass
                for _ in range(max(0, scroll_passes)):
                    page.mouse.wheel(0, 2000)
                    page.wait_for_timeout(800)
                html_text = page.content()
                try:
                    visible_text = page.inner_text("body")
                except Exception:
                    try:
                        visible_text = page.evaluate("document.body.innerText")
                    except Exception:
                        visible_text = None

                if _is_js_placeholder(visible_text):
                    page.wait_for_timeout(SOCIAL_RENDER_EXTRA_WAIT_MS)
                    try:
                        visible_text = page.inner_text("body")
                    except Exception:
                        try:
                            visible_text = page.evaluate("document.body.innerText")
                        except Exception:
                            pass
                    html_text = page.content()
        except Exception as err:
            log.warning("Social playwright fetch failed | url=%s | err=%s", url, err)
            return None, None
        finally:
            try:
                if context:
                    context.close()
            except Exception:
                pass
            try:
                if browser:
                    browser.close()
            except Exception:
                pass

        return html_text, visible_text

    try:
        asyncio.get_running_loop()
        log.info("Social render via thread | url=%s", url)
        future = _PLAYWRIGHT_EXECUTOR.submit(_render_sync)
        return future.result()
    except RuntimeError:
        return _render_sync()


def _extract_all_text(html_text: str, visible_text: str | None = None) -> str:
    text = ""
    if trafilatura is not None:
        try:
            text = trafilatura.html2txt(html_text) or ""
        except Exception:
            text = ""

    if not text:
        text = _strip_tags(html_text)

    visible = (visible_text or "").strip()
    if visible and len(visible) > len(text):
        text = visible

    return re.sub(r"\s+", " ", text).strip()


def _is_js_placeholder(text: str) -> bool:
    snippet = (text or "").lower()
    if not snippet:
        return True
    patterns = [
        "enable javascript",
        "please enable javascript",
        "you need to enable javascript",
        "javascript is required"
    ]
    return any(p in snippet for p in patterns)


def _rewrite_social_url(url: str) -> str:
    if not url:
        return url
    lowered = url.lower()
    if "facebook.com" in lowered and "m.facebook.com" not in lowered:
        return re.sub(r"^https?://(www\.)?facebook\.com", "https://m.facebook.com", url)
    if "instagram.com" in lowered and "www.instagram.com" not in lowered:
        return re.sub(r"^https?://(m\.)?instagram\.com", "https://www.instagram.com", url)
    return url


def _save_social_results_pdf(results: list[dict], query: str, bot_id: str | None) -> str | None:
    if not results:
        return None

    safe_bot = re.sub(r"[^a-zA-Z0-9_-]", "_", bot_id or "bot")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(SOCIAL_PDF_DIR, "social")
    filename = f"social_{safe_bot}_{timestamp}.pdf"
    output_path = os.path.join(output_dir, filename)

    lines = [
        "Social Search Results",
        f"Query: {query}",
        f"Bot ID: {bot_id or ''}",
        f"Generated (UTC): {timestamp}",
        ""
    ]

    for idx, item in enumerate(results, start=1):
        lines.append(f"{idx}. Platform: {item.get('platform') or ''}")
        lines.append(f"Title: {item.get('title') or ''}")
        lines.append(f"URL: {item.get('url') or ''}")
        snippet = item.get("snippet") or item.get("summary") or ""
        if snippet:
            lines.append(f"Snippet: {snippet}")
        content = item.get("content") or item.get("text") or ""
        if content:
            lines.append("Content:")
            lines.append(content)
        lines.append("")

    _write_text_pdf("\n".join(lines), output_path)
    return output_path


def _social_index_file(bot_id: str) -> str:
    safe_bot = re.sub(r"[^a-zA-Z0-9_-]", "_", str(bot_id or "bot")).strip("_") or "bot"
    return os.path.join(SOCIAL_INDEX_ROOT, f"bot_{safe_bot}.json")


def _load_social_index_once() -> None:
    global _SOCIAL_INDEX_LOADED
    with _SOCIAL_LOCK:
        if _SOCIAL_INDEX_LOADED:
            return
        ensure_dir(SOCIAL_INDEX_ROOT)
        for file_name in os.listdir(SOCIAL_INDEX_ROOT):
            if not file_name.endswith(".json"):
                continue
            full = os.path.join(SOCIAL_INDEX_ROOT, file_name)
            try:
                with open(full, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                continue
            bot_id = str(payload.get("bot_id") or "")
            if not bot_id:
                continue
            _SOCIAL_INDEX[bot_id] = payload
        _SOCIAL_INDEX_LOADED = True


def _persist_social_index(bot_id: str, payload: dict) -> None:
    ensure_dir(SOCIAL_INDEX_ROOT)
    with open(_social_index_file(bot_id), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True)


def _query_hints(base_query_hints: List[str] | None, platform: str) -> List[str]:
    hints = [h for h in (base_query_hints or []) if str(h).strip()]
    defaults = [
        f"{platform} official updates",
        f"{platform} latest posts",
        f"{platform} recent news",
    ]
    for hint in hints[:4]:
        defaults.append(f"{hint} {platform} updates")
    return list(dict.fromkeys(defaults))


def refresh_social_index_for_bot(
    bot_id: str,
    social_links: Dict[str, Any] | None = None,
    website_url: str | None = None,
    query_hints: List[str] | None = None,
    max_results_per_platform: int = 2,
) -> Dict[str, Any]:
    _load_social_index_once()

    bot_id = str(bot_id or "").strip()
    if not bot_id:
        return {
            "bot_id": "",
            "updated_at": 0,
            "results": 0,
            "reason": "missing_bot_id",
        }

    social_links = social_links or {}
    if not isinstance(social_links, dict):
        social_links = {}

    max_results_per_platform = max(1, min(int(max_results_per_platform or 2), 4))
    timeout = max(4, int(SOCIAL_REFRESH_TIMEOUT_SEC))

    results: list[dict] = []
    seen_urls: set[str] = set()
    platforms = list(SOCIAL_DOMAINS.keys())

    for platform in platforms:
        domain = SOCIAL_DOMAINS.get(platform, "")
        if not domain:
            continue

        raw_link = (
            social_links.get(platform)
            or social_links.get(f"{platform}_url")
            or social_links.get(f"{platform}Url")
        )
        if raw_link:
            url = _ensure_scheme(str(raw_link))
            if url and url not in seen_urls and domain in url.lower():
                seen_urls.add(url)
                page_summary = _fetch_page_summary(url, timeout=timeout)
                if page_summary:
                    text = (page_summary.get("text") or "").strip()
                    snippet = (
                        page_summary.get("description")
                        or text[:600]
                    )
                    results.append(
                        {
                            "platform": platform,
                            "title": page_summary.get("title") or url,
                            "url": url,
                            "snippet": snippet,
                            "content": text,
                            "source": "preindexed_link",
                        }
                    )

        platform_hits = sum(1 for row in results if row.get("platform") == platform)
        for search_query in _query_hints(query_hints, platform):
            if platform_hits >= max_results_per_platform:
                break
            candidates = _fetch_yahoo_results(search_query, timeout=timeout)
            for item in candidates:
                if platform_hits >= max_results_per_platform:
                    break
                url = (item.get("url") or "").strip()
                if not url or url in seen_urls:
                    continue
                if domain not in url.lower():
                    continue
                seen_urls.add(url)
                page_summary = _fetch_page_summary(url, timeout=timeout)
                if not page_summary:
                    continue
                text = (page_summary.get("text") or "").strip()
                snippet = (
                    page_summary.get("description")
                    or text[:600]
                    or item.get("title")
                    or ""
                )
                results.append(
                    {
                        "platform": platform,
                        "title": page_summary.get("title") or item.get("title") or url,
                        "url": url,
                        "snippet": snippet,
                        "content": text,
                        "source": "preindexed_search",
                    }
                )
                platform_hits += 1

    payload = {
        "bot_id": bot_id,
        "website_url": (website_url or "").strip(),
        "social_links": social_links,
        "updated_at": int(time.time()),
        "results": results,
    }

    with _SOCIAL_LOCK:
        _SOCIAL_INDEX[bot_id] = payload
    _persist_social_index(bot_id, payload)

    return {
        "bot_id": bot_id,
        "updated_at": payload["updated_at"],
        "results": len(results),
    }


def get_social_index_status(bot_id: str) -> Dict[str, Any]:
    _load_social_index_once()
    bot_id = str(bot_id or "").strip()
    payload = _SOCIAL_INDEX.get(bot_id) if bot_id else None
    if not payload:
        return {
            "bot_id": bot_id,
            "available": False,
            "results": 0,
            "updated_at": 0,
        }
    return {
        "bot_id": bot_id,
        "available": True,
        "results": len(payload.get("results", [])),
        "updated_at": int(payload.get("updated_at", 0) or 0),
        "age_seconds": max(0, int(time.time() - float(payload.get("updated_at", 0) or 0))),
    }


def _cached_social_results(bot_id: str, query: str, limit: int = 6) -> list[dict]:
    _load_social_index_once()
    payload = _SOCIAL_INDEX.get(str(bot_id or "").strip())
    if not payload:
        return []

    updated_at = float(payload.get("updated_at", 0) or 0)
    if updated_at <= 0:
        return []
    if time.time() - updated_at > SOCIAL_INDEX_TTL_SEC:
        return []

    rows = payload.get("results", [])
    if not isinstance(rows, list) or not rows:
        return []

    terms = [
        token
        for token in re.findall(r"[a-z0-9]+", (query or "").lower())
        if len(token) >= 3
    ]

    scored: list[tuple[int, dict]] = []
    for row in rows:
        title = str(row.get("title") or "").lower()
        snippet = str(row.get("snippet") or "").lower()
        content = str(row.get("content") or "").lower()
        haystack = " ".join([title, snippet, content])
        score = 0
        for token in terms:
            if token in title:
                score += 3
            elif token in snippet:
                score += 2
            elif token in content:
                score += 1
        if not terms:
            score = 1
        if score <= 0:
            continue
        scored.append((score, row))

    if not scored and rows:
        scored = [(1, row) for row in rows]

    scored.sort(key=lambda item: item[0], reverse=True)
    out = []
    for _, row in scored[: max(1, int(limit))]:
        out.append(
            {
                "platform": row.get("platform"),
                "title": row.get("title"),
                "url": row.get("url"),
                "snippet": row.get("snippet"),
                "content": row.get("content"),
                "source": row.get("source") or "preindexed",
            }
        )
    return out

SOCIAL_DOMAINS: Dict[str, str] = {
    "facebook": "facebook.com",
    "instagram": "instagram.com"
}


def _empty_tool_result(name: str) -> Dict[str, Any]:
    return {
        "tool": name,
        "results": [],
        "references": []
    }


def _strip_tags(text: str) -> str:
    cleaned = re.sub(
        r"(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>",
        " ",
        text or ""
    )
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _has_any(text: str, patterns: List[str]) -> bool:
    return any(p in text for p in patterns)


def _has_word(text: str, word: str) -> bool:
    if not text or not word:
        return False
    return re.search(rf"\\b{re.escape(word)}\\b", text) is not None


def _select_social_platforms(query: str, state: Dict[str, Any]) -> List[str]:
    q = (query or "").strip().lower()
    if q:
        if _has_any(q, ["instagram", "insta"]) or _has_word(q, "ig"):
            return ["instagram"]
        if _has_any(q, ["facebook", "face book"]) or _has_word(q, "fb"):
            return ["facebook"]

    confidence = state.get("intent_confidence")
    try:
        confidence_val = float(confidence)
    except Exception:
        confidence_val = None
    doubtful = (
        confidence_val is not None and
        confidence_val < SOCIAL_INTENT_DOUBTFUL_THRESHOLD
    )

    social_links = state.get("social_links") or {}
    if not isinstance(social_links, dict):
        social_links = {}

    available = [
        platform
        for platform in SOCIAL_DOMAINS.keys()
        if social_links.get(platform)
        or social_links.get(f"{platform}_url")
        or social_links.get(f"{platform}Url")
    ]

    if doubtful:
        return list(SOCIAL_DOMAINS.keys())
    if len(available) == 1:
        return available
    return list(SOCIAL_DOMAINS.keys())


def _ensure_scheme(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return "https://" + url


def _extract_title(html_text: str) -> str:
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", html_text or "")
    if not match:
        return ""
    title = html.unescape(match.group(1))
    return re.sub(r"\s+", " ", title).strip()


def _extract_meta_description(html_text: str) -> str:
    if not html_text:
        return ""
    patterns = [
        r'(?is)<meta[^>]+name=["\\\']description["\\\'][^>]*content=["\\\'](.*?)["\\\']',
        r'(?is)<meta[^>]+property=["\\\']og:description["\\\'][^>]*content=["\\\'](.*?)["\\\']'
    ]
    for pattern in patterns:
        match = re.search(pattern, html_text)
        if match:
            desc = html.unescape(match.group(1))
            return re.sub(r"\s+", " ", desc).strip()
    return ""


def _fetch_page_summary(url: str, timeout: int = 10, max_bytes: int = 1200000) -> dict | None:
    html_text = None
    visible_text = None
    target_url = _rewrite_social_url(url)
    log.info("Social fetch start | url=%s | target=%s", url, target_url)

    if SOCIAL_USE_PLAYWRIGHT:
        html_text, visible_text = _fetch_rendered_html(
            target_url,
            timeout_ms=SOCIAL_RENDER_TIMEOUT_MS,
            scroll_passes=SOCIAL_SCROLL_PASSES
        )

    if not html_text:
        try:
            req = urllib.request.Request(
                target_url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    )
                }
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                content_type = resp.headers.get("Content-Type", "")
                log.info("Social fetch response | url=%s | content_type=%s", target_url, content_type)
                if "text/html" not in content_type.lower():
                    return None
                raw = resp.read(max_bytes)
                html_text = raw.decode("utf-8", "ignore")
        except Exception:
            log.warning("Social fetch failed | url=%s", target_url)
            return None

    title = _extract_title(html_text)
    description = _extract_meta_description(html_text)
    text = _extract_all_text(html_text, visible_text)
    if _is_js_placeholder(text):
        log.warning("Social fetch placeholder content | url=%s", target_url)
    if len(text) > SOCIAL_TEXT_LIMIT:
        text = text[:SOCIAL_TEXT_LIMIT].rstrip() + "..."
    log.info(
        "Social fetch summary | url=%s | title=%s | description=%s | text_preview=%s",
        target_url,
        _preview_text(title, 160),
        _preview_text(description, 200),
        _preview_text(text)
    )

    return {
        "title": title,
        "description": description,
        "text": text
    }


def _decode_yahoo_redirect(url: str) -> str:
    # Yahoo web links are often wrapped as:
    # https://r.search.yahoo.com/.../RU=<encoded_target>/RK=...
    match = re.search(r"/RU=([^/]+)/RK=", url)
    if match:
        return urllib.parse.unquote(match.group(1))
    return url


def _fetch_yahoo_results(query: str, timeout: int = 10) -> list[dict]:
    endpoint = "https://search.yahoo.com/search?p=" + urllib.parse.quote_plus(query)
    req = urllib.request.Request(
        endpoint,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            page = resp.read().decode("utf-8", "ignore")
    except Exception as err:
        log.warning("Social search failed | query=%s | err=%s", query, err)
        return []

    links = re.findall(
        r"<a[^>]+href=\"([^\"]+)\"[^>]*>(.*?)</a>",
        page,
        re.IGNORECASE | re.DOTALL
    )

    output: list[dict] = []
    for raw_url, raw_title in links:
        if raw_url.startswith("#") or raw_url.startswith("/"):
            continue
        title = _strip_tags(raw_title)
        if not title:
            continue
        decoded = _decode_yahoo_redirect(raw_url)
        output.append(
            {
                "title": title,
                "url": decoded
            }
        )
    if output:
        sample = output[:3]
        preview = " | ".join(
            f"{_preview_text(item.get('title') or '', 80)} => {item.get('url')}"
            for item in sample
        )
        log.info(
            "Social search parsed | query=%s | results=%d | sample=%s",
            query,
            len(output),
            preview
        )
    else:
        log.info("Social search parsed | query=%s | results=0", query)
    return output


def _social_media_search_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    query = (state.get("query") or "").strip()
    if not query:
        return _empty_tool_result("social")

    bot_id = str(state.get("bot_id") or "").strip()
    timeout = int(state.get("tool_timeout", 20) or 20)
    max_per_platform = int(state.get("tool_max_results", 2) or 2)
    social_links = state.get("social_links") or {}
    if not isinstance(social_links, dict):
        social_links = {}

    platforms = _select_social_platforms(query, state)
    if not platforms:
        platforms = list(SOCIAL_DOMAINS.keys())

    log.info(
        "Social tool start | query=%s | timeout=%ds | max_per_platform=%d | social_links=%s | playwright=%s",
        query,
        timeout,
        max_per_platform,
        ",".join(sorted(social_links.keys())) or "none",
        int(bool(SOCIAL_USE_PLAYWRIGHT))
    )
    log.info(
        "Social platform selection | platforms=%s | intent_confidence=%s | threshold=%.2f",
        ",".join(platforms),
        str(state.get("intent_confidence", "")),
        SOCIAL_INTENT_DOUBTFUL_THRESHOLD
    )

    cached_results = _cached_social_results(
        bot_id=bot_id,
        query=query,
        limit=max(2, len(platforms) * max_per_platform)
    )
    if cached_results:
        log.info(
            "Social tool served from pre-indexed cache | bot_id=%s | query=%s | results=%d",
            bot_id,
            query,
            len(cached_results)
        )
        references = [
            {
                "type": "social",
                "platform": item.get("platform"),
                "title": item.get("title"),
                "url": item.get("url")
            }
            for item in cached_results
        ]
        return {
            "tool": "social",
            "results": cached_results,
            "references": references
        }

    if bot_id:
        try:
            threading.Thread(
                target=refresh_social_index_for_bot,
                kwargs={
                    "bot_id": bot_id,
                    "social_links": social_links,
                    "website_url": state.get("website_url"),
                    "query_hints": [query],
                    "max_results_per_platform": max_per_platform,
                },
                daemon=True,
            ).start()
            log.info("Social pre-index refresh queued | bot_id=%s", bot_id)
        except Exception as err:
            log.warning("Social pre-index refresh failed | bot_id=%s | err=%s", bot_id, err)

        cached_results = _cached_social_results(
            bot_id=bot_id,
            query=query,
            limit=max(2, len(platforms) * max_per_platform)
        )
        if cached_results:
            references = [
                {
                    "type": "social",
                    "platform": item.get("platform"),
                    "title": item.get("title"),
                    "url": item.get("url")
                }
                for item in cached_results
            ]
            return {
                "tool": "social",
                "results": cached_results,
                "references": references
            }

    if not SOCIAL_TOOL_LIVE_FALLBACK:
        log.info(
            "Social tool live fallback disabled | bot_id=%s | query=%s",
            bot_id,
            query
        )
        return _empty_tool_result("social")

    results: list[dict] = []
    seen_urls: set[str] = set()

    for platform in platforms:
        domain = SOCIAL_DOMAINS.get(platform, "")
        if not domain:
            continue
        log.info("Social platform scan | platform=%s | domain=%s", platform, domain)
        raw_link = (
            social_links.get(platform)
            or social_links.get(f"{platform}_url")
            or social_links.get(f"{platform}Url")
        )
        if raw_link:
            url = _ensure_scheme(raw_link)
            if url and url not in seen_urls and domain in url.lower():
                seen_urls.add(url)
                page_summary = _fetch_page_summary(url, timeout=timeout)
                title = (page_summary or {}).get("title") or url
                description = (page_summary or {}).get("description") or ""
                text = (page_summary or {}).get("text") or ""
                snippet = description or text[:600]
                if _is_js_placeholder(text):
                    log.warning(
                        "Social provided link placeholder | platform=%s | url=%s",
                        platform,
                        url
                    )
                results.append(
                    {
                        "platform": platform,
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                        "content": text,
                        "source": "provided_link"
                    }
                )
                log.info(
                    "Social tool used provided link | platform=%s | url=%s | title=%s | snippet=%s",
                    platform,
                    url,
                    _preview_text(title, 160),
                    _preview_text(snippet)
                )
                continue

        platform_query = f"{query} {platform} official updates"
        log.info("Social search query | platform=%s | query=%s", platform, platform_query)
        try:
            candidates = _fetch_yahoo_results(platform_query, timeout=timeout)
        except Exception:
            candidates = []

        log.info(
            "Social search candidates | platform=%s | count=%d",
            platform,
            len(candidates)
        )

        platform_hits = 0
        for item in candidates:
            url = (item.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            if domain not in url.lower():
                continue

            seen_urls.add(url)
            platform_hits += 1

            page_summary = _fetch_page_summary(url, timeout=timeout)
            page_title = (page_summary or {}).get("title") or item.get("title") or url
            page_desc = (page_summary or {}).get("description") or ""
            page_text = (page_summary or {}).get("text") or ""
            snippet = page_desc or page_text[:600] or (item.get("title") or "")
            if _is_js_placeholder(page_text):
                log.warning(
                    "Social search placeholder | platform=%s | url=%s",
                    platform,
                    url
                )
            results.append(
                {
                    "platform": platform,
                    "title": page_title,
                    "url": url,
                    "snippet": snippet,
                    "content": page_text,
                    "source": "yahoo_search"
                }
            )
            log.info(
                "Social search hit | platform=%s | title=%s | url=%s",
                platform,
                _preview_text(page_title or "", 160),
                url
            )

            if platform_hits >= max_per_platform:
                break

    references = [
        {
            "type": "social",
            "platform": item.get("platform"),
            "title": item.get("title"),
            "url": item.get("url")
        }
        for item in results
    ]

    pdf_path = None
    try:
        pdf_path = _save_social_results_pdf(
            results,
            query=query,
            bot_id=str(state.get("bot_id") or "")
        )
    except Exception as err:
        log.warning("Social PDF save failed | err=%s", err)

    if pdf_path:
        log.info("Social PDF saved | path=%s", pdf_path)

    log.info("Social tool complete | results=%d", len(results))
    return {
        "tool": "social",
        "results": results,
        "references": references
    }


def build_tool_parallel() -> RunnableParallel:
    return RunnableParallel(
        {
            "web": RunnableLambda(lambda _: _empty_tool_result("web")),
            "social": RunnableLambda(_social_media_search_tool),
            "company": RunnableLambda(lambda _: _empty_tool_result("company"))
        }
    )


def collect_tool_references(tool_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    references: List[Dict[str, Any]] = []
    for value in (tool_results or {}).values():
        refs = value.get("references") if isinstance(value, dict) else None
        if refs:
            references.extend(refs)
    return references
