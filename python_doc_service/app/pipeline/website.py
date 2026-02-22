import html
import os
import re
import textwrap
import urllib.parse
import urllib.request
from typing import List, Dict, Tuple
import asyncio
import concurrent.futures

try:
    from playwright.sync_api import sync_playwright
except Exception:
    sync_playwright = None

try:
    import trafilatura
except Exception:
    trafilatura = None

import fitz

from app.pipeline.logger import get_logger

log = get_logger("website")

_PLAYWRIGHT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2)

DEFAULT_MAX_PAGES = int(os.getenv("WEBSITE_CRAWL_MAX_PAGES", "25"))
DEFAULT_TIMEOUT = int(os.getenv("WEBSITE_CRAWL_TIMEOUT", "10"))
DEFAULT_MAX_BYTES = int(os.getenv("WEBSITE_CRAWL_MAX_BYTES", "2000000"))
DEFAULT_TEXT_LIMIT = int(os.getenv("WEBSITE_TEXT_LIMIT", "100000"))
DEFAULT_LOG_PREVIEW = int(os.getenv("WEBSITE_LOG_PREVIEW_CHARS", "400"))
DEFAULT_RENDER_TIMEOUT_MS = int(os.getenv("WEBSITE_RENDER_TIMEOUT_MS", "25000"))
DEFAULT_SCROLL_PASSES = int(os.getenv("WEBSITE_SCROLL_PASSES", "2"))
DEFAULT_USE_PLAYWRIGHT = os.getenv("WEBSITE_USE_PLAYWRIGHT", "true").strip().lower() in {
    "1", "true", "yes", "y"
}
DEFAULT_RENDER_WAIT_MS = int(os.getenv("WEBSITE_RENDER_WAIT_MS", "2500"))
DEFAULT_RENDER_EXTRA_WAIT_MS = int(os.getenv("WEBSITE_RENDER_EXTRA_WAIT_MS", "3500"))


def _preview_text(text: str, limit: int | None = None) -> str:
    if not text:
        return ""
    limit = limit or DEFAULT_LOG_PREVIEW
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "..."


_SKIP_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
    ".pdf", ".zip", ".rar", ".7z", ".tar", ".gz",
    ".mp4", ".mp3", ".mov", ".avi", ".mkv",
    ".css", ".js", ".json", ".xml"
}


def _ensure_scheme(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return "https://" + url


def _normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    url, _ = urllib.parse.urldefrag(url)
    return url.rstrip("/")


def _root_domain(netloc: str) -> str:
    host = (netloc or "").split(":")[0].lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _same_domain(url: str, root: str) -> bool:
    if not url:
        return False
    try:
        netloc = urllib.parse.urlparse(url).netloc
    except Exception:
        return False
    if not netloc:
        return False
    netloc = netloc.lower()
    root = root.lower()
    return netloc == root or netloc.endswith("." + root)


def _extract_links(html_text: str, base_url: str) -> List[str]:
    if not html_text:
        return []

    links = re.findall(r"href=[\"']([^\"']+)[\"']", html_text, flags=re.IGNORECASE)
    output: List[str] = []
    for raw in links:
        href = (raw or "").strip()
        if not href:
            continue
        lowered = href.lower()
        if lowered.startswith("mailto:") or lowered.startswith("tel:"):
            continue
        if lowered.startswith("javascript:"):
            continue

        joined = urllib.parse.urljoin(base_url, href)
        if not joined.startswith("http://") and not joined.startswith("https://"):
            continue

        path = urllib.parse.urlparse(joined).path.lower()
        if any(path.endswith(ext) for ext in _SKIP_EXTENSIONS):
            continue

        output.append(_normalize_url(joined))

    return output


def _strip_tags(html_text: str) -> str:
    if not html_text:
        return ""

    cleaned = re.sub(
        r"(?is)<(script|style|noscript|svg|footer|nav|header|aside)[^>]*>.*?</\1>",
        " ",
        html_text
    )
    cleaned = re.sub(r"(?is)<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


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


def _extract_title(html_text: str) -> str:
    if not html_text:
        return ""
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", html_text)
    if not match:
        return ""
    title = match.group(1)
    title = html.unescape(title)
    return re.sub(r"\s+", " ", title).strip()


def _fetch_raw(url: str, timeout: int, max_bytes: int) -> str | None:
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            }
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(max_bytes)
            return raw.decode("utf-8", "ignore")
    except Exception:
        return None


def _fetch_html(url: str, timeout: int, max_bytes: int) -> str | None:
    try:
        log.info("Website fetch start | url=%s", url)
        req = urllib.request.Request(
            url,
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
            log.info("Website fetch response | url=%s | content_type=%s", url, content_type)
            if "text/html" not in content_type.lower():
                log.info("Website fetch skipped (non-html) | url=%s", url)
                return None
            raw = resp.read(max_bytes)
            return raw.decode("utf-8", "ignore")
    except Exception as err:
        log.warning("Failed to fetch %s | %s", url, err)
        return None


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
                page.wait_for_timeout(DEFAULT_RENDER_WAIT_MS)
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
                    page.wait_for_timeout(DEFAULT_RENDER_EXTRA_WAIT_MS)
                    try:
                        visible_text = page.inner_text("body")
                    except Exception:
                        try:
                            visible_text = page.evaluate("document.body.innerText")
                        except Exception:
                            pass
                    html_text = page.content()
        except Exception as err:
            log.warning("Playwright fetch failed | url=%s | err=%s", url, err)
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
        log.info("Website render via thread | url=%s", url)
        future = _PLAYWRIGHT_EXECUTOR.submit(_render_sync)
        return future.result()
    except RuntimeError:
        return _render_sync()


def _fetch_sitemap_urls(root_url: str, timeout: int, max_bytes: int) -> List[str]:
    candidates = [
        root_url.rstrip("/") + "/sitemap.xml",
        root_url.rstrip("/") + "/sitemap_index.xml"
    ]
    urls: List[str] = []
    for sitemap_url in candidates:
        raw = _fetch_raw(sitemap_url, timeout=timeout, max_bytes=max_bytes)
        if not raw:
            continue
        locs = re.findall(r"(?is)<loc>(.*?)</loc>", raw)
        if locs:
            urls.extend([loc.strip() for loc in locs if loc.strip()])
            log.info(
                "Sitemap discovered | url=%s | entries=%d",
                sitemap_url,
                len(locs)
            )
    return urls


def crawl_website(
    root_url: str,
    max_pages: int | None = None,
    timeout: int | None = None,
    max_bytes: int | None = None,
    text_limit: int | None = None,
    render_timeout_ms: int | None = None,
    scroll_passes: int | None = None,
    use_playwright: bool | None = None
) -> List[Dict[str, str]]:
    root_url = _ensure_scheme(root_url)
    if not root_url:
        log.warning("Website crawl skipped | reason=empty_url")
        return []

    parsed = urllib.parse.urlparse(root_url)
    root_host = _root_domain(parsed.netloc)
    if not root_host:
        log.warning("Website crawl skipped | reason=invalid_root | url=%s", root_url)
        return []

    max_pages = max_pages or DEFAULT_MAX_PAGES
    timeout = timeout or DEFAULT_TIMEOUT
    max_bytes = max_bytes or DEFAULT_MAX_BYTES
    text_limit = text_limit or DEFAULT_TEXT_LIMIT
    render_timeout_ms = render_timeout_ms or DEFAULT_RENDER_TIMEOUT_MS
    scroll_passes = scroll_passes if scroll_passes is not None else DEFAULT_SCROLL_PASSES
    use_playwright = DEFAULT_USE_PLAYWRIGHT if use_playwright is None else use_playwright

    log.info(
        "Website crawl start | root=%s | max_pages=%d | timeout=%ds | max_bytes=%d | text_limit=%d | playwright=%s",
        root_url,
        max_pages,
        timeout,
        max_bytes,
        text_limit,
        int(bool(use_playwright))
    )

    queue = []
    sitemap_urls = _fetch_sitemap_urls(root_url, timeout=timeout, max_bytes=max_bytes)
    for item in sitemap_urls:
        if _same_domain(item, root_host):
            queue.append(_normalize_url(item))

    queue.append(_normalize_url(root_url))
    if sitemap_urls:
        log.info("Sitemap URLs queued | count=%d", len(queue))
    visited = set()
    pages: List[Dict[str, str]] = []

    while queue and len(pages) < max_pages:
        url = queue.pop(0)
        if not url or url in visited:
            continue
        visited.add(url)

        html_text = None
        visible_text = None
        if use_playwright:
            html_text, visible_text = _fetch_rendered_html(
                url,
                timeout_ms=render_timeout_ms,
                scroll_passes=scroll_passes
            )
        if not html_text:
            html_text = _fetch_html(url, timeout=timeout, max_bytes=max_bytes)

        if not html_text:
            log.info("Website crawl skip | url=%s | reason=empty_html", url)
            continue

        title = _extract_title(html_text)
        text = _extract_all_text(html_text, visible_text)
        if not text:
            log.info("Website crawl skip | url=%s | reason=empty_text", url)
            continue

        if _is_js_placeholder(text):
            log.info("Website crawl skip | url=%s | reason=js_placeholder", url)
            continue

        if len(text) > text_limit:
            text = text[:text_limit].rstrip() + "..."

        pages.append({
            "url": url,
            "title": title,
            "text": text
        })

        links = _extract_links(html_text, base_url=url)
        sample_links = ", ".join(links[:5])
        log.info(
            "Website crawl page | url=%s | title=%s | text_chars=%d | links=%d | sample_links=%s",
            url,
            title[:120] if title else "",
            len(text),
            len(links),
            sample_links
        )
        log.info(
            "Website crawl page preview | url=%s | text=%s",
            url,
            _preview_text(text)
        )
        for link in links:
            if not _same_domain(link, root_host):
                continue
            if link in visited:
                continue
            queue.append(link)

    log.info("Website crawl complete | pages=%d | root=%s", len(pages), root_url)
    return pages


def _split_text(text: str, max_chars: int) -> List[str]:
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    parts = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        parts.append(text[start:end])
        start = end
    return parts


def build_website_pdf(
    pages: List[Dict[str, str]],
    output_path: str,
    title: str | None = None
) -> Tuple[str, int]:
    if not pages:
        raise ValueError("No website pages to render")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    log.info(
        "Website PDF build start | output=%s | pages=%d | title=%s",
        output_path,
        len(pages),
        (title or "").strip()
    )

    doc = fitz.open()
    page_width = 612
    page_height = 792
    margin = 36
    font_size = 10
    max_chars = 3200

    header_title = (title or "Website Crawl").strip() or "Website Crawl"

    for page in pages:
        url = page.get("url", "")
        page_title = page.get("title", "")
        body = page.get("text", "")

        header_lines = [header_title]
        if url:
            header_lines.append(f"URL: {url}")
        if page_title:
            header_lines.append(f"Title: {page_title}")
        header = "\n".join(header_lines) + "\n\n"

        wrapped = "\n".join(
            textwrap.fill(line, 100)
            for line in (header + body).splitlines()
            if line.strip()
        )

        for chunk in _split_text(wrapped, max_chars=max_chars):
            pdf_page = doc.new_page(width=page_width, height=page_height)
            rect = fitz.Rect(margin, margin, page_width - margin, page_height - margin)
            pdf_page.insert_textbox(rect, chunk, fontsize=font_size)

    doc.save(output_path)
    doc.close()

    log.info("Website PDF build complete | output=%s", output_path)
    return output_path, len(pages)


def make_website_pdf(
    website_url: str,
    output_dir: str,
    bot_id: str | None = None
) -> Tuple[str, int] | None:
    website_url = (website_url or "").strip()
    if not website_url:
        log.warning("Website PDF skipped | reason=empty_url")
        return None

    safe_bot = re.sub(r"[^a-zA-Z0-9_-]", "_", bot_id or "bot")
    domain = urllib.parse.urlparse(_ensure_scheme(website_url)).netloc
    domain = re.sub(r"[^a-zA-Z0-9_-]", "_", domain or "website")

    filename = f"website_{domain}_{safe_bot}.pdf"
    output_path = os.path.join(output_dir, filename)

    pages = crawl_website(website_url)
    if not pages:
        log.warning("Website PDF skipped | reason=no_pages | url=%s", website_url)
        return None

    return build_website_pdf(pages, output_path, title=f"Website Crawl: {domain}")
