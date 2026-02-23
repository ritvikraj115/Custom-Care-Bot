import fitz
import spacy
from statistics import median
import os
from app.pipeline.logger import get_logger
log = get_logger("extract")


def _load_nlp():
    model_name = str(os.getenv("SPACY_MODEL", "en_core_web_sm")).strip() or "en_core_web_sm"
    try:
        return spacy.load(model_name)
    except Exception:
        log.warning(
            "spaCy model unavailable; using sentencizer fallback | model=%s",
            model_name,
        )
        nlp_fallback = spacy.blank("en")
        if "sentencizer" not in nlp_fallback.pipe_names:
            nlp_fallback.add_pipe("sentencizer")
        return nlp_fallback


nlp = _load_nlp()

def extract_blocks_from_pdfs(pdf_paths, pdf_metadata=None):
    blocks = []
    pdf_metadata = pdf_metadata or {}

    for pdf in pdf_paths:
        doc = fitz.open(pdf)
        pdf_name = os.path.basename(pdf)
        source_meta = pdf_metadata.get(pdf_name, {})
        for page_num, page in enumerate(doc):
            for b in page.get_text("dict")["blocks"]:
                if "lines" not in b:
                    continue

                text = ""
                fonts = []

                for line in b["lines"]:
                    for span in line["spans"]:
                        text += span["text"] + " "
                        fonts.append(span["size"])

                text = text.strip()
                if len(text) < 20:
                    continue

                blocks.append({
                    "text": text,
                    "page": page_num,
                    "pdf": pdf_name,
                    "avg_font": sum(fonts) / len(fonts),
                    "source_type": source_meta.get("source_type"),
                    "source_url": source_meta.get("source_url")
                })
                log.info(f"PDFs received: {len(pdf_paths)}")

    log.info(f"Blocks extracted: {len(blocks)}")

    return blocks


def mark_headings(blocks):
    fonts = sorted([b["avg_font"] for b in blocks])
    threshold = fonts[int(0.85 * len(fonts))]

    for b in blocks:
        b["is_heading"] = (
            b["avg_font"] >= threshold or
            b["text"].isupper() or
            len(b["text"].split()) <= 6
        )
    log.info(
        f"Headings detected: {sum(b['is_heading'] for b in blocks)}"
    )
    return blocks


def sentence_stats(blocks):
    lengths = []
    for b in blocks:
        doc = nlp(b["text"])
        for s in doc.sents:
            lengths.append(len(s.text.split()))
    if not lengths:
        # Safe defaults keep downstream chunking stable when sentence parsing is empty.
        return {"median_sent_len": 12, "p75": 16, "p90": 20}
    return {
        "median_sent_len": median(lengths),
        "p75": sorted(lengths)[int(0.75 * len(lengths))],
        "p90": sorted(lengths)[int(0.9 * len(lengths))]
    }


def build_sections(blocks):
    sections = []
    current = {"title": None, "content": [], "meta": []}

    for b in blocks:
        if b["is_heading"]:
            if current["content"]:
                sections.append(current)
            current = {"title": b["text"], "content": [], "meta": []}
        else:
            current["content"].append(b["text"])
            current["meta"].append(b)

    if current["content"]:
        sections.append(current)
    log.info(f"Sections built: {len(sections)}")
    return sections
