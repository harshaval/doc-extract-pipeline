"""
Central configuration for the extraction pipeline.

Everything that is "tunable" lives here so the rest of the codebase never
hardcodes a magic number. Override any of these with environment variables
of the same name (see the `env` default below) without touching code.
"""
import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
RESULTS_DIR = OUTPUT_DIR / "results"          # final validated JSON/CSV
REVIEW_DB_PATH = OUTPUT_DIR / "review_queue" / "review_queue.db"
RAW_DUMP_DIR = OUTPUT_DIR / "raw_text"         # OCR/text dumps kept for audit

for _d in (OUTPUT_DIR, RESULTS_DIR, REVIEW_DB_PATH.parent, RAW_DUMP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# LLM extraction
# --------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-5")
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "2000"))
# Set to True to fall back to a cheaper/faster model for very short documents.
LLM_CHEAP_MODEL = os.environ.get("LLM_CHEAP_MODEL", "claude-haiku-4-5-20251001")
LLM_CHEAP_MODEL_CHAR_THRESHOLD = 800  # documents shorter than this use the cheap model

# --------------------------------------------------------------------------
# OCR
# --------------------------------------------------------------------------
# A page is treated as "scanned" (needs OCR) if PyMuPDF extracts fewer than
# this many characters of native text from it.
NATIVE_TEXT_CHAR_THRESHOLD = 20
OCR_DPI = 300  # render resolution before running tesseract; 300 is the sweet
               # spot for accuracy vs. speed on typical scanned business docs
TESSERACT_LANG = os.environ.get("TESSERACT_LANG", "eng")
TESSERACT_CONFIG = "--oem 3 --psm 6"  # oem 3 = LSTM engine, psm 6 = uniform block of text

# --------------------------------------------------------------------------
# Confidence scoring & routing
# --------------------------------------------------------------------------
# Final per-document confidence is a weighted blend of three signals:
#   1. OCR confidence   (how legible was the source text)
#   2. LLM confidence   (how sure was the model about what it extracted)
#   3. Validation score (did the extracted data pass sanity/business rules)
WEIGHT_OCR = 0.25
WEIGHT_LLM = 0.35
WEIGHT_VALIDATION = 0.40

# Below this aggregate score, a document is auto-routed to the human review queue.
REVIEW_THRESHOLD = 0.80
# Below this, a document is flagged as likely unusable even after review
# (still queued, but marked "critical" so a human sees it first).
CRITICAL_THRESHOLD = 0.50

# Individual field-level confidence below this gets flagged in the review UI
# even if the document-level score passes.
FIELD_REVIEW_THRESHOLD = 0.75

# --------------------------------------------------------------------------
# Table extraction
# --------------------------------------------------------------------------
TABLE_MIN_ROWS = 2  # ignore tiny/spurious table detections
