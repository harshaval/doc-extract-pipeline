# Document Extraction Pipeline: OCR + LLM + Validation + Human Review

A pipeline that turns messy PDFs — scanned invoices, photographed receipts, native
digital documents with garbled tables — into clean, validated structured data, and
routes anything it isn't confident about to a human review queue instead of silently
guessing.

Ships configured for **invoices/receipts** as the concrete example (vendor, line
items, totals, tax), because that's the most common real-world "messy PDF" extraction
task and has enough numeric structure to make the validation layer meaningful to
demonstrate. The schema is intentionally isolated to one file (`src/schemas.py`) plus
one tool definition (`src/llm_extraction.py`), so retargeting the whole pipeline at a
different document type — resumes, lab reports, contracts — is a schema swap, not a
rewrite.

The per-document pipeline (ingestion → table extraction/OCR → LLM extraction →
validation → routing) is orchestrated as a **LangGraph `StateGraph`**
(`src/graph.py`), and the LLM extraction step uses **LangChain**'s
`ChatAnthropic` with `with_structured_output(...)` rather than calling the
Anthropic SDK directly. See "LLM extraction" and "Orchestration" below for
why.

---

## Why this exists

OCR alone gives you noisy text. An LLM alone will confidently hallucinate a total
that was never on the page. The interesting engineering problem isn't "call an OCR
library" or "call an LLM" — it's building a pipeline that knows when *not* to trust
its own output, and hands exactly those cases to a person, instead of either (a)
publishing wrong data with total confidence, or (b) sending everything to a human and
defeating the point of automating this at all.

That's what the four layers below are each responsible for.

---

## Architecture

```
                          ┌─────────────────┐
   messy PDF ────────────▶│   1. INGESTION   │  classify each page:
                          │  (src/ingestion) │  native text layer, or scanned image?
                          └────────┬─────────┘
                                   │
                 ┌─────────────────┴──────────────────┐
                 ▼                                     ▼
        native-text page                        scanned/image page
                 │                                     │
                 ▼                                     ▼
      ┌─────────────────────┐              ┌───────────────────────┐
      │ 2a. TABLE EXTRACTION │              │      2b. OCR           │
      │ (pdfplumber, grid    │              │  (tesseract, w/ Otsu    │
      │  tables → markdown)  │              │  threshold preprocessing,│
      └──────────┬───────────┘              │  per-word confidence)    │
                 │                          └────────────┬─────────────┘
                 └───────────────┬────────────────────────┘
                                 ▼
                     merged per-document text
                                 │
                                 ▼
                  ┌───────────────────────────┐
                  │  3. LLM EXTRACTION          │  Claude via LangChain's ChatAnthropic,
                  │  (src/llm_extraction)        │  forced structured output,
                  └──────────────┬───────────────┘  self-reported per-field confidence
                                 ▼
                  ┌───────────────────────────┐
                  │  4. VALIDATION               │  deterministic rule checks:
                  │  (src/validation)             │  required fields, arithmetic
                  └──────────────┬───────────────┘  cross-checks, date sanity, etc.
                                 ▼
                    combine OCR + LLM + validation
                       into one confidence score
                                 │
                    ┌────────────┴────────────┐
                    ▼                          ▼
            confidence ≥ threshold      confidence < threshold
                    │                          │
                    ▼                          ▼
             auto-approved,             ┌─────────────────────┐
             written to results/         │ 5. HUMAN REVIEW QUEUE │
                                          │ (SQLite + CLI tool)    │
                                          └─────────────────────┘
```

### 1. Ingestion (`src/ingestion.py`)

Opens the PDF with PyMuPDF and classifies **each page independently** as native-text
or scanned, based on how many characters PyMuPDF can pull from the page's text layer.
This is deliberately per-page, not per-document — a lot of real-world PDFs mix a
digitally-generated cover page with scanned attachments, and treating the whole file
as "one type" would silently skip OCR on pages that need it (or waste time OCR'ing
pages that don't).

Scanned pages get rendered to a 300-DPI image for the OCR step; native pages keep
their extracted text as-is.

### 2a. Table extraction (`src/table_extraction.py`)

For native-text pages, `pdfplumber` detects grid-structured tables (ones with actual
ruled lines/cell boundaries) and renders them as markdown, which gets appended to the
page text. LLMs parse line-item tables far more reliably from an actual markdown grid
than from PDF text where columns have collapsed into a run-on line.

This is an enhancement, not a dependency — plenty of real invoices don't use ruled
tables at all (just aligned whitespace), and for those, the LLM parses the raw text
directly using column-alignment cues in its prompt (see step 3). If pdfplumber finds
nothing, or the PDF is malformed, this step degrades to a no-op rather than failing
the document.

*Efficiency note:* the pdfplumber document is opened **once** per PDF and reused
across all native pages, not reopened per page — reparsing PDF structure per page
was an early, since-fixed inefficiency (see `table_extraction.open_document`).

### 2b. OCR (`src/ocr.py`)

Wraps `pytesseract` with:
- **Otsu thresholding** — grayscale + adaptive binarization before OCR, which
  measurably improves accuracy on low-contrast phone-photo-style scans without risking
  already-clean scans (Otsu adapts to the image rather than using a fixed cutoff).
- **Real confidence scoring** — tesseract reports a confidence per detected word (and
  `-1` for non-text layout boxes). This module filters out the `-1`s and averages the
  rest into one 0.0–1.0 page confidence, which becomes part of the document's overall
  OCR confidence and, downstream, part of the routing decision.

### 3. LLM extraction (`src/llm_extraction.py`)

The merged document text (native + OCR'd + rendered tables) goes to Claude via
**LangChain's `ChatAnthropic`**, bound to the `InvoiceExtraction` Pydantic schema with
`with_structured_output(InvoiceExtraction, method="function_calling")`. Under the
hood this still forces a single tool call matching the schema — the LangChain-native
equivalent of a raw forced tool-use call — rather than asking the model to "please
respond in JSON." This is the difference between *usually* valid JSON and *always*
valid JSON: the schema is enforced at the API level, not hoped for via prompting.

The tool schema requires the model to return a **confidence score per field**, along
with a brief reason for any field it wasn't sure about (e.g. *"OCR text was smudged
near this value"*). This isn't a substitute for genuine model calibration, but in
practice it's a strong, cheap signal — Claude reliably reports lower confidence when a
value was ambiguous, inferred, or contradicted by other parts of the document, and
that signal is exactly what should route a document to a human rather than downstream
systems silently ingesting a guess.

The system prompt explicitly instructs the model to set a field to `null` rather than
invent a plausible-looking value when the source text doesn't support it — this
matters more than almost anything else in the prompt, since hallucinated-but-plausible
numbers are the failure mode that's hardest to catch downstream.

Short documents route to a cheaper/faster model (`LLM_CHEAP_MODEL` in `config.py`);
longer or more complex ones use the full model — configurable per your cost/accuracy
tradeoff.

### 4. Validation (`src/validation.py`)

This is the layer that catches what the LLM's self-reported confidence can't:
internal inconsistency. Two extractions can have identical LLM-reported confidence
while one is arithmetically self-consistent and the other isn't — this layer is what
tells them apart, using **deterministic checks that never trust the model's own
assessment of itself**:

- Required fields present (vendor name, total amount → errors if missing; invoice
  number → warning, since some receipts genuinely don't have one)
- Line items sum to the stated subtotal (within a small rounding tolerance)
- `quantity × unit_price == line_total` per line item
- `subtotal + tax == total` 
- Dates parse against a list of common formats; due date isn't before invoice date
- Currency values look like currency codes/symbols, not garbage
- No negative totals

Each issue is either an `error` (large confidence penalty, almost certainly wrong) or
a `warning` (small penalty, unusual but plausible). The resulting validation score is
one of three inputs to the document's overall confidence.

### Orchestration (`src/graph.py`)

Steps 1–4 plus routing are wired together as a **LangGraph `StateGraph`**, not a plain
function pipeline. Each stage (`ingest`, `extract_tables`, `ocr_pages`, `merge_text`,
`llm_extract`, `validate`, `score`) is a graph node operating on a shared
`PipelineState`. The graph runs linearly through those nodes, then hits the one
graph-level branch that matters: a **conditional edge out of `score`** into either
`finalize_approved` or `finalize_review`, mirroring the fork in the diagram above.

Per-page native-vs-scanned branching deliberately stays *inside* the
`extract_tables`/`ocr_pages` nodes rather than becoming graph-level branches — a single
document mixes page types (see "Ingestion" above), so the natural unit of LangGraph
branching here is the document-level routing decision, not per-page type. `src/pipeline.py`
compiles the graph once and calls `graph.invoke(...)` per document.

### Confidence scoring & routing

```python
overall_confidence = (
    0.25 * ocr_confidence        # how legible was the source
  + 0.35 * llm_confidence        # how sure was the model, per its own report
  + 0.40 * validation_score      # did the data pass deterministic sanity checks
)
```

Weights live in `config.py` (`WEIGHT_OCR`, `WEIGHT_LLM`, `WEIGHT_VALIDATION`) —
validation is weighted highest deliberately, since it's the only fully deterministic
signal of the three.

A document is routed to the **human review queue** if *any* of the following is true:
- `overall_confidence` < `REVIEW_THRESHOLD` (default `0.80`)
- any single field's LLM-reported confidence < `FIELD_REVIEW_THRESHOLD` (default `0.75`)
- validation produced at least one `error`-severity issue

Documents below `CRITICAL_THRESHOLD` (default `0.50`) are additionally flagged
`is_critical` and sorted to the top of the review queue.

### 5. Human review queue (`src/review_queue.py`, `review_app/cli_review.py`)

A plain SQLite table (`output/review_queue/review_queue.db`) — no ORM, on purpose;
it's meant to be readable top-to-bottom by whoever eventually swaps it for Postgres.
**Every** processed document gets a row, not just flagged ones, so you always have a
full audit trail of what confidence a document scored and why, alongside the original
(untouched) LLM output and any human edits.

`python main.py review` launches an interactive terminal session: critical/
lowest-confidence documents first, full extracted data + flagged low-confidence fields
+ validation issues printed together, with `[a]pprove / [e]dit / [r]eject / [s]kip /
[q]uit` actions. Editing accepts `field=value, field=value` and writes the corrected
record back to the same row, preserving the original for audit.

---

## Project structure

```
doc-extract-pipeline/
├── config.py                  # all tunable thresholds/weights/paths in one place
├── main.py                    # CLI entrypoint (extract / review / stats)
├── requirements.txt
├── src/
│   ├── schemas.py              # domain schema (InvoiceExtraction) + pipeline models
│   ├── ingestion.py             # PDF page classification (native vs. scanned)
│   ├── ocr.py                    # tesseract wrapper w/ preprocessing + confidence
│   ├── table_extraction.py        # pdfplumber grid-table → markdown
│   ├── llm_extraction.py           # ChatAnthropic (LangChain) structured extraction
│   ├── validation.py                # deterministic business-rule checks
│   ├── graph.py                       # LangGraph StateGraph wiring the pipeline nodes
│   ├── review_queue.py               # SQLite queue: enqueue/list/resolve/stats
│   └── pipeline.py                    # compiles the graph, invokes it, writes outputs
├── review_app/
│   └── cli_review.py            # interactive terminal review session
├── sample_data/
│   └── generate_samples.py     # generates a clean + a synthetic "scanned" test PDF
├── tests/                       # pytest suite (34 tests, no API key required)
└── output/                      # created at runtime
    ├── results/                  # final validated JSON per document
    ├── raw_text/                  # merged native+OCR text per document, for audit
    └── review_queue/               # review_queue.db
```

---

## Setup

```bash
pip install -r requirements.txt

# Tesseract and poppler (for pdf2image/PyMuPDF rendering) must be installed
# at the OS level — they're not pip packages:
#   Ubuntu/Debian: sudo apt-get install tesseract-ocr poppler-utils
#   macOS:         brew install tesseract poppler

export ANTHROPIC_API_KEY=sk-ant-...
```

The LLM extraction step and pipeline orchestration are built on
[LangChain](https://python.langchain.com/) (`langchain-anthropic`'s `ChatAnthropic`)
and [LangGraph](https://langchain-ai.github.io/langgraph/) (`StateGraph`) — both are
listed in `requirements.txt` and installed by the command above.

Optionally override any setting in `config.py` via environment variable, e.g.:
```bash
export LLM_MODEL=claude-opus-4-8       # more accuracy, higher cost
export REVIEW_THRESHOLD=0.90            # stricter auto-approval bar
```

## Usage

```bash
# Generate two sample PDFs (one clean, one synthetic "scanned") to try the pipeline
# without needing real documents on hand:
python sample_data/generate_samples.py

# Extract a single PDF
python main.py extract sample_data/sample_native_clean.pdf

# Extract every PDF in a folder
python main.py extract sample_data/

# Review whatever landed in the queue
python main.py review

# Queue summary
python main.py stats
```

`extract` prints a one-line summary per document (confidence + routing decision),
writes full results to `output/results/<document_id>.json`, and tells you how many
documents are waiting in the review queue, if any.

## Testing

```bash
pip install pytest
pytest tests/ -v
```

34 tests covering validation rules (clean/broken arithmetic, missing fields, date
sanity), schema behavior (confidence clamping, optional-field handling), the review
queue (routing, editing, resolving, stats), and the LangGraph routing logic (auto-approve
vs. review vs. critical at each threshold) — all run without an Anthropic API key or
network access, using `llm_extraction.extract_from_json_fixture()` to bypass the API
call and exercise everything downstream of it directly.

To test the OCR/ingestion path specifically without an API key:
```python
from src import ingestion, ocr
pages = ingestion.classify_and_load_pages("sample_data/sample_scanned_noisy.pdf")
result = ocr.ocr_page(pages[0].image, pages[0].page_number)
print(result.confidence, result.text)
```

---

## Design decisions worth knowing about

- **Structured output over prompted JSON.** `ChatAnthropic.with_structured_output(...)`
  guarantees schema-valid output at the API level instead of hoping the model followed
  formatting instructions. This is the single biggest reliability lever in the whole
  pipeline.
- **LangGraph for orchestration, but only one real conditional edge.** The pipeline is a
  `StateGraph` so each stage is an inspectable, independently testable node, but the
  graph is deliberately not over-branched: per-page native/scanned handling stays as
  data-level filtering inside nodes, and the single graph-level branch is the
  auto-approve/review fork, which is the one decision that's actually document-level.
- **Review queue stays outside the graph.** Reviewing a document doesn't resume a
  LangGraph run via `interrupt()` — it's a separate, decoupled SQLite + CLI step. This
  keeps the review workflow async/batchable (review whenever, not "while the graph run
  is still live") and keeps `review_app/` swappable without touching the graph.
- **Per-page, not per-document, OCR classification.** Mixed digital/scanned PDFs are
  common enough in practice that a document-level "is this scanned?" flag would
  misclassify real files.
- **Validation never trusts the model's self-reported confidence.** It's a fully
  independent, deterministic signal specifically so that a confidently-wrong
  extraction (plausible-looking hallucinated numbers, high self-reported confidence)
  still gets caught by arithmetic that doesn't add up.
- **Document IDs are content hashes, not filenames.** Re-processing identical bytes
  under a different filename doesn't create a duplicate queue entry; a modified file
  under the same filename does get reprocessed.
- **Every document gets a queue row, not just flagged ones.** The queue doubles as an
  audit log — you can always answer "what confidence did document X get and why?"
  even for documents nobody needed to review.
- **SQLite + a terminal UI, not Postgres + a web app.** Right-sized for a project like
  this: no server to run or secure, and the `review_queue.py` module has zero
  CLI-specific logic in it, so swapping the front end for Streamlit or Flask later
  only touches `review_app/`, not the queue logic itself.

## Known limitations / where this would grow next

- **Table detection on scanned pages** relies on the LLM parsing loosely-aligned OCR
  text rather than true table-structure detection (tesseract's own table/layout
  detection is unreliable enough on real-world scans that leaning on the LLM's
  language understanding wins out in practice). A production version handling dense
  scanned tables would add layout-aware OCR (e.g. `pytesseract.image_to_data`'s
  bounding boxes, clustered into rows/columns) as a preprocessing step.
- **Date parsing** covers common formats but isn't locale-exhaustive; extend
  `DATE_FORMATS` in `validation.py` for other regions' conventions.
- **No retry/backoff on the Anthropic API call** — add if you're running this over a
  large unattended batch and want resilience to transient API errors.
- **Single-schema-at-a-time.** Multiple document types in one pipeline run would need
  a classification step before extraction (cheap to add: one more LLM call, or a
  keyword heuristic, to pick which schema/tool to use per document).
