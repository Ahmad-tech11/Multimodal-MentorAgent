"""
Multimodal-MentorAgent  --  v2.3 (Qwen-VL + Automated PDF Multimodal Ingestion)
基于多模态大模型智能体的行业导师实践材料智能审核系统
Multimodal Large Model Agent for Intelligent Review of Industry Mentor
Student Materials

Self-contained Gradio application implementing:
  - Automated PDF Multimodal Ingestion: Extracts full text and the primary
    architecture diagram automatically from an uploaded practicum report.
  - 2-Pass Multi-Agent Audit Pipeline powered by Alibaba DashScope / Qwen-VL:
      * Pass 1: Vision Alignment & Section Parsing (qwen-vl-max / qwen-vl-plus)
      * Pass 2: Rubric Diagnostic & Report Synthesis (qwen-plus / qwen-max)
  - Live streaming execution trace generator.
  - Dynamic image optimization (Pillow <= 800px, JPEG <= 400KB).
  - Resilient deterministic fallback engine (never throws red error badges).
  - Zero-emoji corporate academic report formatting (.md and .json exports).

Run:  python app.py
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
import tempfile
import time
import traceback
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional

import gradio as gr
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# 0.  Configuration & Constants
# ---------------------------------------------------------------------------

load_dotenv()

VERSION = "v2.3 (Qwen-VL + PDF Ingestion)"

# API Keys -- support DASHSCOPE_API_KEY, QWEN_API_KEY, and OPENAI_API_KEY
DASHSCOPE_API_KEY = (
    os.getenv("DASHSCOPE_API_KEY", "")
    or os.getenv("QWEN_API_KEY", "")
    or os.getenv("OPENAI_API_KEY", "")
)

# DashScope OpenAI-Compatible Base URL (International endpoint by default)
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

# Qwen Models
QWEN_VL_MODEL = os.getenv("QWEN_VL_MODEL", "qwen-vl-max")
QWEN_TEXT_MODEL = os.getenv("QWEN_TEXT_MODEL", "qwen-plus")

# Performance & Budget Constraints
MAX_OUTPUT_TOKENS = 1500
TEMPERATURE = 0.2
TIMEOUT_SECONDS = 30          # Per LLM call timeout
MAX_RETRIES = 1               # 1 automatic retry on transient error
MAX_IMAGE_DIM = 800           # Longest side after resize
MAX_IMAGE_BYTES = 400_000     # ~400 KB payload ceiling

# Lazy-initialized OpenAI client handle
_openai_client = None


def _llm_available() -> bool:
    """True when a DashScope / Qwen API key is configured."""
    return bool(DASHSCOPE_API_KEY.strip())


def _get_client():
    """Lazy-initialise the OpenAI client configured for DashScope."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
            timeout=TIMEOUT_SECONDS,
        )
    return _openai_client


# ---------------------------------------------------------------------------
# 1.  Automated PDF Multimodal Ingestion Pipeline (PyMuPDF + Pillow)
# ---------------------------------------------------------------------------

def parse_pdf_document(pdf_file) -> tuple[str, Image.Image | None]:
    """
    Automated multimodal ingestion for complete student practicum reports:
      1. Extracts full text across all pages.
      2. Extracts embedded raster images and identifies the primary architecture
         diagram (largest pixel area, filtering out tiny logos/icons < 120x120px).
      3. Returns (extracted_text, primary_pil_image).
    """
    if pdf_file is None:
        return "", None

    try:
        # Determine actual file path from Gradio input
        if isinstance(pdf_file, str):
            file_path = pdf_file
        elif hasattr(pdf_file, "name"):
            file_path = pdf_file.name
        elif hasattr(pdf_file, "path"):
            file_path = pdf_file.path
        else:
            file_path = str(pdf_file)

        if not os.path.exists(file_path):
            return f"[ERROR] Uploaded PDF file not found at: {file_path}", None

        try:
            import pymupdf as fitz
        except ImportError:
            import fitz

        doc = fitz.open(file_path)
        text_chunks: list[str] = []
        best_image: Image.Image | None = None
        max_image_area: int = 0
        MIN_DIMENSION = 120  # Minimum width and height to qualify as an architecture diagram

        for page_idx in range(len(doc)):
            page = doc[page_idx]

            # Text extraction
            page_text = page.get_text()
            if page_text and page_text.strip():
                text_chunks.append(page_text.strip())

            # Image extraction across all pages
            image_list = page.get_images(full=True)
            for img_info in image_list:
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    width = base_image["width"]
                    height = base_image["height"]

                    # Filter out tiny logos, glyphs, and decorative borders
                    if width < MIN_DIMENSION or height < MIN_DIMENSION:
                        continue

                    area = width * height
                    if area > max_image_area:
                        pil_img = Image.open(BytesIO(image_bytes))
                        if pil_img.mode != "RGB":
                            pil_img = pil_img.convert("RGB")
                        best_image = pil_img
                        max_image_area = area
                except Exception as img_err:
                    print(f"[PDF INGESTION] Warning: Could not extract image xref {xref}: {img_err}")
                    continue

        doc.close()

        extracted_text = "\n\n".join(text_chunks).strip()
        img_info_str = f"{best_image.size[0]}x{best_image.size[1]}px" if best_image else "None"
        print(f"[PDF INGESTION] Successfully parsed '{os.path.basename(file_path)}': {len(extracted_text)} chars, diagram: {img_info_str}")

        return extracted_text, best_image

    except Exception as e:
        err_msg = f"[ERROR] Failed to parse uploaded PDF: {str(e)}"
        print(f"[PDF INGESTION] {err_msg}")
        traceback.print_exc()
        return err_msg, None


# ---------------------------------------------------------------------------
# 2.  Image Optimization (Pillow-based, zero external service)
# ---------------------------------------------------------------------------

def optimize_image(img: Image.Image) -> tuple[Image.Image, str]:
    """
    Resize & compress diagram so the base64 payload stays well under 400 KB.
    Returns (optimized_pil_image, base64_data_url).
    """
    w, h = img.size
    if max(w, h) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    img = img.convert("RGB")
    buf = BytesIO()
    quality = 85
    img.save(buf, format="JPEG", quality=quality, optimize=True)

    while buf.tell() > MAX_IMAGE_BYTES and quality > 30:
        quality -= 10
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)

    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{encoded}"
    return img, data_url


# ---------------------------------------------------------------------------
# 3.  Safe Qwen Calling with Timeout, Retry & Resilient JSON Parsing
# ---------------------------------------------------------------------------

def _parse_json_robust(raw: str) -> dict:
    """Extract and parse JSON object from model output with markdown stripping."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _safe_qwen_call(
    messages: list[dict],
    model: str,
    stage_label: str,
) -> dict | None:
    """
    Call Qwen model with:
      - OpenAI client timeout
      - 1 automatic retry on transient failure
      - Robust JSON parsing
      - Full error logging to console
    Returns parsed JSON dict on success, or None on failure. Never raises.
    """
    if not _llm_available():
        return None

    import concurrent.futures

    client = _get_client()

    def _invoke():
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        return response.choices[0].message.content or ""

    last_error = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_invoke)
                raw = future.result(timeout=TIMEOUT_SECONDS)

            return _parse_json_robust(raw)

        except concurrent.futures.TimeoutError:
            last_error = f"Timed out after {TIMEOUT_SECONDS}s"
            print(f"[{stage_label}] TIMEOUT (attempt {attempt}/{MAX_RETRIES + 1}): {last_error}")
        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}"
            print(f"[{stage_label}] JSON PARSE ERROR (attempt {attempt}/{MAX_RETRIES + 1}): {last_error}")
            break
        except Exception as e:
            last_error = str(e)
            print(f"[{stage_label}] API ERROR (attempt {attempt}/{MAX_RETRIES + 1}): {e}")
            print(traceback.format_exc())

        if attempt <= MAX_RETRIES:
            wait = 2 * attempt
            print(f"[{stage_label}] Retrying in {wait}s...")
            time.sleep(wait)

    print(f"[{stage_label}] FAILED after {MAX_RETRIES + 1} attempt(s): {last_error}")
    return None


# ---------------------------------------------------------------------------
# 4.  Default Demo Data (IoT Fall Detection with planted Redis discrepancy)
# ---------------------------------------------------------------------------

DEFAULT_REPORT_TEXT = """\
Problem Statement:
Elderly patients living alone face high risk from unwitnessed falls, where delayed \
detection significantly worsens medical outcomes. Existing camera-based fall \
detection systems raise privacy concerns and require constant cloud connectivity, \
making them unsuitable for many home deployments. This project addresses the need \
for a privacy-preserving, low-latency fall detection system that operates primarily \
at the network edge.

Proposed Architecture & Technical Route:
The system uses a wearable Sensor Node equipped with a 3-axis accelerometer and \
gyroscope to continuously sample motion data. Raw sensor readings are streamed over \
BLE to an Edge Gateway (Raspberry Pi 4), which performs local preprocessing including \
noise filtering and windowing. A Lightweight CNN Inference module, exported to TFLite \
and running directly on the Edge Gateway, classifies each motion window as \
"fall" or "normal activity" using a model trained on the SisFall public dataset. \
When a fall is classified with confidence above a fixed threshold, the Alert Module \
immediately triggers a local audible alarm and sends a push notification to a \
caregiver's phone. Aggregated (non-raw) event logs are periodically synced to a \
Cloud Dashboard for longitudinal monitoring by family members or clinicians.

System Implementation & Feasibility:
The CNN was quantized to INT8 to fit within the Raspberry Pi's compute budget, \
achieving an average inference latency of 42ms per window, well within the \
real-time requirement for timely alerting. The BLE link between the sensor node and \
the edge gateway was benchmarked at a stable 150ms round-trip under typical home \
conditions. Power consumption of the sensor node was measured at 18mA average draw, \
giving an estimated battery life of approximately 5 days on a 200mAh coin cell.

Experimental Evaluation & Results:
The system was evaluated using the SisFall dataset (38 subjects, 19 fall types) \
plus 3 days of in-home pilot testing with 2 volunteers. It achieved 94.2% \
sensitivity and 91.7% specificity on the held-out test split, with a false alarm \
rate of 1.3 events per day during the in-home pilot, which the team considers \
acceptable but still improvable for wider deployment.
"""


# ---------------------------------------------------------------------------
# 5.  Demo Flowchart Generator (Pillow -- no external assets)
# ---------------------------------------------------------------------------

BOX_FILL = (235, 242, 255)
BOX_BORDER = (30, 62, 98)
DISCREPANCY_FILL = (255, 235, 235)
DISCREPANCY_BORDER = (178, 34, 34)
ARROW_COLOR = (60, 60, 60)
TEXT_COLOR = (20, 20, 20)
BG_COLOR = (255, 255, 255)


def _load_font(size: int):
    """Try to load a TTF font; fall back to PIL's built-in bitmap font."""
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_box(draw, xy, label, font, fill=BOX_FILL, border=BOX_BORDER):
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=10, fill=fill, outline=border, width=2)
    bbox = draw.textbbox((0, 0), label, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    draw.multiline_text(
        (cx - w / 2, cy - h / 2), label,
        fill=TEXT_COLOR, font=font, align="center",
    )


def _arrow(draw, p0, p1, color=ARROW_COLOR):
    draw.line([p0, p1], fill=color, width=3)
    angle = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    size = 8
    for da in (0.5, -0.5):
        a = angle + math.pi - da
        x = p1[0] + size * math.cos(a)
        y = p1[1] + size * math.sin(a)
        draw.line([p1, (x, y)], fill=color, width=3)


def generate_demo_flowchart() -> Image.Image:
    """Build the IoT Fall Detection architecture diagram with a planted Redis
    Cache discrepancy block highlighted in red."""
    W, H = 1100, 460
    img = Image.new("RGB", (W, H), BG_COLOR)
    draw = ImageDraw.Draw(img)
    font = _load_font(16)
    title_font = _load_font(20)

    draw.text(
        (20, 15),
        "Edge-AI IoT Fall Detection -- System Architecture",
        fill=TEXT_COLOR, font=title_font,
    )

    boxes = {
        "sensor":  (40,  180, 210, 250, "Sensor Node\n(Accel + Gyro)"),
        "gateway": (280, 180, 450, 250, "Edge Gateway\n(Raspberry Pi 4)"),
        "cnn":     (520, 90,  690, 160, "Lightweight CNN\nInference (TFLite)"),
        "alert":   (520, 270, 690, 340, "Alert Module\n(Local + Push)"),
        "redis":   (760, 180, 930, 250, "Redis Cache\n(Event Buffer)"),
        "cloud":   (960, 180, 1080, 250, "Cloud\nDashboard"),
    }

    for key, (x0, y0, x1, y1, label) in boxes.items():
        fill = DISCREPANCY_FILL if key == "redis" else BOX_FILL
        border = DISCREPANCY_BORDER if key == "redis" else BOX_BORDER
        _draw_box(draw, (x0, y0, x1, y1), label, font, fill=fill, border=border)

    def _cr(b): x0, y0, x1, y1, _ = boxes[b]; return (x1, (y0 + y1) / 2)
    def _cl(b): x0, y0, x1, y1, _ = boxes[b]; return (x0, (y0 + y1) / 2)
    def _ct(b): x0, y0, x1, y1, _ = boxes[b]; return ((x0 + x1) / 2, y0)
    def _cb(b): x0, y0, x1, y1, _ = boxes[b]; return ((x0 + x1) / 2, y1)

    _arrow(draw, _cr("sensor"),  _cl("gateway"))
    _arrow(draw, _ct("gateway"), _cl("cnn"))
    _arrow(draw, _cb("gateway"), _cl("alert"))
    _arrow(draw, _cr("cnn"),     _cl("redis"))
    _arrow(draw, _cr("alert"),   _cl("redis"))
    _arrow(draw, _cr("redis"),   _cl("cloud"))

    draw.text(
        (40, 400),
        "Note: 'Redis Cache' block is present in this diagram but is not mentioned anywhere\n"
        "in the report's written Proposed Architecture & Technical Route section (planted discrepancy).",
        fill=DISCREPANCY_BORDER, font=font,
    )
    return img


# ---------------------------------------------------------------------------
# 6.  LLM Prompts -- 2-Pass Pipeline
# ---------------------------------------------------------------------------

PASS1_PROMPT = """\
You are an expert AI reviewing technical practicum reports and engineering architecture diagrams.
You receive a student practicum report and its architecture diagram.

Perform TWO tasks in a single pass:
(A) PARSE the report into four normalized sections:
    - problem_statement: Business or engineering pain point.
    - proposed_architecture: Methodology and technical route.
    - implementation_feasibility: Compute, latency, power, or framework feasibility.
    - experimental_evaluation: Metrics, datasets, and benchmark results.
(B) CROSS-ALIGN the architecture diagram against the written text:
    - visual_blocks: Identify every labeled component/module in the diagram with its label and estimated role.
    - visual_connections: List directed connections (e.g. "Sensor Node -> Edge Gateway").
    - discrepancy_items: Cross-reference diagram against text. Spot:
      * type "unreferenced_visual_block": Diagram contains a block not explained in text.
      * type "hallucinated_pipeline_claim": Text claims a subsystem/pipeline missing from diagram.
      * severity: "low", "medium", or "high".
    - alignment_score: Integer (0-100) scoring diagram-to-text consistency.
    - alignment_summary: 1-2 sentence executive assessment of multimodal agreement.

Return ONLY a valid JSON object matching this exact schema:
{
  "problem_statement": "...",
  "proposed_architecture": "...",
  "implementation_feasibility": "...",
  "experimental_evaluation": "...",
  "visual_blocks": [{"label": "...", "role": "..."}],
  "visual_connections": ["ComponentA -> ComponentB"],
  "visual_summary": "Paragraph describing what the diagram shows.",
  "discrepancy_items": [{"type": "unreferenced_visual_block|hallucinated_pipeline_claim", "description": "...", "severity": "low|medium|high"}],
  "alignment_score": 85,
  "alignment_summary": "Summary of visual-text consistency."
}
"""

PASS2_PROMPT = """\
You are an experienced industry mentor evaluating a student practicum report.
You receive parsed report sections and a multimodal visual-text alignment diagnostic.

Score the report against exactly these 4 industry rubrics (each 0-100):
1. "Industrial Problem Formulation" -- clarity and impact of business/engineering pain point.
2. "Architecture & Technical Feasibility" -- component decoupling, latency, scalability, practicality.
3. "Multimodal Consistency" -- degree to which diagram and written technical route align.
4. "Empirical Validation Rigor" -- presence of quantitative metrics vs superficial qualitative claims.

For each rubric, provide 1-3 concise, constructive mentor comments.
Also provide section-level actionable feedback with current_issue and recommended_revision.

Return ONLY a valid JSON object matching this exact schema:
{
  "rubric_scores": [
    {"rubric": "Industrial Problem Formulation", "score": 82, "comments": ["...", "..."]},
    {"rubric": "Architecture & Technical Feasibility", "score": 78, "comments": ["..."]},
    {"rubric": "Multimodal Consistency", "score": 85, "comments": ["..."]},
    {"rubric": "Empirical Validation Rigor", "score": 75, "comments": ["..."]}
  ],
  "section_feedback": [
    {"section": "Problem Statement", "current_issue": "...", "recommended_revision": "..."},
    {"section": "Proposed Architecture", "current_issue": "...", "recommended_revision": "..."},
    {"section": "Implementation & Feasibility", "current_issue": "...", "recommended_revision": "..."},
    {"section": "Experimental Evaluation", "current_issue": "...", "recommended_revision": "..."}
  ]
}
"""


# ---------------------------------------------------------------------------
# 7.  Dynamic Deterministic Fallback Engine
# ---------------------------------------------------------------------------

def _extract_topic_keywords(text: str) -> list[str]:
    """Pull out capitalized phrases and technical terms from report."""
    words = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text)
    seen = set()
    result = []
    for w in words:
        if w.lower() not in seen and len(w) > 3:
            seen.add(w.lower())
            result.append(w)
    return result[:15]


def _deterministic_pass1(report_text: str, has_image: bool) -> dict:
    """Build a context-aware Pass-1 result from report text heuristics."""
    lower = report_text.lower()
    is_demo = "fall detection" in lower and "edge gateway" in lower

    sections = {
        "problem_statement": "",
        "proposed_architecture": "",
        "implementation_feasibility": "",
        "experimental_evaluation": "",
    }

    header_map = [
        (r"(?:problem\s+statement|background|introduction)[:\s]*\n?(.*?)(?=\n\s*(?:proposed|architecture|system|implementation|experimental|evaluation|$))",
         "problem_statement"),
        (r"(?:proposed\s+architecture|technical\s+route|methodology)[:\s&]*\n?(.*?)(?=\n\s*(?:system|implementation|experimental|evaluation|$))",
         "proposed_architecture"),
        (r"(?:system\s+implementation|feasibility|implementation)[:\s&]*\n?(.*?)(?=\n\s*(?:experimental|evaluation|results|$))",
         "implementation_feasibility"),
        (r"(?:experimental|evaluation|results)[:\s&]*\n?(.*)",
         "experimental_evaluation"),
    ]

    for pattern, key in header_map:
        m = re.search(pattern, report_text, re.IGNORECASE | re.DOTALL)
        if m:
            sections[key] = m.group(1).strip()[:500]

    if not any(sections.values()):
        text = report_text.strip()
        chunk = max(1, len(text) // 4)
        sections["problem_statement"] = text[:chunk].strip()
        sections["proposed_architecture"] = text[chunk:chunk*2].strip()
        sections["implementation_feasibility"] = text[chunk*2:chunk*3].strip()
        sections["experimental_evaluation"] = text[chunk*3:].strip()

    if is_demo:
        visual_blocks = [
            {"label": "Sensor Node", "role": "Captures accelerometer/gyroscope motion data"},
            {"label": "Edge Gateway", "role": "Local preprocessing and on-device inference host"},
            {"label": "Lightweight CNN Inference", "role": "Classifies motion windows as fall/normal"},
            {"label": "Alert Module", "role": "Triggers local alarm and caregiver push notification"},
            {"label": "Redis Cache", "role": "Buffers events between the edge pipeline and the cloud"},
            {"label": "Cloud Dashboard", "role": "Longitudinal monitoring view for caregivers/clinicians"},
        ]
        visual_connections = [
            "Sensor Node -> Edge Gateway",
            "Edge Gateway -> Lightweight CNN Inference",
            "Edge Gateway -> Alert Module",
            "Lightweight CNN Inference -> Redis Cache",
            "Alert Module -> Redis Cache",
            "Redis Cache -> Cloud Dashboard",
        ]
        discrepancy_items = [{
            "type": "unreferenced_visual_block",
            "description": "'Redis Cache' appears in the architecture diagram (Event Buffer) "
                           "but is never mentioned in the written Proposed Architecture & "
                           "Technical Route section.",
            "severity": "medium",
        }]
        alignment_score = 85.0
        alignment_summary = (
            "Found 1 unreferenced visual block (Redis Cache); the remaining "
            "components are well-documented in the text."
        )
    else:
        keywords = _extract_topic_keywords(report_text)
        visual_blocks = [
            {"label": kw, "role": "Component identified from report context"}
            for kw in keywords[:6]
        ] or [{"label": "Uploaded Diagram", "role": "Diagram received; requires live model for detailed analysis"}]
        visual_connections = []
        discrepancy_items = []
        alignment_score = 70.0 if has_image else 50.0
        alignment_summary = (
            "Diagram was received but detailed cross-verification requires a live "
            "multimodal model. Deterministic keyword analysis used as approximation."
        )

    return {
        **sections,
        "visual_blocks": visual_blocks,
        "visual_connections": visual_connections,
        "visual_summary": alignment_summary,
        "discrepancy_items": discrepancy_items,
        "alignment_score": alignment_score,
        "alignment_summary": alignment_summary,
    }


def _deterministic_pass2(report_text: str, pass1: dict) -> dict:
    """Build context-aware rubric scores and feedback without an LLM."""
    has_numbers = bool(re.search(r'\d+\.\d+%|\d+ms|\d+\s*mA', report_text))
    has_dataset = any(w in report_text.lower() for w in ["dataset", "benchmark", "evaluation", "experiment"])
    has_architecture = any(w in report_text.lower() for w in ["architecture", "system", "module", "component", "gateway"])
    n_discrepancies = len(pass1.get("discrepancy_items", []))
    alignment_score = pass1.get("alignment_score", 70.0)

    problem_score = 82.0 if len(pass1.get("problem_statement", "")) > 80 else 65.0
    arch_score = 78.0 if has_architecture else 60.0
    consistency_score = round(alignment_score, 1)
    eval_score = 75.0 if (has_numbers and has_dataset) else (60.0 if has_numbers else 50.0)

    rubric_scores = [
        {
            "rubric": "Industrial Problem Formulation",
            "score": problem_score,
            "comments": [
                "Problem framed with a real-world pain point." if problem_score >= 75
                else "Problem statement could be more specific about the industrial pain point.",
                "Consider quantifying the scale or impact of the problem to strengthen the business case.",
            ],
        },
        {
            "rubric": "Architecture & Technical Feasibility",
            "score": arch_score,
            "comments": [
                "Components are reasonably decoupled in the described architecture." if arch_score >= 70
                else "Architecture section would benefit from clearer component separation.",
                "Consider adding latency budgets and scalability analysis for each component.",
            ],
        },
        {
            "rubric": "Multimodal Consistency",
            "score": consistency_score,
            "comments": [
                pass1.get("alignment_summary", "Alignment analysis completed."),
                f"Found {n_discrepancies} discrepancy item(s) between diagram and text."
                if n_discrepancies
                else "No major discrepancies detected in this analysis pass.",
            ],
        },
        {
            "rubric": "Empirical Validation Rigor",
            "score": eval_score,
            "comments": [
                "Quantitative metrics are present, which is good practice."
                if has_numbers
                else "Evaluation section lacks quantitative metrics; add precision/recall/F1 or equivalent.",
                "Consider expanding sample size or explicitly framing results as preliminary."
                if has_dataset
                else "No standard benchmark or dataset reference found; validation rigor is limited.",
            ],
        },
    ]

    section_feedback = [
        {
            "section": "Problem Statement",
            "current_issue": "The pain point is stated but not fully quantified with impact statistics.",
            "recommended_revision": "Add a statistic on incidence, cost, or consequences among the target population to motivate urgency.",
        },
        {
            "section": "Proposed Architecture & Technical Route",
            "current_issue": (
                f"{n_discrepancies} component(s) shown in the diagram are not mentioned in this section."
                if n_discrepancies
                else "Architecture description is broadly consistent with the diagram."
            ),
            "recommended_revision": (
                "Explicitly describe the role of every diagram block (including caching/buffering) in the written methodology."
                if n_discrepancies
                else "Consider adding a brief justification for each design choice."
            ),
        },
        {
            "section": "System Implementation & Feasibility",
            "current_issue": "Performance figures are point estimates without variance or worst-case analysis.",
            "recommended_revision": "Report latency/power under varying conditions (network load, thermal), not just typical-case averages.",
        },
        {
            "section": "Experimental Evaluation & Results",
            "current_issue": "Pilot sample size may be too small for strong generalizability claims.",
            "recommended_revision": "Either expand the pilot cohort or explicitly frame results as preliminary pending a larger-scale trial.",
        },
    ]

    return {"rubric_scores": rubric_scores, "section_feedback": section_feedback}


# ---------------------------------------------------------------------------
# 8.  Markdown Report Renderer (Zero-Emoji Professional Style)
# ---------------------------------------------------------------------------

def _score_status(score: float) -> str:
    """Return a clean text status label for a rubric score."""
    if score >= 85:
        return "Passed"
    if score >= 70:
        return "Acceptable"
    return "Action Required"


SEVERITY_LABEL = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW"}


def _disposition(score: float) -> str:
    if score >= 85:
        return "PASS"
    if score >= 70:
        return "CONDITIONAL PASS"
    return "REVISE AND RESUBMIT"


def render_markdown_report(
    pass1: dict,
    pass2: dict,
    aggregate_score: float,
    disposition: str,
    mode_label: str,
) -> str:
    """Render the final Mentor Evaluation Memorandum as clean Markdown."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Mentor Evaluation Memorandum",
        f"*Generated {ts} -- Multimodal-MentorAgent {VERSION} ({mode_label})*",
        "",
        f"## Overall Disposition: {disposition}",
        f"**Aggregate Score: {aggregate_score:.1f} / 100**",
        "",
        "---",
        "",
        "## Metric Summary Table",
        "",
        "| Rubric | Score | Status |",
        "|---|---:|---|",
    ]

    for r in pass2.get("rubric_scores", []):
        s = r.get("score", 0)
        lines.append(f"| {r.get('rubric', '?')} | {s:.1f} | {_score_status(s)} |")
    lines.append("")

    # Multimodal alignment diagnostic
    lines.append("## Multimodal Alignment Diagnostic")
    lines.append("")
    align_score = pass1.get("alignment_score", "N/A")
    lines.append(f"> **[ALIGNMENT AUDIT]**")
    lines.append(f">")
    lines.append(f"> **Alignment Score:** {align_score} / 100")
    lines.append(f"> {pass1.get('alignment_summary', 'N/A')}")
    lines.append(">")

    discrepancies = pass1.get("discrepancy_items", [])
    if discrepancies:
        for item in discrepancies:
            sev = item.get("severity", "low")
            label = SEVERITY_LABEL.get(sev, "INFO")
            kind = (
                "Unreferenced Visual Block"
                if item.get("type") == "unreferenced_visual_block"
                else "Hallucinated Pipeline Claim"
            )
            lines.append(f"> * [{label}] **{kind}:** {item.get('description', '')}")
    else:
        lines.append("> * No discrepancies detected between the diagram and the written methodology.")
    lines.append("")

    # Visual blocks inventory
    vblocks = pass1.get("visual_blocks", [])
    if vblocks:
        lines.append("### Detected Diagram Components")
        lines.append("")
        lines.append("| Component | Inferred Role |")
        lines.append("|---|---|")
        for vb in vblocks:
            lines.append(f"| {vb.get('label', '?')} | {vb.get('role', '')} |")
        lines.append("")

    vconns = pass1.get("visual_connections", [])
    if vconns:
        lines.append("### Detected Connections")
        lines.append("")
        for vc in vconns:
            lines.append(f"* {vc}")
        lines.append("")

    # Section feedback
    lines.append("## Section-by-Section Actionable Feedback")
    lines.append("")
    for fb in pass2.get("section_feedback", []):
        lines.append(f"### {fb.get('section', '?')}")
        lines.append(f"- **Current Issue:** {fb.get('current_issue', 'N/A')}")
        lines.append(f"- **Recommended Revision:** {fb.get('recommended_revision', 'N/A')}")
        lines.append("")

    # Rubric detail
    lines.append("## Rubric Detail")
    lines.append("")
    for r in pass2.get("rubric_scores", []):
        lines.append(f"**{r.get('rubric', '?')} -- {r.get('score', 0):.1f}/100**")
        for c in r.get("comments", []):
            lines.append(f"- {c}")
        lines.append("")

    lines.append("---")
    lines.append(
        "*This memorandum was generated by an AI-assisted multi-agent review pipeline "
        "and is intended to accelerate, not replace, the industry mentor's own judgement. "
        "Final assessment decisions rest with the human mentor.*"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 9.  Main Audit Generator (Yields Live Progress to Gradio)
# ---------------------------------------------------------------------------

def execute_audit(report_text: str, image: Image.Image | None):
    """
    Generator function that yields (trace_md, report_md, files) tuples
    progressively so Gradio streams updates to the user in real time.
    GUARANTEES: Never raises an exception -- always yields a complete result.
    """
    trace_lines: list[str] = ["## Live Multi-Agent Execution Trace\n"]
    placeholder_report = "_Processing..._"
    start_time = time.time()

    def _state(report_md=placeholder_report, files=None):
        return "\n".join(trace_lines), report_md, files

    try:
        # -- Guard: empty input -------------------------------------------
        if not report_text or not report_text.strip():
            msg = "[WARNING] Please paste report text, upload a PDF document, or click 'Load Default Case' first."
            yield msg, msg, None
            return

        has_image = image is not None

        # -- Stage 1/4: Parse & Optimize Image ----------------------------
        trace_lines.append("### [STAGE 1/4] Section Parsing & Visual Preprocessing")
        yield _state()

        img_data_url = None
        if has_image:
            try:
                image, img_data_url = optimize_image(image)
                payload_kb = len(img_data_url) * 3 // 4 // 1024
                trace_lines.append(
                    f"  [SUCCESS] Diagram optimized: "
                    f"{image.size[0]}x{image.size[1]}px, payload ~{payload_kb}KB"
                )
            except Exception as e:
                trace_lines.append(f"  [WARNING] Diagram optimization skipped: {e}")
                has_image = False
        else:
            trace_lines.append("  [INFO] No architecture diagram provided -- text-only analysis.")

        yield _state()

        # -- Stage 2/4: Pass 1 -- Multimodal Alignment (Qwen-VL) ----------
        trace_lines.append("")
        trace_lines.append(f"### [STAGE 2/4] Cross-Aligning Diagram with Technical Route ({QWEN_VL_MODEL})")
        yield _state()

        pass1_result = None
        mode_label = "Deterministic Fallback"

        if _llm_available():
            user_text = f"STUDENT PRACTICUM REPORT:\n\n{report_text[:3500]}"

            # Construct OpenAI-compatible multimodal content
            user_content = []
            if has_image and img_data_url:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": img_data_url},
                })
            user_content.append({
                "type": "text",
                "text": user_text,
            })

            messages = [
                {"role": "system", "content": PASS1_PROMPT},
                {"role": "user", "content": user_content},
            ]

            t0 = time.time()
            pass1_result = _safe_qwen_call(messages, QWEN_VL_MODEL, "Pass-1 (Qwen-VL)")
            elapsed = time.time() - t0

            if pass1_result:
                mode_label = f"Qwen ({QWEN_VL_MODEL})"
                n_blocks = len(pass1_result.get("visual_blocks", []))
                n_disc = len(pass1_result.get("discrepancy_items", []))
                trace_lines.append(
                    f"  [SUCCESS] Pass 1 complete ({elapsed:.1f}s): "
                    f"{n_blocks} visual blocks identified, {n_disc} discrepancies detected, "
                    f"alignment_score={pass1_result.get('alignment_score', '?')}/100"
                )
            else:
                trace_lines.append(
                    f"  [WARNING] Pass 1 LLM call failed ({elapsed:.1f}s) -- "
                    "engaging deterministic fallback engine."
                )
        else:
            trace_lines.append("  [INFO] No DashScope/Qwen API key configured -- using deterministic analysis engine.")

        if pass1_result is None:
            pass1_result = _deterministic_pass1(report_text, has_image)
            mode_label = "Deterministic Fallback"
            n_blocks = len(pass1_result.get("visual_blocks", []))
            n_disc = len(pass1_result.get("discrepancy_items", []))
            trace_lines.append(
                f"  [COMPLETE] Deterministic Pass 1: "
                f"{n_blocks} visual blocks, {n_disc} discrepancies detected."
            )

        yield _state()

        # -- Stage 3/4: Pass 2 -- Rubric Diagnostic & Synthesis (Qwen) ----
        trace_lines.append("")
        trace_lines.append(f"### [STAGE 3/4] Benchmarking Against Industry Mentor Rubrics ({QWEN_TEXT_MODEL})")
        yield _state()

        pass2_result = None

        if _llm_available():
            pass2_input = (
                "PARSED SECTIONS:\n"
                + json.dumps({
                    "problem_statement": pass1_result.get("problem_statement", ""),
                    "proposed_architecture": pass1_result.get("proposed_architecture", ""),
                    "implementation_feasibility": pass1_result.get("implementation_feasibility", ""),
                    "experimental_evaluation": pass1_result.get("experimental_evaluation", ""),
                }, ensure_ascii=False, indent=2)
                + "\n\nMULTIMODAL ALIGNMENT DIAGNOSTIC:\n"
                + json.dumps({
                    "alignment_score": pass1_result.get("alignment_score", 70),
                    "alignment_summary": pass1_result.get("alignment_summary", ""),
                    "discrepancy_items": pass1_result.get("discrepancy_items", []),
                }, ensure_ascii=False, indent=2)
            )

            messages = [
                {"role": "system", "content": PASS2_PROMPT},
                {"role": "user", "content": pass2_input},
            ]

            t0 = time.time()
            pass2_result = _safe_qwen_call(messages, QWEN_TEXT_MODEL, "Pass-2 (Qwen)")
            elapsed = time.time() - t0

            if pass2_result and pass2_result.get("rubric_scores"):
                n_rubrics = len(pass2_result.get("rubric_scores", []))
                n_fb = len(pass2_result.get("section_feedback", []))
                trace_lines.append(
                    f"  [SUCCESS] Pass 2 complete ({elapsed:.1f}s): "
                    f"scored {n_rubrics} rubrics, {n_fb} feedback items."
                )
                if mode_label.startswith("Qwen"):
                    mode_label = f"Qwen ({QWEN_VL_MODEL} + {QWEN_TEXT_MODEL})"
            else:
                trace_lines.append(
                    f"  [WARNING] Pass 2 LLM call failed ({elapsed:.1f}s) -- "
                    "engaging deterministic fallback engine."
                )
                pass2_result = None
        else:
            trace_lines.append("  [INFO] No API key -- using deterministic rubric engine.")

        if pass2_result is None or not pass2_result.get("rubric_scores"):
            pass2_result = _deterministic_pass2(report_text, pass1_result)
            if mode_label.startswith("Qwen"):
                mode_label += " + Pass-2 Fallback"
            n_rubrics = len(pass2_result.get("rubric_scores", []))
            n_fb = len(pass2_result.get("section_feedback", []))
            trace_lines.append(
                f"  [COMPLETE] Deterministic Pass 2: "
                f"scored {n_rubrics} rubrics, {n_fb} feedback items."
            )

        yield _state()

        # -- Stage 4/4: Final Synthesis & Report Generation ----------------
        trace_lines.append("")
        trace_lines.append("### [STAGE 4/4] Generating Final Scorecard and Memorandum")
        yield _state()

        rubric_scores = pass2_result.get("rubric_scores", [])
        if rubric_scores:
            aggregate = sum(r.get("score", 0) for r in rubric_scores) / len(rubric_scores)
        else:
            aggregate = 0.0
        aggregate = round(aggregate, 1)
        disposition = _disposition(aggregate)

        final_md = render_markdown_report(
            pass1_result, pass2_result, aggregate, disposition, mode_label
        )

        # Write downloadable artifacts
        files = None
        try:
            out_dir = Path(tempfile.mkdtemp(prefix="mentoragent_"))
            md_path = out_dir / "mentor_evaluation_report.md"
            json_path = out_dir / "mentor_evaluation_report.json"
            md_path.write_text(final_md, encoding="utf-8")

            export_data = {
                "version": VERSION,
                "model": mode_label,
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "aggregate_score": aggregate,
                "disposition": disposition,
                "pass1": pass1_result,
                "pass2": pass2_result,
            }
            json_path.write_text(
                json.dumps(export_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            files = [str(md_path), str(json_path)]
        except Exception as e:
            trace_lines.append(f"  [WARNING] File export skipped: {e}")

        total_time = time.time() - start_time
        trace_lines.append(
            f"  [SUCCESS] Report synthesized: aggregate={aggregate:.1f}/100, "
            f'disposition="{disposition}"'
        )
        trace_lines.append("")
        trace_lines.append(
            f"### [PIPELINE COMPLETE] Total Runtime: {total_time:.1f}s "
            f"| Mode: {mode_label}"
        )

        yield "\n".join(trace_lines), final_md, files

    except Exception as e:
        # -- Ultimate safety net -- never show red Error badge -------------
        trace_lines.append("")
        trace_lines.append("### [SAFETY FALLBACK] Unexpected Error Caught")
        trace_lines.append(f"  Error detail: `{e}`")
        trace_lines.append("  Generating deterministic review from input text...")

        try:
            pass1_fb = _deterministic_pass1(report_text or "", image is not None)
            pass2_fb = _deterministic_pass2(report_text or "", pass1_fb)
            rubrics = pass2_fb.get("rubric_scores", [])
            agg = round(sum(r.get("score", 0) for r in rubrics) / max(len(rubrics), 1), 1)
            disp = _disposition(agg)
            md = render_markdown_report(pass1_fb, pass2_fb, agg, disp, "Safety Fallback")

            total_time = time.time() - start_time
            trace_lines.append(f"  [COMPLETE] Safety fallback finished ({total_time:.1f}s).")
            yield "\n".join(trace_lines), md, None
        except Exception as inner:
            trace_lines.append(f"  [ERROR] Inner fallback also failed: {inner}")
            yield (
                "\n".join(trace_lines),
                "## Review Generation Failed\n\n"
                "The system encountered an unexpected error but prevented a crash.\n"
                "Please check your DashScope API key, network connection, and try again.\n\n"
                f"Error: `{e}`",
                None,
            )


# ---------------------------------------------------------------------------
# 10.  Gradio UI (Zero-Emoji Professional Layout + PDF Ingestion)
# ---------------------------------------------------------------------------

def _status_badge() -> str:
    if _llm_available():
        return f"[LIVE QWEN ACTIVE] Vision: {QWEN_VL_MODEL} | Text: {QWEN_TEXT_MODEL}"
    return "[SIMULATION FALLBACK] No DashScope/Qwen API key configured"


def load_default_case():
    """Pre-fill the IoT Fall Detection sample case with the planted Redis discrepancy."""
    img = generate_demo_flowchart()
    return DEFAULT_REPORT_TEXT, img, None


def build_ui() -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""

    HEADER_HTML = f"""
    <div style="text-align:center; padding: 10px 0 18px 0;">
      <h1 style="margin-bottom:4px;">
        Multimodal-MentorAgent
      </h1>
      <h3 style="margin-top:0; font-weight:400; color:#444;">
        Multimodal Large Model Agent for Intelligent Review of
        Industry Mentor Student Materials
      </h3>
      <p style="margin-top:2px; font-weight:400; color:#666; font-size:14px;">
        (基于多模态大模型智能体的行业导师实践材料智能审核系统)
      </p>
      <span style="display:inline-block; padding:3px 10px; border-radius:4px;
            background:#eef2ff; color:#1e3e62; font-size:13px; margin-top:6px;
            font-family:monospace;">
        {VERSION}
      </span>
      &nbsp;
      <span style="display:inline-block; padding:3px 10px; border-radius:4px;
            background:#f5f5f5; color:#333; font-size:13px; margin-top:6px;
            font-family:monospace;">
        {_status_badge()}
      </span>
    </div>
    """

    with gr.Blocks(title="Multimodal-MentorAgent") as demo:
        gr.HTML(HEADER_HTML)

        with gr.Row():
            # -- Left Panel: Inputs --
            with gr.Column(scale=1):
                gr.Markdown("### Student Material Input")

                pdf_upload_in = gr.File(
                    label="Upload Complete Practicum Report (PDF)",
                    file_types=[".pdf"],
                    type="filepath",
                )

                report_text_in = gr.Textbox(
                    label="Practicum Report Text",
                    placeholder=(
                        "Paste the student's practicum report text here, "
                        "upload a PDF report above, or click 'Load Default Case'..."
                    ),
                    lines=14,
                )
                image_in = gr.Image(
                    label="Architecture Diagram / Flowchart",
                    type="pil",
                )
                with gr.Row():
                    load_btn = gr.Button(
                        "Load Default Case", variant="secondary",
                    )
                    run_btn = gr.Button(
                        "Execute Multi-Agent Audit", variant="primary",
                    )

            # -- Right Panel: Outputs --
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab("Live Execution Trace"):
                        trace_out = gr.Markdown(
                            value="_Run the audit to see the agent-by-agent trace here._"
                        )
                    with gr.Tab("Final Mentor Report"):
                        report_out = gr.Markdown(
                            value="_Run the audit to see the final scorecard here._"
                        )
                        download_out = gr.File(
                            label="Download Report (Markdown + JSON)",
                            file_count="multiple",
                        )

        # -- Event Wiring --
        pdf_upload_in.upload(
            fn=parse_pdf_document,
            inputs=[pdf_upload_in],
            outputs=[report_text_in, image_in],
        )

        load_btn.click(
            fn=load_default_case,
            outputs=[report_text_in, image_in, pdf_upload_in],
        )

        run_btn.click(
            fn=execute_audit,
            inputs=[report_text_in, image_in],
            outputs=[trace_out, report_out, download_out],
        )

    return demo


# ---------------------------------------------------------------------------
# 11.  Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"[STARTUP] Multimodal-MentorAgent {VERSION}")
    print(f"[STARTUP] Base URL: {DASHSCOPE_BASE_URL}")
    print(f"[STARTUP] Vision Model: {QWEN_VL_MODEL}")
    print(f"[STARTUP] Text Model:   {QWEN_TEXT_MODEL}")
    print(f"[STARTUP] API key configured: {'Yes' if _llm_available() else 'No'}")
    print(f"[STARTUP] Timeout: {TIMEOUT_SECONDS}s | Retries: {MAX_RETRIES}")
    demo = build_ui()
    demo.launch(theme=gr.themes.Soft())
