"""Agent_Parser — normalizes raw report text + diagram into structured sections."""

from __future__ import annotations
from typing import Optional
from PIL import Image

from agents.schemas import ParsedSections, VisualBlock, TraceStep
from agents.llm_client import call_structured, llm_available

SYSTEM_PROMPT = """You are Agent_Parser, the first stage of a multi-agent academic \
review pipeline. You read a student's technical practicum report together with its \
architecture diagram and normalize both into a structured schema. Extract the four \
required text sections verbatim/summarized from the report, and separately analyze \
the image: list every distinct component/box you see as a visual block with a short \
role description, list the directed connections between them, and write a one \
paragraph plain-language summary of what the diagram shows. Do not invent content \
that is not present in the report or the diagram."""


def _fallback(report_text: str) -> ParsedSections:
    """Deterministic simulation used when no API key is configured. Tuned to the
    bundled default demo scenario, but degrades gracefully to a generic
    placeholder for arbitrary user-supplied text."""
    lower_text = report_text.lower()
    is_default_demo = "fall detection" in lower_text and "edge gateway" in lower_text

    if is_default_demo:
        return ParsedSections(
            problem_statement=(
                "Elderly patients living alone face high risk from unwitnessed falls; "
                "existing camera-based systems raise privacy concerns and require "
                "constant connectivity, motivating a privacy-preserving, edge-based "
                "fall detection system."
            ),
            proposed_architecture=(
                "A wearable Sensor Node streams accelerometer/gyroscope data over BLE "
                "to an Edge Gateway (Raspberry Pi 4), which hosts a Lightweight CNN "
                "Inference module (TFLite, INT8) to classify falls locally, triggering "
                "a local Alert Module and periodically syncing aggregated logs to a "
                "Cloud Dashboard."
            ),
            implementation_feasibility=(
                "INT8-quantized CNN achieves 42ms inference latency on the Raspberry "
                "Pi; BLE link benchmarked at 150ms round-trip; sensor node draws 18mA "
                "average, giving ~5 days battery life on a 200mAh cell."
            ),
            experimental_evaluation=(
                "Evaluated on the SisFall dataset (38 subjects, 19 fall types) plus a "
                "3-day in-home pilot with 2 volunteers: 94.2% sensitivity, 91.7% "
                "specificity, 1.3 false alarms/day."
            ),
            visual_blocks=[
                VisualBlock(label="Sensor Node", role="Captures accelerometer/gyroscope motion data"),
                VisualBlock(label="Edge Gateway", role="Local preprocessing and on-device inference host"),
                VisualBlock(label="Lightweight CNN Inference", role="Classifies motion windows as fall/normal"),
                VisualBlock(label="Alert Module", role="Triggers local alarm and caregiver push notification"),
                VisualBlock(label="Redis Cache", role="Buffers events between the edge pipeline and the cloud"),
                VisualBlock(label="Cloud Dashboard", role="Longitudinal monitoring view for caregivers/clinicians"),
            ],
            visual_connections=[
                "Sensor Node -> Edge Gateway",
                "Edge Gateway -> Lightweight CNN Inference",
                "Edge Gateway -> Alert Module",
                "Lightweight CNN Inference -> Redis Cache",
                "Alert Module -> Redis Cache",
                "Redis Cache -> Cloud Dashboard",
            ],
            visual_summary=(
                "The diagram shows a sensor-to-cloud pipeline: a wearable sensor feeds "
                "an edge gateway that branches into CNN inference and alerting, both of "
                "which route through an intermediate Redis Cache block before reaching "
                "a cloud dashboard."
            ),
        )

    # Generic fallback for arbitrary pasted text when no API key is present.
    snippet = (report_text or "").strip()[:400] or "No report text was provided."
    return ParsedSections(
        problem_statement=f"[Simulated — no API key] Problem statement inferred from input: {snippet[:150]}...",
        proposed_architecture="[Simulated — no API key] Architecture section could not be deeply parsed without a live model.",
        implementation_feasibility="[Simulated — no API key] Feasibility section could not be deeply parsed without a live model.",
        experimental_evaluation="[Simulated — no API key] Evaluation section could not be deeply parsed without a live model.",
        visual_blocks=[VisualBlock(label="Uploaded Diagram", role="Component-level detail requires a live multimodal model")],
        visual_connections=[],
        visual_summary="[Simulated — no API key] Diagram was received but not analyzed in detail without a live model.",
    )


def run_agent_parser(report_text: str, image: Optional[Image.Image]) -> tuple[ParsedSections, TraceStep]:
    if llm_available():
        user_prompt = f"STUDENT REPORT TEXT:\n\n{report_text}\n\nAnalyze the attached architecture diagram alongside this text."
        result = call_structured(SYSTEM_PROMPT, user_prompt, ParsedSections, image=image)
        trace = TraceStep(
            agent="Agent_Parser",
            thought="Normalize raw report text and architecture diagram into the four-section schema plus visual block/connection inventory.",
            action="call_structured(ParsedSections) on Gemini with report text + diagram image",
            observation=f"Extracted {len(result.visual_blocks)} visual blocks, {len(result.visual_connections)} connections.",
        )
        return result, trace

    result = _fallback(report_text)
    trace = TraceStep(
        agent="Agent_Parser",
        thought="No GOOGLE_API_KEY found — running deterministic fallback parser tuned to the default demo scenario.",
        action="_fallback(report_text)",
        observation=f"Simulated {len(result.visual_blocks)} visual blocks, {len(result.visual_connections)} connections.",
    )
    return result, trace
