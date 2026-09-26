"""Agent_VisionAlign — compares diagram content against the written methodology
to surface unreferenced visual blocks and hallucinated pipeline claims."""

from __future__ import annotations

from agents.schemas import ParsedSections, DiscrepancyMatrix, DiscrepancyItem, TraceStep
from agents.llm_client import call_structured, llm_available

SYSTEM_PROMPT = """You are Agent_VisionAlign, the second stage of a multi-agent \
academic review pipeline. You receive the parsed report sections and the parsed \
diagram (visual blocks + connections). Cross-check them: (1) flag any visual block \
present in the diagram that is never referenced in the 'Proposed Architecture & \
Technical Route' text as an unreferenced_visual_block; (2) flag any specific claim \
in that text about a component or connection that has no corresponding block or \
connection in the diagram as a hallucinated_pipeline_claim. Assign an overall \
alignment_score from 0 to 100 reflecting how well the text and diagram agree \
(100 = perfect agreement, no discrepancies)."""


def _fallback(parsed: ParsedSections) -> DiscrepancyMatrix:
    text_blob = parsed.proposed_architecture.lower()
    items = []
    for block in parsed.visual_blocks:
        if block.label.lower() not in text_blob:
            items.append(
                DiscrepancyItem(
                    type="unreferenced_visual_block",
                    description=(
                        f"'{block.label}' appears in the architecture diagram "
                        f"({block.role}) but is never mentioned in the written "
                        "Proposed Architecture & Technical Route section."
                    ),
                    severity="medium",
                )
            )
    alignment_score = max(0.0, 100.0 - 15.0 * len(items))
    summary = (
        f"Found {len(items)} unreferenced visual block(s); no hallucinated text-only "
        "claims detected in this simulated pass."
        if items
        else "Diagram and text appear consistent in this simulated pass."
    )
    return DiscrepancyMatrix(items=items, alignment_score=alignment_score, summary=summary)


def run_agent_vision_align(parsed: ParsedSections) -> tuple[DiscrepancyMatrix, TraceStep]:
    if llm_available():
        user_prompt = (
            "PARSED TEXT SECTIONS:\n"
            f"Proposed Architecture & Technical Route:\n{parsed.proposed_architecture}\n\n"
            "PARSED VISUAL BLOCKS:\n"
            + "\n".join(f"- {b.label}: {b.role}" for b in parsed.visual_blocks)
            + "\n\nPARSED VISUAL CONNECTIONS:\n"
            + "\n".join(f"- {c}" for c in parsed.visual_connections)
        )
        result = call_structured(SYSTEM_PROMPT, user_prompt, DiscrepancyMatrix)
        trace = TraceStep(
            agent="Agent_VisionAlign",
            thought="Cross-check every visual block/connection against the written architecture section for omissions or unsupported claims.",
            action="call_structured(DiscrepancyMatrix) on Gemini with parsed text + visual inventory",
            observation=f"Found {len(result.items)} discrepancy item(s); alignment_score={result.alignment_score:.1f}.",
        )
        return result, trace

    result = _fallback(parsed)
    trace = TraceStep(
        agent="Agent_VisionAlign",
        thought="No GOOGLE_API_KEY found — running deterministic keyword cross-check between visual blocks and architecture text.",
        action="_fallback(parsed)",
        observation=f"Found {len(result.items)} discrepancy item(s); alignment_score={result.alignment_score:.1f}.",
    )
    return result, trace
