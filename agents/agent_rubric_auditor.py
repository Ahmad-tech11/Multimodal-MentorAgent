"""Agent_RubricAuditor — benchmarks parsed content + discrepancies against
four industry rubrics and produces section-anchored feedback."""

from __future__ import annotations

from agents.schemas import (
    ParsedSections,
    DiscrepancyMatrix,
    RubricAudit,
    RubricScore,
    SectionFeedback,
    TraceStep,
)
from agents.llm_client import call_structured, llm_available

SYSTEM_PROMPT = """You are Agent_RubricAuditor, the third stage of a multi-agent \
academic review pipeline, acting with the judgement of an experienced industry \
mentor. Score the report against exactly these four rubrics, each 0-100: \
'Industrial Problem Formulation' (clarity of the business/engineering pain point), \
'Architecture & Technical Feasibility' (component decoupling, latency, scalability), \
'Multimodal Consistency' (use the provided alignment_score as strong evidence for \
this one), 'Empirical Validation Rigor' (quantitative metrics vs superficial \
qualitative claims). For each rubric include 1-3 short comments. Also produce \
paragraph-level, section-anchored feedback (current_issue + recommended_revision) \
for each of the four report sections. Be specific and critical, not generic praise."""


def _fallback(parsed: ParsedSections, discrepancy: DiscrepancyMatrix) -> RubricAudit:
    has_numbers = any(ch.isdigit() for ch in parsed.experimental_evaluation)

    rubric_scores = [
        RubricScore(
            rubric="Industrial Problem Formulation",
            score=82.0,
            comments=[
                "Problem is framed with a clear real-world pain point and a concrete comparison to existing (camera-based) approaches.",
                "Could quantify the scale of the problem (e.g., fall incidence rates) to strengthen the business case.",
            ],
        ),
        RubricScore(
            rubric="Architecture & Technical Feasibility",
            score=78.0,
            comments=[
                "Components are reasonably decoupled (sensor / gateway / inference / alerting / cloud).",
                "Latency and power figures are reported, which is good practice for edge feasibility claims.",
            ],
        ),
        RubricScore(
            rubric="Multimodal Consistency",
            score=round(discrepancy.alignment_score, 1),
            comments=[discrepancy.summary],
        ),
        RubricScore(
            rubric="Empirical Validation Rigor",
            score=75.0 if has_numbers else 55.0,
            comments=[
                "Quantitative metrics (sensitivity/specificity/false-alarm rate) are present, which is good."
                if has_numbers
                else "Evaluation section lacks quantitative metrics.",
                "In-home pilot sample size (2 volunteers, 3 days) is small; results should be qualified as preliminary.",
            ],
        ),
    ]

    section_feedback = [
        SectionFeedback(
            section="Problem Statement",
            current_issue="The pain point is stated but not quantified.",
            recommended_revision="Add a statistic on fall incidence/consequences among the target population to motivate urgency.",
        ),
        SectionFeedback(
            section="Proposed Architecture & Technical Route",
            current_issue=(
                f"{len(discrepancy.items)} component(s) shown in the diagram are not mentioned in this section."
                if discrepancy.items
                else "Architecture description is broadly consistent with the diagram."
            ),
            recommended_revision=(
                "Explicitly describe the role of every diagram block (including buffering/caching components) in the written methodology."
                if discrepancy.items
                else "No structural changes required; consider adding a brief justification for each design choice."
            ),
        ),
        SectionFeedback(
            section="System Implementation & Feasibility",
            current_issue="Latency and power figures are point estimates without variance or worst-case conditions.",
            recommended_revision="Report latency/power under varying network and thermal conditions, not just typical-case averages.",
        ),
        SectionFeedback(
            section="Experimental Evaluation & Results",
            current_issue="In-home pilot is very small (2 volunteers, 3 days), limiting generalizability.",
            recommended_revision="Either expand the pilot cohort or explicitly frame these as preliminary results pending a larger-scale trial.",
        ),
    ]

    return RubricAudit(rubric_scores=rubric_scores, section_feedback=section_feedback)


def run_agent_rubric_auditor(
    parsed: ParsedSections, discrepancy: DiscrepancyMatrix
) -> tuple[RubricAudit, TraceStep]:
    if llm_available():
        user_prompt = (
            "PARSED SECTIONS:\n"
            f"Problem Statement:\n{parsed.problem_statement}\n\n"
            f"Proposed Architecture & Technical Route:\n{parsed.proposed_architecture}\n\n"
            f"System Implementation & Feasibility:\n{parsed.implementation_feasibility}\n\n"
            f"Experimental Evaluation & Results:\n{parsed.experimental_evaluation}\n\n"
            f"MULTIMODAL DISCREPANCY MATRIX (alignment_score={discrepancy.alignment_score}):\n"
            + "\n".join(f"- [{i.type}/{i.severity}] {i.description}" for i in discrepancy.items)
        )
        result = call_structured(SYSTEM_PROMPT, user_prompt, RubricAudit)
        trace = TraceStep(
            agent="Agent_RubricAuditor",
            thought="Benchmark parsed content and multimodal findings against the four industry rubrics; generate section-anchored feedback.",
            action="call_structured(RubricAudit) on Gemini with parsed sections + discrepancy matrix",
            observation=f"Scored {len(result.rubric_scores)} rubrics; produced {len(result.section_feedback)} section feedback item(s).",
        )
        return result, trace

    result = _fallback(parsed, discrepancy)
    trace = TraceStep(
        agent="Agent_RubricAuditor",
        thought="No GOOGLE_API_KEY found — running deterministic rubric scoring tuned to the default demo scenario.",
        action="_fallback(parsed, discrepancy)",
        observation=f"Scored {len(result.rubric_scores)} rubrics; produced {len(result.section_feedback)} section feedback item(s).",
    )
    return result, trace
