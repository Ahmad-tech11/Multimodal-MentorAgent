"""
Pydantic schemas for structured message routing between agents.
Every agent hand-off in the pipeline is a validated schema, not free text,
so downstream agents can rely on field names rather than re-parsing prose.
"""

from __future__ import annotations
from typing import List, Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Agent 1: Document & Visual Parser
# ---------------------------------------------------------------------------

class VisualBlock(BaseModel):
    """A single component/box detected in the architecture diagram."""
    label: str = Field(description="Text label of the component as drawn in the diagram")
    role: str = Field(description="Short description of what this component appears to do")


class ParsedSections(BaseModel):
    """Normalized output of Agent_Parser."""
    problem_statement: str = Field(description="The business/engineering pain point being addressed")
    proposed_architecture: str = Field(description="The written methodology / technical route section")
    implementation_feasibility: str = Field(description="System implementation & feasibility discussion")
    experimental_evaluation: str = Field(description="Experimental evaluation & results section")

    visual_blocks: List[VisualBlock] = Field(
        default_factory=list,
        description="Components identified in the architecture diagram",
    )
    visual_connections: List[str] = Field(
        default_factory=list,
        description="Directed connections between components, e.g. 'Sensor Node -> Edge Gateway'",
    )
    visual_summary: str = Field(description="One-paragraph plain-language description of the diagram")


# ---------------------------------------------------------------------------
# Agent 2: Multimodal Cross-Verification
# ---------------------------------------------------------------------------

class DiscrepancyItem(BaseModel):
    type: Literal["unreferenced_visual_block", "hallucinated_pipeline_claim"] = Field(
        description="Whether the diagram shows something the text omits, or the text claims "
        "something the diagram does not support"
    )
    description: str = Field(description="Plain description of the specific discrepancy")
    severity: Literal["low", "medium", "high"] = Field(description="Impact on technical credibility")


class DiscrepancyMatrix(BaseModel):
    items: List[DiscrepancyItem] = Field(default_factory=list)
    alignment_score: float = Field(
        ge=0, le=100, description="0-100 score of how well the text and diagram agree"
    )
    summary: str = Field(description="One or two sentence overview of multimodal consistency")


# ---------------------------------------------------------------------------
# Agent 3: Rubric & Diagnostic Auditor
# ---------------------------------------------------------------------------

class SectionFeedback(BaseModel):
    section: str = Field(description="Which section this feedback targets")
    current_issue: str = Field(description="What is currently wrong or weak")
    recommended_revision: str = Field(description="Concrete instruction on how to fix it")


class RubricScore(BaseModel):
    rubric: Literal[
        "Industrial Problem Formulation",
        "Architecture & Technical Feasibility",
        "Multimodal Consistency",
        "Empirical Validation Rigor",
    ]
    score: float = Field(ge=0, le=100)
    comments: List[str] = Field(default_factory=list)


class RubricAudit(BaseModel):
    rubric_scores: List[RubricScore]
    section_feedback: List[SectionFeedback]


# ---------------------------------------------------------------------------
# Agent 4: Audit Synthesis & Report Generator
# ---------------------------------------------------------------------------

class AuditReport(BaseModel):
    aggregate_score: float = Field(ge=0, le=100)
    disposition: Literal[
        "Ready for Defense",
        "Conditional Pass with Minor Revisions",
        "Requires Major Structural Revision",
    ]
    rubric_scores: List[RubricScore]
    discrepancy_matrix: DiscrepancyMatrix
    section_feedback: List[SectionFeedback]
    markdown_report: str = Field(description="Fully rendered Markdown evaluation memorandum")


# ---------------------------------------------------------------------------
# Execution trace (for the Gradio "Live Execution Trace" tab)
# ---------------------------------------------------------------------------

class TraceStep(BaseModel):
    agent: str
    thought: str
    action: str
    observation: str
