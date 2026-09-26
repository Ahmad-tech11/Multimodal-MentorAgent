"""Sequential orchestration of the 4-agent pipeline:
Agent_Parser -> Agent_VisionAlign -> Agent_RubricAuditor -> Agent_Synthesizer

Each stage's output is the next stage's validated input (Pydantic schemas),
and every stage also emits a TraceStep for the Gradio "Live Execution Trace" tab.
"""

from __future__ import annotations
from typing import Optional, List
from PIL import Image

from agents.schemas import AuditReport, TraceStep
from agents.agent_parser import run_agent_parser
from agents.agent_vision_align import run_agent_vision_align
from agents.agent_rubric_auditor import run_agent_rubric_auditor
from agents.agent_synthesizer import run_agent_synthesizer


def run_pipeline(report_text: str, image: Optional[Image.Image]) -> tuple[AuditReport, List[TraceStep]]:
    trace: List[TraceStep] = []

    parsed, t1 = run_agent_parser(report_text, image)
    trace.append(t1)

    discrepancy, t2 = run_agent_vision_align(parsed)
    trace.append(t2)

    audit, t3 = run_agent_rubric_auditor(parsed, discrepancy)
    trace.append(t3)

    report, t4 = run_agent_synthesizer(parsed, discrepancy, audit)
    trace.append(t4)

    return report, trace
