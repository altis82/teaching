"""
Day 5 – Multi-Agent Orchestration and Planning System
Uses LangGraph to coordinate specialized agents that research, analyze,
write, and critique a market analysis report.
"""

from __future__ import annotations

import json
from typing import TypedDict

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

from config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    TEMPERATURE, TOP_P, TOP_K,
    QUALITY_THRESHOLD, MAX_REVISIONS,
)


# ── Shared State ─────────────────────────────────────────────────────
class ReportState(TypedDict):
    topic: str
    research_notes: str
    analysis: str
    draft_report: str
    critique: str
    quality_score: float
    final_report: str
    revision_count: int


# ── LLM helper ───────────────────────────────────────────────────────
def get_llm(temperature: float | None = None):
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=temperature if temperature is not None else TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
    )


def run_prompt(system: str, user: str, temperature: float | None = None) -> str:
    """Run a single prompt through the LLM and return text."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{input}"),
    ])
    chain = prompt | get_llm(temperature) | StrOutputParser()
    return chain.invoke({"input": user})


# ── Node 1: Research Agent ───────────────────────────────────────────
def research_node(state: ReportState) -> ReportState:
    topic = state["topic"]
    print(f"\n[1/4 Research Agent] Gathering information on: {topic}")

    system = (
        "You are a research analyst. Your job is to gather comprehensive "
        "information on a given topic. Produce detailed research notes "
        "organized by subtopic. Include: market size, key players, trends, "
        "growth drivers, challenges, and recent developments. "
        "Be factual and thorough."
    )
    user = f"Research the following topic thoroughly:\n\n{topic}"

    notes = run_prompt(system, user)
    state["research_notes"] = notes
    print(f"  → {len(notes)} chars of research notes")
    return state


# ── Node 2: Analyst Agent ───────────────────────────────────────────
def analyst_node(state: ReportState) -> ReportState:
    print("\n[2/4 Analyst Agent] Processing research into structured insights...")

    system = (
        "You are a data analyst. Given raw research notes, produce a "
        "structured analysis with:\n"
        "1. Executive summary (3-4 sentences)\n"
        "2. Key statistics and metrics\n"
        "3. Trend analysis (what's growing, declining)\n"
        "4. Competitive landscape\n"
        "5. Risks and opportunities\n"
        "Be precise and data-driven."
    )
    user = f"Analyze these research notes:\n\n{state['research_notes']}"

    analysis = run_prompt(system, user)
    state["analysis"] = analysis
    print(f"  → {len(analysis)} chars of analysis")
    return state


# ── Node 3: Writer Agent ────────────────────────────────────────────
def writer_node(state: ReportState) -> ReportState:
    print("\n[3/4 Writer Agent] Drafting the report...")

    # If there's previous critique, include it for revision
    revision_context = ""
    if state.get("critique"):
        revision_context = (
            f"\n\nPrevious draft was critiqued. Feedback:\n{state['critique']}\n"
            "Please address the feedback in this revision."
        )

    system = (
        "You are a professional report writer. Given structured analysis, "
        "write a polished market analysis report with:\n"
        "- Title\n"
        "- Executive Summary\n"
        "- Market Overview\n"
        "- Key Players & Competitive Landscape\n"
        "- Trends & Growth Drivers\n"
        "- Risks & Challenges\n"
        "- Outlook & Recommendations\n\n"
        "Write in clear, professional language. Use markdown formatting."
    )
    user = (
        f"Topic: {state['topic']}\n\n"
        f"Analysis:\n{state['analysis']}"
        f"{revision_context}"
    )

    draft = run_prompt(system, user, temperature=0.7)
    state["draft_report"] = draft
    state["revision_count"] = state.get("revision_count", 0) + 1
    print(f"  → {len(draft)} chars (revision #{state['revision_count']})")
    return state


# ── Node 4: Critic Agent ────────────────────────────────────────────
def critic_node(state: ReportState) -> ReportState:
    print("\n[4/4 Critic Agent] Reviewing the report...")

    system = (
        "You are a strict quality reviewer. Evaluate the report on:\n"
        "1. Completeness (all sections present?)\n"
        "2. Accuracy (claims supported by data?)\n"
        "3. Clarity (well-written, no jargon?)\n"
        "4. Structure (logical flow?)\n"
        "5. Actionability (useful recommendations?)\n\n"
        "Give a quality score from 0.0 to 1.0.\n"
        "List specific improvements needed.\n\n"
        "IMPORTANT: Respond in this exact format:\n"
        "SCORE: <number between 0.0 and 1.0>\n"
        "FEEDBACK:\n<your detailed feedback>"
    )
    user = f"Review this report:\n\n{state['draft_report']}"

    critique = run_prompt(system, user, temperature=0.2)
    state["critique"] = critique

    # Parse score from response
    score = 0.5  # default
    for line in critique.split("\n"):
        line_stripped = line.strip().upper()
        if line_stripped.startswith("SCORE:"):
            try:
                score = float(line_stripped.split(":", 1)[1].strip())
                score = max(0.0, min(1.0, score))
            except ValueError:
                pass
            break

    state["quality_score"] = score
    print(f"  → Quality score: {score:.2f} (threshold: {QUALITY_THRESHOLD})")
    return state


# ── Node 5: Finalize ────────────────────────────────────────────────
def finalize_node(state: ReportState) -> ReportState:
    print("\n[Done] Finalizing report...")
    state["final_report"] = state["draft_report"]
    return state


# ── Routing Logic ────────────────────────────────────────────────────
def route_after_critique(state: ReportState) -> str:
    """Decide whether to revise or finalize."""
    if (state["quality_score"] < QUALITY_THRESHOLD
            and state.get("revision_count", 0) < MAX_REVISIONS):
        print(f"  → Score below {QUALITY_THRESHOLD}, sending back for revision...")
        return "revise"
    return "finalize"


# ── Build Graph ──────────────────────────────────────────────────────
def build_workflow():
    """
    Graph:
        START → research → analyst → writer → critic
                                        ↑         ↓
                                        └── revise (if score < threshold)
                                                   ↓
                                              finalize → END
    """
    workflow = StateGraph(ReportState)

    workflow.add_node("research", research_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("finalize", finalize_node)

    workflow.add_edge(START, "research")
    workflow.add_edge("research", "analyst")
    workflow.add_edge("analyst", "writer")
    workflow.add_edge("writer", "critic")
    workflow.add_conditional_edges("critic", route_after_critique, {
        "revise": "writer",
        "finalize": "finalize",
    })
    workflow.add_edge("finalize", END)

    return workflow.compile()


# ── Main ─────────────────────────────────────────────────────────────
def main():
    print("=== Multi-Agent Report Generator ===\n")

    topic = input("Enter report topic (or press Enter for default): ").strip()
    if not topic:
        topic = "The AI market in Canada: market size, key players, growth trends, and future outlook"

    print(f"\nTopic: {topic}")
    print(f"Quality threshold: {QUALITY_THRESHOLD}")
    print(f"Max revisions: {MAX_REVISIONS}")
    print("=" * 60)

    app = build_workflow()

    initial_state: ReportState = {
        "topic": topic,
        "research_notes": "",
        "analysis": "",
        "draft_report": "",
        "critique": "",
        "quality_score": 0.0,
        "final_report": "",
        "revision_count": 0,
    }

    result = app.invoke(initial_state)

    # Print final report
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(result["final_report"])
    print("=" * 60)
    print(f"\nQuality score: {result['quality_score']:.2f}")
    print(f"Revisions: {result['revision_count']}")

    # Save to file
    output_file = "report_output.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result["final_report"])
    print(f"Report saved to: {output_file}")


if __name__ == "__main__":
    main()
