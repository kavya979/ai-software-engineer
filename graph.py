"""
AI Software Engineer - Main LangGraph Orchestration
"""
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from state import AgentState
from agents.planner import planner_node
from agents.coder import coder_node
from agents.reviewer import reviewer_node
from agents.test_runner import test_runner_node

MAX_ITERATIONS = 3


def should_retry_after_review(state: AgentState) -> str:
    """After review: if passed or max retries hit, go to test runner. Else retry coder."""
    if state["review_passed"] or state["iteration"] >= MAX_ITERATIONS:
        return "test_runner"
    return "coder"


def should_retry_after_tests(state: AgentState) -> str:
    """After tests: if all passed or max retries hit, finish. Else retry coder."""
    if state["tests_passed"] or state["iteration"] >= MAX_ITERATIONS:
        return "end"
    return "coder"


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("planner",     planner_node)
    builder.add_node("coder",       coder_node)
    builder.add_node("reviewer",    reviewer_node)
    builder.add_node("test_runner", test_runner_node)

    # Entry point
    builder.set_entry_point("planner")

    # Fixed edges
    builder.add_edge("planner", "coder")
    builder.add_edge("coder",   "reviewer")

    # Conditional: after review
    builder.add_conditional_edges(
        "reviewer",
        should_retry_after_review,
        {
            "coder":       "coder",
            "test_runner": "test_runner",
        }
    )

    # Conditional: after tests
    builder.add_conditional_edges(
        "test_runner",
        should_retry_after_tests,
        {
            "coder": "coder",
            "end":   END,
        }
    )

    return builder.compile()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    from pathlib import Path

    feature_request = input("📋 Enter your feature request:\n> ").strip()

    graph = build_graph()

    initial_state: AgentState = {
        "feature_request":  feature_request,
        "subtasks":         [],
        "code_files":       {},
        "review_feedback":  "",
        "review_passed":    False,
        "test_results":     {},
        "tests_passed":     False,
        "iteration":        0,
    }

    print("\n🚀 Running AI Software Engineer...\n")
    final_state = graph.invoke(initial_state)

    # ── Save output files ─────────────────────────────────────────────────────
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    for filename, code in final_state["code_files"].items():
        out_path = output_dir / filename
        out_path.write_text(code)
        print(f"✅ Written: {out_path}")

    # Save summary report
    report = {
        "feature_request": feature_request,
        "subtasks":        final_state["subtasks"],
        "review_feedback": final_state["review_feedback"],
        "test_results":    final_state["test_results"],
        "iterations":      final_state["iteration"],
        "tests_passed":    final_state["tests_passed"],
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"📄 Report saved: {report_path}")
    print(f"\n🏁 Done in {final_state['iteration']} iteration(s). Tests passed: {final_state['tests_passed']}")
