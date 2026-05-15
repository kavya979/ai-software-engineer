"""
Shared LangGraph state — passed between every node.
"""

from typing import TypedDict, List, Dict


class AgentState(TypedDict):
    feature_request: str        # Original user request
    subtasks:        List[str]  # Planner output
    code_files:      Dict[str, str]  # filename -> source code
    review_feedback: str        # Reviewer comments / "PASS"
    review_passed:   bool       # Did reviewer approve?
    test_results:    Dict[str, str]  # filename -> stdout/stderr
    tests_passed:    bool       # Did all files run cleanly?
    iteration:       int        # Fix-loop counter
