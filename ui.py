"""
Streamlit web UI for the AI Software Engineer.
Run with: streamlit run ui.py
"""

import json
import sys
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from graph import build_graph
from state import AgentState

st.set_page_config(page_title="AI Software Engineer", page_icon="🤖", layout="wide")

st.title("🤖 AI Software Engineer")
st.caption("Powered by LangGraph + Google Gemini")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    max_iterations = st.slider("Max fix iterations", 1, 5, 3)
    st.divider()
    st.markdown("**How it works:**")
    st.markdown("""
1. 🧠 Planner breaks request into subtasks  
2. 💻 Coder writes Python files  
3. 🔍 Reviewer critiques the code  
4. 🧪 Test Runner executes files  
5. 🔄 Loops back if errors found  
    """)

# ── Main input ────────────────────────────────────────────────────────────────
feature_request = st.text_area(
    "📋 Feature Request",
    placeholder="e.g. Build a Python script that fetches weather data from the OpenWeatherMap API and caches results to a JSON file for 10 minutes.",
    height=120,
)

run_button = st.button("🚀 Generate Code", type="primary", disabled=not feature_request.strip())

if run_button and feature_request.strip():
    graph = build_graph()

    initial_state: AgentState = {
        "feature_request": feature_request,
        "subtasks":        [],
        "code_files":      {},
        "review_feedback": "",
        "review_passed":   False,
        "test_results":    {},
        "tests_passed":    False,
        "iteration":       0,
    }

    # ── Stream progress ───────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📊 Progress")
        status_box = st.empty()

    with col2:
        st.subheader("📁 Generated Files")
        files_box = st.empty()

    with st.spinner("Running AI Software Engineer..."):
        # Stream node-by-node
        node_statuses = []
        final_state = None

        for step in graph.stream(initial_state):
            node_name = list(step.keys())[0]
            state = list(step.values())[0]
            final_state = state

            emoji_map = {
                "planner":     "🧠 Planner",
                "coder":       "💻 Coder",
                "reviewer":    "🔍 Reviewer",
                "test_runner": "🧪 Test Runner",
            }
            label = emoji_map.get(node_name, node_name)
            node_statuses.append(f"✅ {label} — done (iteration {state.get('iteration', 0)})")

            status_box.markdown("\n".join(node_statuses))

            if state.get("code_files"):
                files_preview = "\n".join(
                    f"- `{f}`" for f in state["code_files"]
                )
                files_box.markdown(files_preview)

    # ── Results ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🎉 Results")

    if final_state:
        tests_ok = final_state.get("tests_passed", False)
        st.success("All tests passed! ✅") if tests_ok else st.warning("Some tests failed — see details below.")

        tabs = st.tabs(
            ["📋 Subtasks", "💻 Code Files", "🔍 Review", "🧪 Test Output"]
        )

        with tabs[0]:
            for task in final_state.get("subtasks", []):
                st.markdown(f"- {task}")

        with tabs[1]:
            for fname, code in final_state.get("code_files", {}).items():
                st.markdown(f"**{fname}**")
                st.code(code, language="python")

                # Download button per file
                st.download_button(
                    label=f"⬇️ Download {fname}",
                    data=code,
                    file_name=fname,
                    mime="text/plain",
                )

        with tabs[2]:
            feedback = final_state.get("review_feedback", "")
            if feedback.upper().startswith("PASS"):
                st.success(feedback)
            else:
                st.error(feedback)

        with tabs[3]:
            for fname, output in final_state.get("test_results", {}).items():
                st.markdown(f"**{fname}**")
                st.code(output, language="text")
