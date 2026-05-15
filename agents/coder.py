from dotenv import load_dotenv
load_dotenv()

import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from state import AgentState


def _extract_json(text: str) -> dict:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"No JSON found in response:\n{text}")


def coder_node(state: AgentState) -> AgentState:
    iteration = state["iteration"] + 1
    print(f"\n💻 [Coder] Writing code (iteration {iteration})...")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    feedback_section = ""
    if state["review_feedback"]:
        feedback_section = f"\n--- REVIEWER FEEDBACK ---\n{state['review_feedback']}\n"

    test_section = ""
    if state["test_results"]:
        test_section = f"\n--- TEST ERRORS ---\n{json.dumps(state['test_results'], indent=2)}\n"

    prompt = f"""You are an expert Python developer.
Implement the following subtasks as clean, well-commented Python code.
{feedback_section}{test_section}

Subtasks:
{chr(10).join(state['subtasks'])}

Rules:
- Return a single JSON object where each key is a filename (e.g. "main.py") and the value is the complete file source.
- Every file must be syntactically valid Python.
- Include a __main__ block so the primary file can run directly.
- Do NOT include any text outside the JSON object.

Example:
{{
  "main.py": "# full source\\n..."
}}
"""

    result = llm.invoke(prompt)

    try:
        code_files = _extract_json(result.content)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"   ⚠️  Could not parse response: {e}")
        code_files = state["code_files"]

    for fname in code_files:
        print(f"   → Generated: {fname}")

    return {**state, "code_files": code_files, "iteration": iteration}
