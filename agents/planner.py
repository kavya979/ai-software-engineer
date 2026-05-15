from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from state import AgentState


def planner_node(state: AgentState) -> AgentState:
    print("🧠 [Planner] Decomposing feature request...")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    prompt = f"""You are a senior software architect.
Break the following feature request into a numbered list of clear, concrete coding subtasks.
Each subtask should be specific enough for a developer to implement independently.
Return ONLY the numbered list — no preamble, no explanation.

Feature request:
{state['feature_request']}
"""

    result = llm.invoke(prompt)
    lines = [
        line.strip()
        for line in result.content.strip().splitlines()
        if line.strip()
    ]

    print(f"   → {len(lines)} subtasks identified.")
    for line in lines:
        print(f"     {line}")

    return {**state, "subtasks": lines, "iteration": 0}
