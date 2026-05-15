from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from state import AgentState


def reviewer_node(state: AgentState) -> AgentState:
    print("\n🔍 [Reviewer] Reviewing code...")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    files_block = "\n\n".join(
        f"### {fname}\n```python\n{code}\n```"
        for fname, code in state["code_files"].items()
    )

    prompt = f"""You are a senior Python code reviewer.
Review the following code files against these subtasks:
{chr(10).join(state['subtasks'])}

Code:
{files_block}

Reply with EXACTLY:
- The single word "PASS" if everything is correct and all subtasks are implemented.
- A numbered list of specific issues if anything needs fixing.
Do NOT say PASS and also list issues.
"""

    result = llm.invoke(prompt)
    feedback = result.content.strip()
    passed = feedback.upper().startswith("PASS")

    if passed:
        print("   ✅ Review passed!")
    else:
        print("   ❌ Issues found:")
        for line in feedback.splitlines()[:5]:
            print(f"     {line}")

    return {**state, "review_feedback": feedback, "review_passed": passed}
