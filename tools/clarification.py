import os
from typing import List
import anthropic
from tools.utils import tool  # 👈 Import the decorator

@tool  # 👈 Register as a CodeAct tool
def tool_generate_clarification_questions(
    query: str,
    file_paths: List[str],
    api_key: str = None,
    model: str = "claude-3-5-sonnet-20241022"
) -> List[str]:  # ⬅️ Note return type changed to List[str]
    """
    Generates 3-5 clarification questions using Claude based on user query and file content.
    """
    if not api_key:
        api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Anthropic API key not provided")

    client = anthropic.Anthropic(api_key=api_key)

    file_snippets = []
    for path in file_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                file_snippets.append(f"--- {os.path.basename(path)} ---\n{content}")
        except Exception as e:
            file_snippets.append(f"--- {os.path.basename(path)} ---\n[Error reading file: {e}]")

    full_context = "\n\n".join(file_snippets)

    system_prompt = (
        "You are an expert software assistant. Your job is to ask clarification questions "
        "to the user that would help generate a more accurate response to their query."
    )

    user_prompt = (
        f"Given the user query:\n\n\"{query}\"\n\n"
        f"and the following relevant documents:\n\n{full_context}\n\n"
        f"Ask 3-5 concise clarification questions to improve understanding of the query."
    )

    response = client.messages.create(
        model=model,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=512,
        temperature=0.3
    )

    # ✅ Extract questions
    answer = response.content[0].text.strip()
    questions = [q.strip(" -").strip() for q in answer.split("\n") if q.strip()]

    return questions
