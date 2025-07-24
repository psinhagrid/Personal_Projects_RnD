import os
import argparse
import re
from tool_registry import ToolRegistry
from anthropic import Anthropic

# ---------------------------
# Step 1: Parse CLI arguments
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--query", required=True, help="User query or task description")
parser.add_argument("--folder", required=True, help="Folder path for documents to analyze")
parser.add_argument("--api-key", help="Anthropic API key (or use env var)")
parser.add_argument("--model", default="claude-3-5-sonnet-20241022", help="Claude model to use")
parser.add_argument("--save-code", help="Optional file path to save generated code")
args = parser.parse_args()

# --------------------------------
# Step 2: Resolve Anthropic API key
# --------------------------------
api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("Anthropic API key must be provided via --api-key or ANTHROPIC_API_KEY environment variable.")

# -------------------------------
# Step 3: Define available tools
# -------------------------------
TOOL_SPECS = [
    {
        "id": "similarity_searcher.tool_similarity_search",
        "description": "Find semantically similar documents in a folder",
    },
    {
        "id": "clarification.tool_generate_clarification_questions",
        "description": "Generate 3-5 clarification questions based on the query and file content",
    }
]

# ----------------------------------------
# Step 4: Create ToolRegistry & LLM client
# ----------------------------------------
tool_registry = ToolRegistry("tools")
client = Anthropic(api_key=api_key)

# -------------------------------
# Step 5: Construct system prompt
# -------------------------------
tool_list_description = "\n".join(
    f'{i+1}. "{tool["id"]}" – {tool["description"]}'
    for i, tool in enumerate(TOOL_SPECS)
)

system_prompt = f"""
You are an expert AI that writes Python scripts using ToolRegistry-based tools.

Important rules:
- You MUST call tools using: tool_registry.invoke_tool("<tool_id>", **kwargs)
- DO NOT write placeholder functions.
- DO NOT hardcode user inputs — use the variables provided.
- Use the available tools below exactly as defined.

Available tools:
{tool_list_description}

Variables available in the script:
- query: user query string
- folder_path: input folder path
- tool_registry: ToolRegistry instance
- api_key, model: for downstream tool access

Required steps in code:
1. Use the similarity search tool to get relevant documents.
2. Filter out only those marked relevant.
3. Pass those to the clarification tool.
4. Log or print results.

Ensure clean structure, logging, and exception handling.
"""

# -------------------------------
# Step 6: Construct user prompt
# -------------------------------
final_user_prompt = f"""
Generate a Python script that does the following:
- Based on this query: "{args.query}"
- Scans the folder: "{args.folder}"
- Uses the tools defined in the ToolRegistry
- Handles the outputs and prints relevant results

The script must:
- Use tool_registry.invoke_tool()
- Be directly executable
- Avoid mockups or placeholders
"""

# -------------------------------
# Step 7: Call Claude
# -------------------------------
print("🛠️ Generating CodeAct solution...\n")

response = client.messages.create(
    model=args.model,
    system=system_prompt,
    messages=[{"role": "user", "content": final_user_prompt}],
    max_tokens=4096,
    temperature=0.1,
)

# -------------------------------
# Step 8: Extract and display code
# -------------------------------
generated_code = ""
if not response.content:
    raise RuntimeError("Claude response is empty.")

for block in response.content:
    if block.type == "text":
        generated_code += block.text

# Optional: remove Markdown code fences if present
match = re.search(r"```(?:python)?\n(.*?)```", generated_code, re.DOTALL)
if match:
    generated_code = match.group(1)

print("✅ Code Generation Complete:\n")
print(generated_code)

# -------------------------------
# Step 9: Save to file if requested
# -------------------------------
if args.save_code:
    try:
        with open(args.save_code, "w", encoding="utf-8") as f:
            f.write(generated_code)
        print(f"\n💾 Code saved to {args.save_code}")
    except Exception as e:
        print(f"❌ Failed to save code: {e}")
