import os
import importlib.util
from pathlib import Path
from typing import Dict, Callable, Any

class ToolRegistry:
    def __init__(self, tools_path: str):
        self.tools_path = Path(tools_path)
        self.tools: Dict[str, Callable] = {}
        self._load_all_tools()

    def _load_all_tools(self):
        for file in os.listdir(self.tools_path):
            if file.endswith(".py") and not file.startswith("__"):
                module_name = file[:-3]
                module_path = self.tools_path / file
                self._load_tool_module(module_name, module_path)

    def _load_tool_module(self, name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        if not spec or not spec.loader:
            print(f"⚠️ Failed to load spec for {name}")
            return

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Register all callables starting with "tool_"
        for attr in dir(module):
            if attr.startswith("tool_") and callable(getattr(module, attr)):
                tool_func = getattr(module, attr)
                tool_name = f"{name}.{attr}"
                self.tools[tool_name] = tool_func
                print(f"✅ Registered tool: {tool_name}")

    def invoke_tool(self, tool_name: str, **kwargs) -> Any:
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found.")
        return self.tools[tool_name](**kwargs)

    def list_tools(self) -> Dict[str, Callable]:
        return self.tools
