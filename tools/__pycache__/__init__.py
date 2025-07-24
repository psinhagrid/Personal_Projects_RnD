import os
import importlib.util
from pathlib import Path
from typing import Dict, Callable

def discover_tools() -> Dict[str, Callable]:
    tools_dir = Path(__file__).parent
    tool_functions = {}

    for file in os.listdir(tools_dir):
        if file.startswith("__") or not file.endswith(".py"):
            continue

        module_name = file[:-3]
        module_path = tools_dir / file

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Register all functions that start with "tool_"
            for attr_name in dir(module):
                if attr_name.startswith("tool_"):
                    func = getattr(module, attr_name)
                    if callable(func):
                        tool_functions[attr_name] = func

    return tool_functions
