def tool(func):
    """
    Marks a function as a tool to be registered by the agent.
    """
    func._is_tool = True
    return func