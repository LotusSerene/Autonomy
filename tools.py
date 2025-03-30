import json
import os  # Needed for file/dir operations
import subprocess  # Potentially for Git operations
from typing import Dict, Any, Callable, List
import datetime  # Already imported, but good to note dependency

# --- Tool Implementations (Stubs) ---

# Existing Web Search (assuming duckduckgo-search is installed)
try:
    from duckduckgo_search import DDGS

    def web_search(query: str, max_results: int = 3) -> str:
        """Performs a web search using DuckDuckGo. Returns JSON string of results."""
        print(f"--- Executing Web Search Tool ---")
        print(f"Query: {query}")
        try:
            # Use a context manager for DDGS
            with DDGS() as ddgs:
                # Use ddgs.text which returns a list of dicts
                results = list(ddgs.text(query, max_results=max_results))
            print(f"Found {len(results)} results.")
            # Format results as a JSON string for consistency
            return json.dumps(results) if results else "No results found."
        except Exception as e:
            print(f"Error during web search: {e}")
            return f"Error performing web search: {e}"

except ImportError:
    print(
        "Warning: duckduckgo-search not installed. Web search tool will not be available."
    )
    print("Install using: pip install duckduckgo-search")

    def web_search(query: str, max_results: int = 3) -> str:
        """Placeholder if duckduckgo-search is not installed."""
        print("--- Web Search Tool (Not Available) ---")
        return "Error: Web search tool requires 'duckduckgo-search' library."


# --- New Tool Stubs ---


def arxiv_search(query: str, max_results: int = 3) -> str:
    """Searches arXiv for academic papers. Input is the search query."""
    # TODO: Implement using the arxiv library (pip install arxiv)
    print(f"--- Executing ArXiv Search Tool (Stub) ---")
    print(f"Query: {query}, Max Results: {max_results}")
    return f"Placeholder: ArXiv search results for '{query}' would appear here."


def wikipedia_search(query: str) -> str:
    """Searches Wikipedia for a summary. Input is the search query."""
    # TODO: Implement using the wikipedia library (pip install wikipedia)
    print(f"--- Executing Wikipedia Search Tool (Stub) ---")
    print(f"Query: {query}")
    return f"Placeholder: Wikipedia summary for '{query}' would appear here."


def read_file(filepath: str) -> str:
    """Reads the content of a file at the specified path."""
    # IMPORTANT: Add security checks - limit accessible paths?
    print(f"--- Executing Read File Tool (Stub) ---")
    print(f"Filepath: {filepath}")
    try:
        # Example basic implementation (needs error handling, path validation)
        if not os.path.exists(filepath):
            return f"Error: File not found at '{filepath}'"
        # Add checks here to prevent reading sensitive files
        # e.g., restrict to a specific workspace directory
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return content[:2000]  # Limit output size
    except Exception as e:
        return f"Error reading file '{filepath}': {e}"


def edit_file(instructions: str) -> str:
    """
    Edits an existing file based on instructions.
    Input should clearly specify the filepath and the changes required.
    Example format for instructions: '{"filepath": "path/to/file.txt", "action": "replace", "old_text": "abc", "new_text": "xyz"}'
    Or: '{"filepath": "path/to/file.py", "action": "insert", "line_number": 10, "text": "new code line"}'
    """
    # IMPORTANT: VERY HIGH RISK. Needs strict validation, sandboxing, backups.
    # TODO: Implement parsing of instructions (e.g., JSON) and file editing logic.
    print(f"--- Executing Edit File Tool (Stub) ---")
    print(f"Instructions: {instructions}")
    # Needs careful implementation: parse instructions, validate path, perform edit safely.
    return f"Placeholder: File editing based on '{instructions}' would happen here. Requires careful implementation!"


def api_call(request_details: str) -> str:
    """
    Makes an API call based on provided details.
    Input should specify URL, method (GET/POST), headers, body, etc.
    Example format: '{"url": "https://api.example.com/data", "method": "GET", "params": {"id": 123}}'
    """
    # TODO: Implement using requests library (pip install requests). Add error handling.
    # Consider security: restrict allowed domains? Handle API keys securely.
    print(f"--- Executing API Call Tool (Stub) ---")
    print(f"Request Details: {request_details}")
    # Needs implementation: parse details, make request using 'requests' library.
    return f"Placeholder: API call result for '{request_details}' would appear here."


def datausa_api(query_params: str) -> str:
    """
    Queries the Data USA API (datausa.io).
    Input should be query parameters, e.g., '{"data_source": "acs/profile", "measures": "Population", "geo": "nation/us"}'
    """
    # TODO: Implement using requests library, targeting datausa.io endpoints.
    print(f"--- Executing Data USA API Tool (Stub) ---")
    print(f"Query Params: {query_params}")
    base_url = "https://datausa.io/api/data"
    # Needs implementation: parse params, construct URL, make GET request.
    return f"Placeholder: Data USA API result for '{query_params}' would appear here."


def create_subgoal(goal_description: str) -> str:
    """
    Registers a new sub-goal for the agent to potentially address later.
    Input is a description of the sub-goal.
    (This might interact with the agent's internal state or goal memory).
    """
    # TODO: Decide how this interacts with the agent state/memory.
    # Could it add to goal_memory? Or modify AgentState directly?
    print(f"--- Executing Create Subgoal Tool (Stub) ---")
    print(f"Sub-goal Description: {goal_description}")
    # Needs integration with the agent's goal management logic.
    return f"Placeholder: Sub-goal '{goal_description}' registered. Needs integration."


def request_human_input(question: str) -> str:
    """
    Pauses execution and asks the user (human) a specific question.
    The agent will wait for the human's response to proceed.
    """
    # TODO: Implement the mechanism to pause and get input (e.g., print and input()).
    # This will likely require changes in the agent graph execution flow.
    print(f"--- Executing Request Human Input Tool (Stub) ---")
    print(f"Question for Human: {question}")
    # Needs implementation to actually pause and wait for input.
    response = input(f"AGENT REQUEST: {question}\nYour Input: ")
    return f"Human response received: {response}"  # Simulate receiving input


def create_file(filepath_and_content: str) -> str:
    """
    Creates a new file with specified content.
    Input format example: '{"filepath": "path/new_file.txt", "content": "Initial content."}'
    """
    # IMPORTANT: Add security checks - limit accessible paths?
    # TODO: Implement parsing and file creation logic. Ensure directory exists or handle creation.
    print(f"--- Executing Create File Tool (Stub) ---")
    print(f"Filepath and Content: {filepath_and_content}")
    # Needs implementation: parse input, validate path, create file.
    return f"Placeholder: File creation based on '{filepath_and_content}' would happen here."


def create_directory(dirpath: str) -> str:
    """Creates a new directory at the specified path."""
    # IMPORTANT: Add security checks - limit accessible paths?
    print(f"--- Executing Create Directory Tool (Stub) ---")
    print(f"Directory Path: {dirpath}")
    try:
        # Example basic implementation (needs path validation)
        # Add checks here to prevent creating directories in sensitive locations
        os.makedirs(
            dirpath, exist_ok=True
        )  # exist_ok=True prevents error if dir exists
        return f"Directory '{dirpath}' created or already exists."
    except Exception as e:
        return f"Error creating directory '{dirpath}': {e}"


def git_operation(command_details: str) -> str:
    """
    Executes a Git command in a specified repository directory.
    Input format example: '{"repo_path": "/path/to/repo", "command": "git status"}'
    Or: '{"repo_path": "/path/to/repo", "command": "git clone https://github.com/user/repo.git ."}'
    """
    # IMPORTANT: VERY HIGH RISK. Needs strict validation, sandboxing.
    # TODO: Implement parsing, validation, and execution using subprocess.
    print(f"--- Executing Git Operation Tool (Stub) ---")
    print(f"Command Details: {command_details}")
    # Needs careful implementation: parse details, validate path/command, run using subprocess safely.
    return f"Placeholder: Git operation '{command_details}' would happen here. Requires careful implementation!"


def list_tools() -> str:
    """Lists the names and descriptions of all available tools."""
    print(f"--- Executing List Tools Tool ---")
    available = get_available_tools_list()
    return json.dumps(available, indent=2)  # Return as formatted JSON string


def prioritize_goals(goal_list_and_context: str) -> str:
    """
    Analyzes a list of goals and suggests a priority order based on context.
    Input format example: '{"goals": ["goal A", "goal B"], "context": "Urgent deadline approaching"}'
    (This might involve an LLM call internally).
    """
    # TODO: Implement logic, potentially calling an LLM. Needs integration with goal management.
    print(f"--- Executing Prioritize Goals Tool (Stub) ---")
    print(f"Input: {goal_list_and_context}")
    # Needs implementation, likely involves LLM call.
    return f"Placeholder: Goal prioritization based on '{goal_list_and_context}' would happen here."


def manage_goals(action_details: str) -> str:
    """
    Manages the agent's goals (add, remove, update status).
    Input format example: '{"action": "update", "goal_id": "goal_123", "status": "achieved"}'
    Or: '{"action": "add", "description": "New goal C"}'
    (Interacts with the agent's goal memory/state).
    """
    # TODO: Implement interaction with goal memory (e.g., Qdrant goal_memory collection).
    print(f"--- Executing Manage Goals Tool (Stub) ---")
    print(f"Action Details: {action_details}")
    # Needs integration with goal memory system.
    return f"Placeholder: Goal management action '{action_details}' would happen here."


def request_tool_enhancement(request_description: str) -> str:
    """
    Logs a request for a new tool or enhancement to an existing one.
    Input is a description of the needed capability.
    """
    # TODO: Implement logging mechanism (e.g., write to a file, send to a specific API endpoint).
    print(f"--- Executing Request Tool Enhancement Tool (Stub) ---")
    print(f"Enhancement Request: {request_description}")
    # Needs implementation for logging the request.
    log_file = "tool_enhancement_requests.log"
    try:
        with open(log_file, "a") as f:
            f.write(f"{datetime.datetime.now().isoformat()}: {request_description}\n")
        return f"Enhancement request logged to {log_file}."
    except Exception as e:
        return f"Error logging enhancement request: {e}"


# --- New Tool Stubs for Autonomy ---


def monitor_environment(trigger_conditions: str) -> str:
    """
    Checks predefined triggers or monitors specific conditions.
    Input format example: '{"type": "file_change", "path": "/path/to/watch/file.txt"}'
    Or: '{"type": "api_check", "url": "http://status.example.com", "expected_value": "OK"}'
    Or: '{"type": "time_schedule", "cron": "0 9 * * MON"}' # Check if it's time for a scheduled task

    Returns a description of triggered conditions or 'No triggers activated'.
    """
    # TODO: Implement logic to check various trigger types.
    # This is complex. A real implementation might involve:
    # - Reading a configuration file of triggers.
    # - Checking file modification times (os.path.getmtime).
    # - Making API calls (using requests).
    # - Checking current time against schedules (using croniter library?).
    # - Checking system metrics (psutil library?).
    # This tool might be called periodically by the agent's main loop or a separate process.
    print(f"--- Executing Monitor Environment Tool (Stub) ---")
    print(f"Trigger Conditions to Check: {trigger_conditions}")
    # Needs implementation to parse conditions and perform checks.
    return f"Placeholder: Environment monitoring based on '{trigger_conditions}' would happen here. Needs implementation."


def generate_hypothesis(context: str) -> str:
    """
    Analyzes the current situation and capabilities to propose potential new goals or opportunities for proactive action.
    Input is a summary of the current context (e.g., recent observations, available tools, idle status).
    (This will likely involve an LLM call internally).
    """
    # TODO: Implement using an LLM (e.g., main_llm).
    # Prompt should guide the LLM to think creatively based on agent's purpose, tools, and state.
    print(f"--- Executing Generate Hypothesis Tool (Stub) ---")
    print(f"Context for Hypothesis: {context}")
    # Needs implementation: Construct prompt, call LLM, parse response.
    # Example Prompt Idea: "Given my tools ({tool_list}), recent activity ({context}), and my goal to be proactively helpful, suggest one specific, actionable task I could undertake."
    return f"Placeholder: Hypothesis/opportunity generation based on '{context}' would happen here. Needs LLM implementation."


def evaluate_self_performance(recent_activity_summary: str) -> str:
    """
    Performs self-reflection on recent actions, focusing on autonomy, efficiency, and goal achievement.
    Input is a summary of recent goals, actions, and outcomes.
    (This will likely involve an LLM call internally).
    """
    # TODO: Implement using an LLM (e.g., main_llm).
    # Prompt should ask specific questions about proactivity, efficiency, tool use, goal success, reliance on human input.
    print(f"--- Executing Evaluate Self Performance Tool (Stub) ---")
    print(f"Summary for Self-Evaluation: {recent_activity_summary}")
    # Needs implementation: Construct prompt, call LLM, parse response.
    # The result could be stored in semantic_memory or a dedicated reflection memory.
    # Example Prompt Idea: "Evaluate the agent's performance based on the following activity: {recent_activity_summary}. Assess proactivity, efficiency, goal success, and reliance on human input. Provide key learnings."
    return f"Placeholder: Self-performance evaluation based on '{recent_activity_summary}' would happen here. Needs LLM implementation."


# --- Tool Registry ---

# Simple dictionary mapping tool names to their functions
AVAILABLE_TOOLS: Dict[str, Callable[..., str]] = {
    "web_search": web_search,
    "arxiv_search": arxiv_search,
    "wikipedia_search": wikipedia_search,
    "read_file": read_file,
    "edit_file": edit_file,  # High risk
    "api_call": api_call,
    "datausa_api": datausa_api,
    "create_subgoal": create_subgoal,
    "request_human_input": request_human_input,  # Requires graph changes
    "create_file": create_file,  # High risk
    "create_directory": create_directory,  # High risk
    "git_operation": git_operation,  # High risk
    "list_tools": list_tools,
    "prioritize_goals": prioritize_goals,  # Needs LLM? Goal integration
    "manage_goals": manage_goals,  # Needs Goal integration
    "request_tool_enhancement": request_tool_enhancement,
    # New autonomy tools:
    "monitor_environment": monitor_environment,  # Needs implementation (complex)
    "generate_hypothesis": generate_hypothesis,  # Needs LLM implementation
    "evaluate_self_performance": evaluate_self_performance,  # Needs LLM implementation
}

# --- Tool Description and Execution Logic (Mostly Unchanged) ---


def get_tool_description(tool_name: str) -> str:
    """Returns the docstring (description) of a tool."""
    tool_func = AVAILABLE_TOOLS.get(tool_name)
    if tool_func and tool_func.__doc__:
        # Add input format hints from the docstring if available
        doc = tool_func.__doc__.strip()
        # Example: Add a line about expected input if mentioned
        # if "Input format example:" in doc:
        #    doc += "\n   (See function docstring for input format details)"
        return doc
    return "No description available."


def get_available_tools_list() -> Dict[str, str]:
    """Returns a dictionary of available tool names and their descriptions."""
    return {name: get_tool_description(name) for name in AVAILABLE_TOOLS}


def execute_tool(action_str: str) -> Dict[str, Any]:
    """
    Parses an action string and executes the corresponding tool.

    Expected action_str format: "TOOL_NAME[argument]"
    Argument might be a simple string or a JSON string for complex inputs.

    Returns:
        A dictionary containing:
        - 'tool_name': The name of the tool executed.
        - 'argument': The argument passed to the tool (string).
        - 'result': The result from the tool execution (string).
        - 'error': An error message if execution failed, otherwise None.
    """
    print(f"Attempting to parse and execute action: {action_str}")
    tool_name = ""
    argument = ""
    error = None
    result = None

    try:
        # Basic parsing: Find first '[' and assume it separates tool name and argument
        if "[" in action_str and action_str.endswith("]"):
            split_index = action_str.find("[")
            tool_name = action_str[:split_index].strip()
            # Argument is everything between the first '[' and the last ']'
            argument = action_str[split_index + 1 : -1].strip()
        else:
            # Handle tools that might not need arguments, like list_tools
            if action_str.strip() in AVAILABLE_TOOLS:
                tool_name = action_str.strip()
                argument = ""  # No argument provided
            else:
                error = f"Invalid action format. Expected 'TOOL_NAME[argument]' or 'TOOL_NAME', got '{action_str}'."
                print(error)

        if tool_name and tool_name in AVAILABLE_TOOLS:
            tool_function = AVAILABLE_TOOLS[tool_name]
            print(f"Executing tool '{tool_name}' with argument '{argument}'")

            # Check if the function expects an argument
            import inspect

            sig = inspect.signature(tool_function)
            if len(sig.parameters) > 0:
                # Tool expects argument(s)
                if (
                    argument == ""
                    and list(sig.parameters.values())[0].default
                    == inspect.Parameter.empty
                ):
                    # Argument is required but not provided
                    error = f"Tool '{tool_name}' requires an argument, but none was provided in the format TOOL_NAME[argument]."
                    print(error)
                else:
                    # Pass the argument string. The tool itself must handle parsing if it's complex (e.g., JSON).
                    result = tool_function(argument)
            else:
                # Tool takes no arguments (e.g., list_tools)
                if argument != "":
                    print(
                        f"Warning: Tool '{tool_name}' does not take arguments, but '{argument}' was provided. Ignoring argument."
                    )
                result = tool_function()

            if (
                error is None
            ):  # Only print success if no error occurred during execution check
                print(f"Tool '{tool_name}' executed.")  # Simplified success message

        elif not error:  # Only set error if not already set by parsing
            error = f"Tool '{tool_name}' not found in available tools."
            print(error)

    except Exception as e:
        error_msg = f"Error during execution of tool '{tool_name}': {e}"
        print(error_msg)
        error = error_msg
        result = None  # Ensure result is None on error

    return {
        "tool_name": tool_name,
        "argument": argument,  # Store the raw argument string
        "result": result,
        "error": error,
    }


# --- Example Usage (Updated) ---
if __name__ == "__main__":
    print("Available Tools:")
    print(json.dumps(get_available_tools_list(), indent=2))

    print("\n--- Testing Tool Execution ---")

    # Test existing tool
    test_action_1 = "web_search[latest news about AI agents]"
    output_1 = execute_tool(test_action_1)
    print(f"\nExecution Output 1 ({test_action_1}):\n{output_1}")

    # Test tool with no argument
    test_action_2 = "list_tools"
    output_2 = execute_tool(test_action_2)
    print(f"\nExecution Output 2 ({test_action_2}):\n{output_2}")

    # Test tool requiring argument (stub)
    test_action_3 = "read_file[/path/to/some/file.txt]"
    output_3 = execute_tool(test_action_3)
    print(f"\nExecution Output 3 ({test_action_3}):\n{output_3}")

    # Test tool requiring complex argument (stub) - pass as string
    test_action_4 = (
        'edit_file[{"filepath": "test.txt", "action": "append", "text": "new line"}]'
    )
    output_4 = execute_tool(test_action_4)
    print(f"\nExecution Output 4 ({test_action_4}):\n{output_4}")

    # Test non-existent tool
    test_action_5 = "non_existent_tool[query]"
    output_5 = execute_tool(test_action_5)
    print(f"\nExecution Output 5 ({test_action_5}):\n{output_5}")

    # Test invalid format
    test_action_6 = "invalid format"
    output_6 = execute_tool(test_action_6)
    print(f"\nExecution Output 6 ({test_action_6}):\n{output_6}")

    # Test tool requiring argument but missing it
    test_action_7 = "arxiv_search"
    output_7 = execute_tool(test_action_7)
    print(f"\nExecution Output 7 ({test_action_7}):\n{output_7}")

    # Test tool not requiring argument but providing one
    test_action_8 = "list_tools[some argument]"
    output_8 = execute_tool(test_action_8)
    print(f"\nExecution Output 8 ({test_action_8}):\n{output_8}")

    # Test human input tool (will require manual input in terminal)
    # test_action_9 = "request_human_input[What is the primary objective?]"
    # output_9 = execute_tool(test_action_9)
    # print(f"\nExecution Output 9 ({test_action_9}):\n{output_9}")
