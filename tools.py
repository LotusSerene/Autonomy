import json
import os
import subprocess
from typing import Dict, Any, Callable, List
import datetime
import inspect # Added for execute_tool logic
import shlex # Added for safe command splitting in git_operation
from pathlib import Path # Added for path validation

# --- Workspace Configuration (Example) ---
# Define a base directory to restrict file operations for security
WORKSPACE_DIR = Path(os.getenv("AGENT_WORKSPACE", Path.cwd() / "agent_workspace")).resolve()
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True) # Ensure workspace exists
print(f"--- File operations restricted to: {WORKSPACE_DIR} ---")

def _is_path_safe(path_to_check: Path) -> bool:
    """Checks if the path is within the defined WORKSPACE_DIR."""
    try:
        # Resolve the path to make it absolute and normalize it (e.g., remove '..')
        resolved_path = path_to_check.resolve()
        # Check if the resolved path is relative to the workspace directory
        # This prevents directory traversal attacks (e.g., ../../etc/passwd)
        return resolved_path.is_relative_to(WORKSPACE_DIR)
    except Exception:
        # Handle potential errors during path resolution (e.g., invalid characters)
        return False


# --- Tool Implementations ---

# Existing Web Search (assuming duckduckgo-search is installed)
try:
    from duckduckgo_search import DDGS

    def web_search(query: str, max_results: int = 3) -> str:
        """Performs a web search using DuckDuckGo. Returns JSON string of results."""
        print(f"--- Executing Web Search Tool ---")
        print(f"Query: {query}")
        try:
            # Ensure max_results is an integer
            try:
                max_r = int(max_results)
            except (ValueError, TypeError):
                print(f"Warning: Invalid max_results '{max_results}', using default 3.")
                max_r = 3

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_r))
            print(f"Found {len(results)} results.")
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

# ArXiv Search (Implementation)
try:
    import arxiv

    def arxiv_search(query: str, max_results: int = 3) -> str:
        """
        Searches arXiv for academic papers.
        Input is the search query string. max_results defaults to 3.
        Returns a JSON string of search results.
        """
        print(f"--- Executing ArXiv Search Tool ---")
        print(f"Query: {query}, Max Results: {max_results}")
        try:
            # Ensure max_results is an integer
            try:
                max_r = int(max_results)
            except (ValueError, TypeError):
                print(f"Warning: Invalid max_results '{max_results}', using default 3.")
                max_r = 3

            search = arxiv.Search(
                query=query,
                max_results=max_r,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            client = arxiv.Client() # Use client for fetching
            results_list = list(client.results(search)) # Consume the generator

            results = []
            for result in results_list:
                results.append(
                    {
                        "entry_id": result.entry_id,
                        "title": result.title,
                        "summary": result.summary,
                        "authors": [str(a) for a in result.authors],
                        "published": result.published.isoformat(),
                        "pdf_url": result.pdf_url,
                        "comment": result.comment, # Add comment field if available
                    }
                )
            print(f"Found {len(results)} results on ArXiv.")
            return json.dumps(results) if results else "No results found on ArXiv."
        except Exception as e:
            print(f"Error during ArXiv search: {e}")
            return f"Error performing ArXiv search: {e}"

except ImportError:
    print("Warning: arxiv library not installed. ArXiv search tool will not be available.")
    print("Install using: pip install arxiv")

    def arxiv_search(query: str, max_results: int = 3) -> str:
        """Placeholder if arxiv is not installed."""
        print("--- ArXiv Search Tool (Not Available) ---")
        return "Error: ArXiv search tool requires 'arxiv' library."


# Wikipedia Search (Implementation)
try:
    import wikipedia

    def wikipedia_search(query: str) -> str:
        """
        Searches Wikipedia for a summary. Input is the search query string.
        Returns the summary text or an error message.
        """
        print(f"--- Executing Wikipedia Search Tool ---")
        print(f"Query: {query}")
        try:
            # Set language if needed: wikipedia.set_lang("en")
            # Auto suggest can help with ambiguous queries
            # Rate limiting can prevent blocking
            wikipedia.set_rate_limiting(True, min_wait=datetime.timedelta(milliseconds=50))
            summary = wikipedia.summary(query, sentences=3, auto_suggest=True, redirect=True)
            print(f"Found Wikipedia summary for '{query}'.")
            return summary
        except wikipedia.exceptions.PageError:
            return f"Error: Wikipedia page not found for '{query}'."
        except wikipedia.exceptions.DisambiguationError as e:
            # Return options if the query is ambiguous
            options = ", ".join(e.options[:5]) # Limit options shown
            return f"Error: Ambiguous query '{query}'. Wikipedia suggests: {options}. Please be more specific."
        except Exception as e:
            print(f"Error during Wikipedia search: {e}")
            return f"Error performing Wikipedia search: {e}"

except ImportError:
    print(
        "Warning: wikipedia library not installed. Wikipedia search tool will not be available."
    )
    print("Install using: pip install wikipedia")

    def wikipedia_search(query: str) -> str:
        """Placeholder if wikipedia is not installed."""
        print("--- Wikipedia Search Tool (Not Available) ---")
        return "Error: Wikipedia search tool requires 'wikipedia' library."


def read_file(filepath: str) -> str:
    """
    Reads the content of a file at the specified path within the agent's workspace.
    Input: Filepath string (relative to workspace or absolute within workspace).
    Returns the file content (up to 4000 chars) or an error message.
    """
    print(f"--- Executing Read File Tool ---")
    print(f"Requested Filepath: {filepath}")
    try:
        # Resolve the path relative to the workspace if it's not absolute
        target_path = Path(filepath)
        if not target_path.is_absolute():
            # Ensure the relative path doesn't try to escape the workspace early
            # This check is somewhat redundant with _is_path_safe but adds an early exit
            if ".." in filepath:
                 return f"Error: Relative path cannot contain '..'. Path: '{filepath}'"
            target_path = WORKSPACE_DIR / target_path

        # Security Check: Ensure the path is within the workspace
        if not _is_path_safe(target_path):
            return f"Error: Access denied. Path '{filepath}' resolves outside the allowed workspace '{WORKSPACE_DIR}'."

        if not target_path.exists():
            return f"Error: File not found at '{target_path}'"
        if not target_path.is_file():
            return f"Error: Path '{target_path}' is a directory, not a file."

        # Limit file size read to prevent memory issues
        file_size = target_path.stat().st_size
        MAX_READ_SIZE = 1 * 1024 * 1024 # 1 MB limit
        if file_size > MAX_READ_SIZE:
            return f"Error: File size ({file_size} bytes) exceeds the maximum allowed limit ({MAX_READ_SIZE} bytes)."

        with open(target_path, "r", encoding="utf-8", errors='ignore') as f: # Ignore decoding errors
            content = f.read()
        print(f"Successfully read file: {target_path}")
        # Limit output size returned to the LLM
        MAX_OUTPUT_CHARS = 4000
        if len(content) > MAX_OUTPUT_CHARS:
            return content[:MAX_OUTPUT_CHARS] + f"\n... [truncated, total {len(content)} chars]"
        else:
            return content
    except Exception as e:
        return f"Error reading file '{filepath}': {e}"


def edit_file(instructions: str) -> str:
    """
    Edits an existing file within the agent's workspace based on JSON instructions.
    HIGH RISK: Requires careful validation and potentially backups.
    Input: JSON string specifying 'filepath', 'action' ('replace', 'insert', 'append', 'delete_line'), and necessary parameters.
    Example: '{"filepath": "path/to/file.txt", "action": "replace", "old_text": "abc", "new_text": "xyz", "count": 1}' (optional count for replace)
    Example: '{"filepath": "path/to/file.py", "action": "insert", "line_number": 10, "text": "new code line"}' (1-based line number)
    Example: '{"filepath": "path/to/file.txt", "action": "append", "text": "new line at end"}'
    Example: '{"filepath": "path/to/file.txt", "action": "delete_line", "line_number": 5}' (1-based line number)
    Returns a success message or an error message.
    """
    # IMPORTANT: VERY HIGH RISK. Needs strict validation, sandboxing, backups.
    print(f"--- Executing Edit File Tool ---")
    print(f"Instructions: {instructions}")
    filepath = None # Initialize filepath for error reporting
    try:
        params = json.loads(instructions)
        filepath = params.get("filepath")
        action = params.get("action")

        if not filepath or not action:
            return "Error: Missing 'filepath' or 'action' in instructions."

        # Resolve and validate path
        target_path = Path(filepath)
        if not target_path.is_absolute():
             if ".." in filepath:
                 return f"Error: Relative path cannot contain '..'. Path: '{filepath}'"
             target_path = WORKSPACE_DIR / target_path

        if not _is_path_safe(target_path):
            return f"Error: Access denied. Path '{filepath}' resolves outside the allowed workspace '{WORKSPACE_DIR}'."
        if not target_path.exists() or not target_path.is_file():
            return f"Error: File not found or is not a file at '{target_path}'."

        # Read current content
        with open(target_path, "r", encoding="utf-8", errors='ignore') as f:
            lines = f.readlines() # Read as lines for easier manipulation

        original_content = "".join(lines)
        new_content = original_content # Start with original
        modified = False

        # Perform action
        if action == "replace":
            old_text = params.get("old_text")
            new_text = params.get("new_text")
            count = params.get("count") # Optional: number of replacements
            if old_text is None or new_text is None:
                return "Error: 'replace' action requires 'old_text' and 'new_text'."
            try:
                replace_count = int(count) if count is not None else -1 # -1 means replace all
            except ValueError:
                return "Error: 'count' for replace must be an integer."

            if replace_count > 0:
                new_content = original_content.replace(old_text, new_text, replace_count)
            else:
                new_content = original_content.replace(old_text, new_text)

            if new_content != original_content:
                modified = True

        elif action == "insert":
            line_number = params.get("line_number") # 1-based index
            text = params.get("text")
            if line_number is None or text is None:
                return "Error: 'insert' action requires 'line_number' (1-based) and 'text'."
            try:
                line_idx = int(line_number) - 1 # Convert to 0-based index
            except ValueError:
                 return "Error: 'line_number' must be an integer."

            if not isinstance(text, str):
                 return "Error: 'text' for insert must be a string."
            if line_idx < 0 or line_idx > len(lines): # Allow inserting at the very end
                return f"Error: Invalid 'line_number' {line_number}. File has {len(lines)} lines. Must be between 1 and {len(lines) + 1}."

            # Ensure text ends with a newline if inserting as a line
            if not text.endswith('\n'):
                text += '\n'
            lines.insert(line_idx, text)
            new_content = "".join(lines)
            modified = True

        elif action == "append":
            text = params.get("text")
            if text is None:
                return "Error: 'append' action requires 'text'."
            if not isinstance(text, str):
                 return "Error: 'text' for append must be a string."
            # Ensure text starts with a newline if the file doesn't end with one
            if lines and not lines[-1].endswith('\n') and not text.startswith('\n'):
                 text = '\n' + text
            # Ensure appended text ends with a newline for consistency
            if not text.endswith('\n'):
                text += '\n'
            lines.append(text)
            new_content = "".join(lines)
            modified = True

        elif action == "delete_line":
            line_number = params.get("line_number") # 1-based index
            if line_number is None:
                return "Error: 'delete_line' action requires 'line_number' (1-based)."
            try:
                line_idx = int(line_number) - 1 # Convert to 0-based index
            except ValueError:
                 return "Error: 'line_number' must be an integer."

            if line_idx < 0 or line_idx >= len(lines):
                return f"Error: Invalid 'line_number' {line_number}. File has {len(lines)} lines. Must be between 1 and {len(lines)}."
            del lines[line_idx]
            new_content = "".join(lines)
            modified = True

        else:
            return f"Error: Unknown action '{action}'. Supported actions: replace, insert, append, delete_line."

        # Write changes if modified
        if modified:
            # Optional: Create a backup before writing
            # backup_path = target_path.with_suffix(target_path.suffix + '.bak')
            # shutil.copy2(target_path, backup_path)
            # print(f"Created backup: {backup_path}")

            with open(target_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            return f"File '{target_path}' edited successfully (action: {action})."
        else:
            return f"File '{target_path}' not modified (action: {action}, no changes needed or text not found)."

    except json.JSONDecodeError:
        return f"Error: Invalid JSON format in instructions: {instructions}"
    except Exception as e:
        # Include filepath in the error message if available
        filepath_str = f" '{filepath}'" if filepath else ""
        return f"Error editing file{filepath_str}: {e}"


# API Call (Implementation)
try:
    import requests

    # Configure allowed domains (example - adjust as needed)
    ALLOWED_API_DOMAINS = {
        "api.example.com", # Allow specific domains
        "datausa.io",
        "httpbin.org", # For testing
        # Add other trusted domains here
    }
    ALLOW_ALL_DOMAINS = False # Set to True to disable domain checking (less secure)

    def _is_domain_allowed(url: str) -> bool:
        """Checks if the URL's domain is in the allowed list."""
        if ALLOW_ALL_DOMAINS:
            return True
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            if not domain:
                return False
            # Check against exact matches and subdomains (e.g., api.example.com allows sub.api.example.com)
            # This basic check might need refinement for complex cases
            return any(domain == allowed or domain.endswith('.' + allowed) for allowed in ALLOWED_API_DOMAINS)
        except Exception:
            return False # Deny if parsing fails

    def api_call(request_details: str) -> str:
        """
        Makes an API call based on provided JSON details to allowed domains.
        Input: JSON string specifying 'url', 'method' (GET/POST/PUT/DELETE etc.), optional 'params', 'headers', 'json_body'.
        Example: '{"url": "https://httpbin.org/get", "method": "GET", "params": {"id": 123}}'
        Example: '{"url": "https://httpbin.org/post", "method": "POST", "json_body": {"name": "John"}}'
        Returns the API response text/JSON (up to 4000 chars) or an error message.
        """
        print(f"--- Executing API Call Tool ---")
        print(f"Request Details: {request_details}")
        url = None # Initialize for error reporting
        try:
            params = json.loads(request_details)
            url = params.get("url")
            method = params.get("method", "GET").upper()
            req_params = params.get("params")
            headers = params.get("headers", {}) # Default to empty dict
            json_body = params.get("json_body") # For POST/PUT requests

            if not url:
                return "Error: 'url' is required in request details."
            if not isinstance(url, str):
                 return "Error: 'url' must be a string."

            # Security Check: Validate URL scheme and domain
            if not url.startswith(("http://", "https://")):
                return f"Error: Invalid URL scheme. Only http and https are allowed. URL: {url}"
            if not _is_domain_allowed(url):
                 return f"Error: Access denied. Domain for URL '{url}' is not in the allowed list: {ALLOWED_API_DOMAINS}"

            # Add a default User-Agent if not provided
            if 'User-Agent' not in headers:
                headers['User-Agent'] = 'AgentLLM/1.0 (Generic API Tool)'

            # Basic validation for other params
            if req_params is not None and not isinstance(req_params, dict):
                return "Error: 'params' must be a JSON object (dictionary)."
            if headers is not None and not isinstance(headers, dict):
                 return "Error: 'headers' must be a JSON object (dictionary)."
            # json_body can be various JSON types, requests handles it.

            response = requests.request(
                method=method,
                url=url,
                params=req_params,
                headers=headers,
                json=json_body, # Use json parameter for automatic serialization and content-type header
                timeout=30 # Add a timeout
            )

            # Raise an exception for bad status codes (4xx or 5xx)
            response.raise_for_status()

            print(f"API call to {url} ({method}) successful (Status: {response.status_code}).")

            # Limit response size returned to LLM
            MAX_RESPONSE_CHARS = 4000
            response_text = ""

            # Try to return JSON if possible, otherwise text
            try:
                response_json = response.json() # Return parsed JSON
                response_text = json.dumps(response_json, indent=2) # Pretty print JSON
            except json.JSONDecodeError:
                response_text = response.text # Return raw text

            if len(response_text) > MAX_RESPONSE_CHARS:
                 return response_text[:MAX_RESPONSE_CHARS] + f"\n... [truncated, total {len(response_text)} chars]"
            else:
                 return response_text

        except json.JSONDecodeError:
            return f"Error: Invalid JSON format in request_details: {request_details}"
        except requests.exceptions.RequestException as e:
            url_str = f" to {url}" if url else ""
            print(f"Error during API call{url_str}: {e}")
            # Try to include response body in error if available
            error_detail = str(e)
            if e.response is not None:
                # Limit error response size as well
                error_body = e.response.text[:500]
                error_detail += f"\nResponse Status: {e.response.status_code}\nResponse Body: {error_body}"
                if len(e.response.text) > 500:
                    error_detail += "..."
            return f"Error performing API call: {error_detail}"
        except Exception as e:
            url_str = f" to {url}" if url else ""
            print(f"Unexpected error during API call{url_str}: {e}")
            return f"Unexpected error during API call: {e}"

except ImportError:
    print("Warning: requests library not installed. API call tools will not be available.")
    print("Install using: pip install requests")

    def api_call(request_details: str) -> str:
        """Placeholder if requests is not installed."""
        print("--- API Call Tool (Not Available) ---")
        return "Error: API call tool requires 'requests' library."
    def _is_domain_allowed(url: str) -> bool: return False # Ensure placeholder exists


# Data USA API (Implementation using api_call)
def datausa_api(query_params: str) -> str:
    """
    Queries the Data USA API (datausa.io) using the generic api_call tool.
    Input: JSON string of query parameters, e.g., '{"measures": "Population", "geo": "nation/us"}'
           (data_source is often inferred by the API based on measures/geo).
    Returns the API response JSON or an error message.
    """
    print(f"--- Executing Data USA API Tool ---")
    print(f"Query Params: {query_params}")
    base_url = "https://datausa.io/api/data"
    try:
        # Check if requests library is available (needed by api_call)
        if 'requests' not in sys.modules:
             return "Error: Data USA API tool requires the 'requests' library to be installed for the underlying 'api_call' tool."

        # Parse the input string into a dictionary
        params_dict = json.loads(query_params)
        if not isinstance(params_dict, dict):
            return "Error: Input must be a JSON object (dictionary)."

        # Construct the request details for the generic api_call tool
        request_details = {
            "url": base_url,
            "method": "GET",
            "params": params_dict
        }
        request_details_str = json.dumps(request_details)

        # Use the existing api_call tool
        return api_call(request_details_str)

    except json.JSONDecodeError:
        return f"Error: Invalid JSON format in query_params: {query_params}"
    except Exception as e:
        return f"Error preparing Data USA API request: {e}"


# ... create_subgoal (Stub - Needs Agent Integration) ...
def create_subgoal(goal_description: str) -> str:
    """
    Registers a new sub-goal for the agent to potentially address later.
    Input is a description string of the sub-goal.
    (This requires integration with the agent's internal state or goal memory).
    """
    # TODO: Implement interaction with the agent's goal management system.
    # This might involve adding the goal to a list, database, or vector store.
    print(f"--- Executing Create Subgoal Tool (Stub) ---")
    if not isinstance(goal_description, str) or not goal_description.strip():
        return "Error: Goal description must be a non-empty string."
    print(f"Sub-goal Description: {goal_description}")
    # Example: Could potentially write to a goals file or call an internal agent method.
    # agent_state.add_goal(goal_description) # Hypothetical
    return f"Placeholder: Sub-goal '{goal_description}' registration requested. Requires agent state integration."


# ... request_human_input (Basic Implementation) ...
def request_human_input(question: str) -> str:
    """
    Pauses execution and asks the user (human) a specific question via the console.
    The agent will wait for the human's response to proceed.
    WARNING: This blocks execution. Use carefully in asynchronous agent loops.
    Input: The question string to ask the user.
    Returns the user's response string.
    """
    print(f"--- Executing Request Human Input Tool ---")
    if not isinstance(question, str) or not question.strip():
        return "Error: Question must be a non-empty string."
    print(f"Question for Human: {question}")
    try:
        # This uses standard input, which might block the agent's main loop
        # A more robust implementation might involve message queues or dedicated UI interaction.
        response = input(f"\n--- AGENT REQUEST ---\n{question}\nYour Input: ")
        print("--- Human response received ---")
        # Basic sanitization (optional)
        response = response.strip()
        return response
    except EOFError:
        # Handle cases where input stream is closed unexpectedly (e.g., running non-interactively)
        print("Warning: EOF received while waiting for human input. Returning empty response.")
        return "Error: No human input received (EOF)."
    except Exception as e:
        return f"Error getting human input: {e}"


def create_file(filepath_and_content: str) -> str:
    """
    Creates a new file with specified content within the agent's workspace.
    Input: JSON string format: '{"filepath": "path/new_file.txt", "content": "Initial content."}'
    Returns a success message or an error message.
    """
    print(f"--- Executing Create File Tool ---")
    print(f"Filepath and Content: {filepath_and_content}")
    filepath = None # Initialize for error reporting
    try:
        params = json.loads(filepath_and_content)
        filepath = params.get("filepath")
        content = params.get("content", "") # Default to empty content

        if not filepath:
            return "Error: Missing 'filepath' in input."
        if not isinstance(filepath, str):
             return "Error: 'filepath' must be a string."
        if not isinstance(content, str):
             return "Error: 'content' must be a string."

        # Resolve and validate path
        target_path = Path(filepath)
        if not target_path.is_absolute():
            if ".." in filepath:
                 return f"Error: Relative path cannot contain '..'. Path: '{filepath}'"
            target_path = WORKSPACE_DIR / target_path

        if not _is_path_safe(target_path):
            return f"Error: Access denied. Path '{filepath}' resolves outside the allowed workspace '{WORKSPACE_DIR}'."

        # Check if path already exists (could be file or directory)
        if target_path.exists():
            return f"Error: Path already exists at '{target_path}'. Use 'edit_file' to modify files or ensure the path is unique."
        # Check if parent exists and is a directory (mkdir will handle creation, but good practice)
        if target_path.parent.exists() and not target_path.parent.is_dir():
            return f"Error: Parent path '{target_path.parent}' exists but is not a directory."

        # Ensure parent directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the file
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Successfully created file: {target_path}")
        return f"File '{target_path}' created successfully."

    except json.JSONDecodeError:
        return f"Error: Invalid JSON format in input: {filepath_and_content}"
    except Exception as e:
        filepath_str = f" '{filepath}'" if filepath else ""
        return f"Error creating file{filepath_str}: {e}"


def create_directory(dirpath: str) -> str:
    """
    Creates a new directory at the specified path within the agent's workspace.
    Input: Directory path string (relative to workspace or absolute within workspace).
    Returns a success message or an error message.
    """
    print(f"--- Executing Create Directory Tool ---")
    print(f"Requested Directory Path: {dirpath}")
    try:
        if not isinstance(dirpath, str) or not dirpath.strip():
             return "Error: Directory path must be a non-empty string."

        # Resolve and validate path
        target_path = Path(dirpath)
        if not target_path.is_absolute():
            if ".." in dirpath:
                 return f"Error: Relative path cannot contain '..'. Path: '{dirpath}'"
            target_path = WORKSPACE_DIR / target_path

        if not _is_path_safe(target_path):
            return f"Error: Access denied. Path '{dirpath}' resolves outside the allowed workspace '{WORKSPACE_DIR}'."

        # Check if path exists and is a file
        if target_path.exists() and target_path.is_file():
            return f"Error: Path '{target_path}' exists and is a file, cannot create directory."

        # Create directory
        target_path.mkdir(parents=True, exist_ok=True) # exist_ok=True prevents error if dir exists

        print(f"Successfully created directory (or it already existed): {target_path}")
        return f"Directory '{target_path}' created or already exists."
    except Exception as e:
        return f"Error creating directory '{dirpath}': {e}"


def git_operation(command_details: str) -> str:
    """
    Executes a limited set of safe Git commands in a specified repository directory within the workspace.
    HIGH RISK: Command execution is inherently risky. Only allows specific safe commands.
    Input: JSON string format: '{"repo_path": "relative/path/to/repo", "command": "status"}'
           or '{"repo_path": "relative/path/to/repo", "command": "clone", "url": "https://..."}'
           or '{"repo_path": "relative/path/to/repo", "command": "pull"}'
           or '{"repo_path": "relative/path/to/repo", "command": "diff", "options": ["HEAD~1"]}' # Optional options list
           or '{"repo_path": "relative/path/to/repo", "command": "log", "options": ["-n", "5"]}'
           or '{"repo_path": "relative/path/to/repo", "command": "branch"}'
           or '{"repo_path": "relative/path/to/repo", "command": "checkout", "options": ["main"]}' # Be careful with checkout
    Allowed commands: status, clone, pull, diff, log, branch, checkout (use checkout with caution).
    Returns the command output (stdout/stderr, up to 4000 chars) or an error message.
    """
    # IMPORTANT: VERY HIGH RISK. Needs strict validation and sandboxing.
    print(f"--- Executing Git Operation Tool ---")
    print(f"Command Details: {command_details}")

    # Define allowed commands and potentially risky options to scrutinize
    ALLOWED_GIT_COMMANDS = {"status", "clone", "pull", "diff", "log", "branch", "checkout"}
    # Basic check for potentially dangerous option patterns (can be expanded)
    DANGEROUS_OPTION_PATTERNS = ["--exec", "--output", "-o", "|", ";", "&", "`", "$("]

    repo_path_str = None # Initialize for error reporting
    try:
        params = json.loads(command_details)
        repo_path_str = params.get("repo_path")
        command = params.get("command")
        options = params.get("options", []) # Optional list of arguments for the command
        clone_url = params.get("url") # Only used for clone

        # --- Input Validation ---
        if not repo_path_str or not command:
            return "Error: Missing 'repo_path' or 'command' in details."
        if not isinstance(repo_path_str, str):
             return "Error: 'repo_path' must be a string."
        if not isinstance(command, str):
             return "Error: 'command' must be a string."
        if not isinstance(options, list):
             return "Error: 'options' must be a list of strings."
        if command == "clone" and (not clone_url or not isinstance(clone_url, str)):
             return "Error: 'clone' command requires a string 'url'."

        if command not in ALLOWED_GIT_COMMANDS:
            return f"Error: Git command '{command}' is not allowed. Allowed: {', '.join(ALLOWED_GIT_COMMANDS)}"

        # Validate options against dangerous patterns
        for opt in options:
            if not isinstance(opt, str):
                 return f"Error: All items in 'options' must be strings. Found: {type(opt)}"
            for pattern in DANGEROUS_OPTION_PATTERNS:
                if pattern in opt:
                    return f"Error: Potentially dangerous pattern '{pattern}' found in git options: {opt}"
        # Specific check for checkout: prevent checking out paths or detached HEADs easily
        if command == "checkout" and any(opt.startswith('-') or '/' in opt or '\\' in opt for opt in options):
             return "Error: 'checkout' options cannot start with '-' or contain path separators."


        # --- Path Resolution and Validation ---
        repo_path = Path(repo_path_str)
        if not repo_path.is_absolute():
            if ".." in repo_path_str:
                 return f"Error: Relative repo_path cannot contain '..'. Path: '{repo_path_str}'"
            repo_path = WORKSPACE_DIR / repo_path

        # For 'clone', the target directory must be within the workspace, but might not exist yet.
        # The parent directory must be safe.
        if command == "clone":
            # Check parent first
            if not _is_path_safe(repo_path.parent):
                 return f"Error: Access denied. Clone destination parent '{repo_path.parent}' is outside the allowed workspace '{WORKSPACE_DIR}'."
            # Check the target directory itself resolves safely
            if not _is_path_safe(repo_path):
                 return f"Error: Access denied. Clone destination '{repo_path}' resolves outside the allowed workspace '{WORKSPACE_DIR}'."
            # Check if path exists and is not an empty directory (git clone requires empty or non-existent)
            if repo_path.exists() and repo_path.is_file():
                 return f"Error: Clone destination '{repo_path}' exists and is a file."
            if repo_path.exists() and any(repo_path.iterdir()):
                 return f"Error: Clone destination '{repo_path}' exists and is not empty."

            # Basic URL validation
            if not clone_url.startswith(("http://", "https://", "git@")):
                 return f"Error: Invalid or potentially unsafe clone URL scheme: {clone_url}. Only http(s) or git@ allowed."

            # Ensure repo_path exists for clone command execution context (will clone *into* it)
            repo_path.mkdir(parents=True, exist_ok=True)
            # Construct command: git clone <url> <options> . (clone into the target dir)
            cmd_list = ["git", "clone", clone_url] + options + ["."] # Clone into the specified repo_path
            cwd = repo_path # Execute 'git clone' *in* the target directory

        else:
            # For other commands, the repo must exist and be a safe directory
            if not _is_path_safe(repo_path):
                return f"Error: Access denied. Path '{repo_path_str}' resolves outside the allowed workspace '{WORKSPACE_DIR}'."
            if not repo_path.is_dir():
                return f"Error: Repository path '{repo_path}' does not exist or is not a directory."
            # Check if it looks like a git repo (basic check)
            if not (repo_path / ".git").is_dir():
                 return f"Error: Directory '{repo_path}' does not appear to be a Git repository (missing .git subdir)."

            # Construct command list safely
            cmd_list = ["git", command] + options
            cwd = repo_path # Execute command within the repo directory

        # --- Execution ---
        # Using shlex.quote is generally good, but git commands often handle paths, so direct list is okay here
        # given the prior validation. Double-check if allowing complex options.
        print(f"Executing command: {' '.join(cmd_list)} in {cwd}")

        # Execute using subprocess
        process = subprocess.run(
            cmd_list,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding='utf-8', # Specify encoding
            errors='ignore', # Ignore decoding errors in output
            check=False, # Don't raise exception on non-zero exit code, handle below
            timeout=120 # Increase timeout for potentially long operations like clone/pull
        )

        # Combine stdout and stderr for the result
        full_output = ""
        if process.stdout:
            full_output += f"STDOUT:\n{process.stdout}\n"
        if process.stderr:
            # Don't treat warnings on stderr as critical errors for certain commands (like pull/status)
            # but still include them in the output.
            full_output += f"STDERR:\n{process.stderr}\n"

        # Limit output size
        MAX_GIT_OUTPUT = 4000
        if len(full_output) > MAX_GIT_OUTPUT:
            output = full_output[:MAX_GIT_OUTPUT] + f"\n... [truncated, total {len(full_output)} chars]"
        else:
            output = full_output.strip() # Remove trailing newline if any

        result_prefix = f"--- Git {command} Result (in {repo_path}) ---\n"
        final_output = result_prefix + output

        if process.returncode != 0:
            print(f"Git command failed with return code {process.returncode}")
            # Return the potentially truncated output including stderr for debugging
            return f"Error executing Git command (code {process.returncode}):\n{final_output}"
        else:
            print(f"Git command executed successfully.")
            return final_output

    except json.JSONDecodeError:
        return f"Error: Invalid JSON format in command_details: {command_details}"
    except subprocess.TimeoutExpired:
        return f"Error: Git command timed out after 120 seconds."
    except FileNotFoundError:
        return "Error: 'git' command not found. Is Git installed and in the system PATH?"
    except Exception as e:
        repo_path_info = f" for repo '{repo_path_str}'" if repo_path_str else ""
        return f"Error executing Git operation{repo_path_info}: {e}"


# ... list_tools ...
def list_tools() -> str:
    """Lists the names and descriptions of all available tools."""
    print(f"--- Executing List Tools Tool ---")
    available = get_available_tools_list()
    # Use ensure_ascii=False for better readability if descriptions contain non-ASCII
    try:
        return json.dumps(available, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error encoding tool list to JSON: {e}")
        # Fallback to simpler representation
        return str(available)


# ... prioritize_goals (Stub - Needs LLM/Agent Integration) ...
def prioritize_goals(goal_list_and_context: str) -> str:
    """
    Analyzes a list of goals and suggests a priority order based on context.
    Input: JSON string format: '{"goals": ["goal A", "goal B"], "context": "Urgent deadline approaching"}'
    (This likely requires an LLM call internally and integration with goal management).
    """
    # TODO: Implement logic, potentially calling an LLM. Needs integration with goal management.
    print(f"--- Executing Prioritize Goals Tool (Stub) ---")
    print(f"Input: {goal_list_and_context}")
    try:
        data = json.loads(goal_list_and_context)
        if not isinstance(data.get("goals"), list) or not isinstance(data.get("context"), str):
             return "Error: Invalid input format. Expected JSON with 'goals' (list) and 'context' (string)."
        # Needs implementation: Parse input, construct prompt, call LLM, parse response.
        return f"Placeholder: Goal prioritization based on context '{data['context']}' requested. Needs LLM/Agent integration."
    except json.JSONDecodeError:
        return f"Error: Invalid JSON format in input: {goal_list_and_context}"
    except Exception as e:
        return f"Error processing prioritize_goals request: {e}"


# ... manage_goals (Stub - Needs Agent Integration) ...
def manage_goals(action_details: str) -> str:
    """
    Manages the agent's goals (add, remove, update status).
    Input: JSON string format: '{"action": "update", "goal_id": "goal_123", "status": "achieved"}'
           or '{"action": "add", "description": "New goal C"}'
           or '{"action": "remove", "goal_id": "goal_456"}'
    (Requires integration with the agent's goal memory/state, e.g., Qdrant).
    """
    # TODO: Implement interaction with goal memory (e.g., Qdrant goal_memory collection).
    print(f"--- Executing Manage Goals Tool (Stub) ---")
    print(f"Action Details: {action_details}")
    try:
        params = json.loads(action_details)
        action = params.get("action")
        if not action:
            return "Error: Missing 'action' in details."
        # Basic validation based on action
        if action == "add" and not params.get("description"):
            return "Error: 'add' action requires 'description'."
        if action in ["update", "remove"] and not params.get("goal_id"):
            return f"Error: '{action}' action requires 'goal_id'."
        if action == "update" and not params.get("status"):
             return "Error: 'update' action requires 'status'."

        # Needs implementation: Parse details, interact with goal storage system.
        return f"Placeholder: Goal management action '{action_details}' requested. Needs Agent integration."
    except json.JSONDecodeError:
        return f"Error: Invalid JSON format in action_details: {action_details}"
    except Exception as e:
        return f"Error processing manage_goals request: {e}"


# ... request_tool_enhancement (Basic Implementation) ...
def request_tool_enhancement(request_description: str) -> str:
    """
    Logs a request for a new tool or enhancement to an existing one.
    Input is a description string of the needed capability. Logs to a file in the workspace.
    """
    print(f"--- Executing Request Tool Enhancement Tool ---")
    if not isinstance(request_description, str) or not request_description.strip():
        return "Error: Request description must be a non-empty string."
    print(f"Enhancement Request: {request_description}")
    log_file = WORKSPACE_DIR / "tool_enhancement_requests.log"
    try:
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        log_entry = f"{timestamp}: {request_description}\n"
        # Ensure workspace exists (should already, but belt-and-suspenders)
        WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)
        return f"Enhancement request logged to {log_file}."
    except Exception as e:
        return f"Error logging enhancement request: {e}"


# --- New Tool Stubs for Autonomy (Stubs - Need Implementation/Integration) ---

def monitor_environment(trigger_conditions: str) -> str:
    """
    Checks predefined triggers or monitors specific conditions. (STUB)
    Input: JSON string format example: '{"type": "file_change", "path": "path/to/watch/file.txt"}'
           or '{"type": "api_check", "url": "http://status.example.com", "expected_value": "OK"}'
           or '{"type": "time_schedule", "cron": "0 9 * * MON"}' # Check if it's time

    Returns a description of triggered conditions or 'No triggers activated'.
    (Requires complex implementation involving scheduling, file watching, API calls etc.)
    """
    # TODO: Implement logic to check various trigger types based on parsed 'trigger_conditions'.
    # This is complex and likely needs a dedicated monitoring loop or integration with external services.
    # Libraries like 'watchdog' (file system), 'schedule' or 'croniter' (time), 'requests' (API) could be used.
    # Path validation (_is_path_safe) MUST be used for file-based triggers.
    # Domain validation (_is_domain_allowed) MUST be used for API-based triggers.
    print(f"--- Executing Monitor Environment Tool (Stub) ---")
    print(f"Trigger Conditions to Check: {trigger_conditions}")
    try:
        conditions = json.loads(trigger_conditions)
        # Basic validation
        if not isinstance(conditions, dict) or "type" not in conditions:
             return "Error: Invalid format for trigger_conditions. Expected JSON object with 'type'."
        # Needs implementation to parse conditions and perform checks safely.
        return f"Placeholder: Environment monitoring based on '{trigger_conditions}' requested. Needs complex implementation."
    except json.JSONDecodeError:
        return f"Error: Invalid JSON format in trigger_conditions: {trigger_conditions}"
    except Exception as e:
        return f"Error processing monitor_environment request: {e}"


def generate_hypothesis(context: str) -> str:
    """
    Analyzes the current situation and capabilities to propose potential new goals or opportunities for proactive action. (STUB)
    Input: A summary string of the current context (e.g., recent observations, available tools, idle status).
    (This requires an LLM call internally).
    """
    # TODO: Implement using an LLM (e.g., main_llm).
    # Prompt should guide the LLM based on agent's purpose, tools, and state.
    print(f"--- Executing Generate Hypothesis Tool (Stub) ---")
    if not isinstance(context, str):
        return "Error: Context must be a string."
    print(f"Context for Hypothesis: {context[:200]}...") # Log truncated context
    # Needs implementation: Construct prompt, call LLM, parse response.
    # Example Prompt Idea: "Given my tools ({tool_list}), recent activity ({context}), and my goal to be proactively helpful, suggest one specific, actionable task I could undertake."
    return f"Placeholder: Hypothesis/opportunity generation based on provided context requested. Needs LLM implementation."


def evaluate_self_performance(recent_activity_summary: str) -> str:
    """
    Performs self-reflection on recent actions, focusing on autonomy, efficiency, and goal achievement. (STUB)
    Input: A summary string of recent goals, actions, and outcomes.
    (This requires an LLM call internally).
    """
    # TODO: Implement using an LLM (e.g., main_llm).
    # Prompt should ask specific questions about proactivity, efficiency, tool use, goal success, reliance on human input.
    print(f"--- Executing Evaluate Self Performance Tool (Stub) ---")
    if not isinstance(recent_activity_summary, str):
        return "Error: Recent activity summary must be a string."
    print(f"Summary for Self-Evaluation: {recent_activity_summary[:200]}...") # Log truncated summary
    # Needs implementation: Construct prompt, call LLM, parse response.
    # The result could be stored in semantic_memory or a dedicated reflection memory.
    # Example Prompt Idea: "Evaluate the agent's performance based on the following activity: {recent_activity_summary}. Assess proactivity, efficiency, goal success, and reliance on human input. Provide key learnings."
    return f"Placeholder: Self-performance evaluation based on provided summary requested. Needs LLM implementation."


# --- Tool Registry (Updated with potential missing libraries) ---
import sys # Needed for checking module availability

# Simple dictionary mapping tool names to their functions
# Check for library availability before adding tools that depend on them
AVAILABLE_TOOLS: Dict[str, Callable[..., str]] = {}

# Web Search
if 'duckduckgo_search' in sys.modules:
    AVAILABLE_TOOLS["web_search"] = web_search
else:
    AVAILABLE_TOOLS["web_search"] = lambda *args, **kwargs: "Error: Web search tool requires 'duckduckgo-search' library."

# ArXiv Search
if 'arxiv' in sys.modules:
    AVAILABLE_TOOLS["arxiv_search"] = arxiv_search
else:
    AVAILABLE_TOOLS["arxiv_search"] = lambda *args, **kwargs: "Error: ArXiv search tool requires 'arxiv' library."

# Wikipedia Search
if 'wikipedia' in sys.modules:
    AVAILABLE_TOOLS["wikipedia_search"] = wikipedia_search
else:
    AVAILABLE_TOOLS["wikipedia_search"] = lambda *args, **kwargs: "Error: Wikipedia search tool requires 'wikipedia' library."

# File Operations (Built-in, but depend on WORKSPACE_DIR logic)
AVAILABLE_TOOLS["read_file"] = read_file
AVAILABLE_TOOLS["edit_file"] = edit_file  # HIGH RISK
AVAILABLE_TOOLS["create_file"] = create_file  # HIGH RISK
AVAILABLE_TOOLS["create_directory"] = create_directory

# API Calls
if 'requests' in sys.modules:
    AVAILABLE_TOOLS["api_call"] = api_call
    AVAILABLE_TOOLS["datausa_api"] = datausa_api
else:
    AVAILABLE_TOOLS["api_call"] = lambda *args, **kwargs: "Error: API call tool requires 'requests' library."
    AVAILABLE_TOOLS["datausa_api"] = lambda *args, **kwargs: "Error: Data USA API tool requires 'requests' library."

# Git Operations (Requires git executable)
# We can't easily check for the executable here, so assume it might be available
# The tool itself handles the FileNotFoundError if 'git' isn't callable.
AVAILABLE_TOOLS["git_operation"] = git_operation # HIGH RISK

# Agent State / Human Interaction / Meta Tools (Built-in or Stubs)
AVAILABLE_TOOLS["create_subgoal"] = create_subgoal # Stub - Needs Agent Integration
AVAILABLE_TOOLS["request_human_input"] = request_human_input # Built-in (basic console input)
AVAILABLE_TOOLS["list_tools"] = list_tools # Built-in
AVAILABLE_TOOLS["prioritize_goals"] = prioritize_goals # Stub - Needs LLM/Agent Integration
AVAILABLE_TOOLS["manage_goals"] = manage_goals # Stub - Needs Agent Integration
AVAILABLE_TOOLS["request_tool_enhancement"] = request_tool_enhancement # Built-in (logs to file)

# New autonomy tools (Stubs):
AVAILABLE_TOOLS["monitor_environment"] = monitor_environment # Stub - Needs complex implementation
AVAILABLE_TOOLS["generate_hypothesis"] = generate_hypothesis # Stub - Needs LLM implementation
AVAILABLE_TOOLS["evaluate_self_performance"] = evaluate_self_performance # Stub - Needs LLM implementation


# --- Tool Description and Execution Logic ---

def get_tool_description(tool_name: str) -> str:
    """Returns the docstring (description) of a tool."""
    tool_func = AVAILABLE_TOOLS.get(tool_name)
    if tool_func and tool_func.__doc__:
        # Add input format hints from the docstring if available
        doc = tool_func.__doc__.strip()
        # Add a note about risk level if applicable
        if "_RISK" in tool_name.upper() or tool_name in ["edit_file", "create_file", "git_operation"]:
             doc += "\n   [Security Risk: High - Use with caution and specific instructions]"
        elif tool_name in ["api_call", "create_directory", "read_file"]:
             doc += "\n   [Security Risk: Medium - Operates within workspace/allowed domains]"

        # Add note about dependencies if the function is a placeholder
        if "Error:" in tool_func("") if not inspect.signature(tool_func).parameters else tool_func("test"): # Basic check if it's a placeholder
             if "requires" in tool_func(""):
                  doc += f"\n   [Note: Requires external library/setup - {tool_func('')}]"

        return doc
    return "No description available or tool not found."


def get_available_tools_list() -> Dict[str, str]:
    """Returns a dictionary of available tool names and their descriptions."""
    return {name: get_tool_description(name) for name in AVAILABLE_TOOLS}


def execute_tool(action_str: str) -> Dict[str, Any]:
    """
    Parses an action string and executes the corresponding tool.

    Expected action_str format: "TOOL_NAME[argument]"
    Argument MUST be a single string. If the tool expects JSON, the string must BE the JSON.
    Example: web_search[latest AI news]
    Example: edit_file[{"filepath": "file.txt", "action": "append", "text": "more data"}]
    Example: list_tools[] or just list_tools

    Returns:
        A dictionary containing:
        - 'tool_name': The name of the tool called.
        - 'argument': The raw argument string passed to the tool.
        - 'result': The result from the tool execution (string, potentially JSON formatted).
        - 'error': An error message string if execution failed, otherwise None.
    """
    print(f"Attempting to parse and execute action: {action_str}")
    tool_name = ""
    argument = "" # The raw string argument
    error = None
    result = None # Will store the raw result from the tool

    try:
        action_str = action_str.strip()
        # Parsing: Find first '[' and last ']'
        if "[" in action_str and action_str.endswith("]"):
            split_index = action_str.find("[")
            tool_name = action_str[:split_index].strip()
            # Argument is everything between the first '[' and the last ']'
            argument = action_str[split_index + 1 : -1] # Keep as raw string
            # Basic check for common LLM hallucination of nested calls:
            if "[" in argument and "]" in argument and any(t in argument for t in AVAILABLE_TOOLS):
                 error = f"Error: Detected nested tool call structure within argument '{argument}'. Arguments must be plain strings or valid JSON strings, not nested calls."
                 print(error)

        else:
            # Handle tools that might not need arguments (or called without brackets)
            if action_str in AVAILABLE_TOOLS:
                tool_name = action_str
                argument = ""  # No argument provided
            else:
                # Check if it looks like a tool name but has no brackets
                potential_tool = action_str.split('[')[0].strip()
                if potential_tool in AVAILABLE_TOOLS:
                     error = f"Invalid action format. Tool '{potential_tool}' found, but missing brackets '[]' around argument. Use '{potential_tool}[argument]' or '{potential_tool}[]' if no argument is needed."
                else:
                     error = f"Invalid action format or unknown tool. Expected 'TOOL_NAME[argument]' or 'TOOL_NAME', got '{action_str}'."
                print(error)

        # If no parsing error so far, proceed to execution
        if not error and tool_name:
            if tool_name in AVAILABLE_TOOLS:
                tool_function = AVAILABLE_TOOLS[tool_name]
                print(f"Executing tool '{tool_name}' with raw argument string: '{argument[:100]}{'...' if len(argument)>100 else ''}'")

                # Inspect the tool function signature
                sig = inspect.signature(tool_function)
                params = list(sig.parameters.values())
                num_params = len(params)
                takes_arg = num_params > 0

                # Check if an argument is required vs optional
                is_required = False
                if takes_arg:
                    first_param = params[0]
                    is_required = first_param.default == inspect.Parameter.empty

                # Validate argument presence/absence
                if is_required and argument == "":
                    error = f"Tool '{tool_name}' requires an argument, but none was provided inside the brackets []."
                    print(error)
                elif not takes_arg and argument != "":
                    # Allow tools that take no args to be called with empty brackets TOOL[]
                    if argument: # Only warn if non-empty argument provided
                        print(f"Warning: Tool '{tool_name}' does not take arguments, but '{argument}' was provided. Ignoring argument.")
                    result = tool_function() # Call without argument
                elif takes_arg:
                    # Pass the raw argument string. The tool itself handles parsing (e.g., JSON).
                    # Handle tools with multiple args (like web_search(query, max_results)) - basic attempt
                    # This assumes the LLM provides args correctly within the single string if needed,
                    # or the tool primarily uses the first arg.
                    # A more robust solution might involve the LLM specifying args by name in JSON.
                    if num_params == 1 or (num_params > 1 and params[1].default != inspect.Parameter.empty):
                        # If only one param, or subsequent params have defaults, pass the single arg string.
                        result = tool_function(argument)
                    elif num_params == 2 and params[0].annotation is str and params[1].annotation is int:
                         # Special case for (query: str, max_results: int = 3) pattern
                         # Try to pass both if possible, otherwise just the query string
                         # This is brittle; ideally the tool handles the single string arg robustly.
                         # For now, just pass the main argument string. Let the tool handle it.
                         result = tool_function(argument) # Tool needs to parse max_results from string if needed
                    else:
                         # Fallback for other multi-arg functions: pass the single string argument.
                         result = tool_function(argument)

                else: # Tool takes no arguments, and none was provided
                     result = tool_function()

                if error is None:
                    print(f"Tool '{tool_name}' execution finished.")
                    # Ensure result is a string or None before returning
                    if result is not None and not isinstance(result, str):
                        try:
                            # If it's dict/list, format as JSON string
                            if isinstance(result, (dict, list)):
                                result = json.dumps(result, indent=2, ensure_ascii=False)
                            else:
                                # Convert other types (int, bool, etc.) to string
                                result = str(result)
                        except Exception as json_e:
                            print(f"Warning: Could not serialize tool result to JSON: {json_e}")
                            result = f"Error: Could not serialize tool result of type {type(result)}."


            else: # Tool name parsed but not found in registry
                error = f"Tool '{tool_name}' not found in available tools."
                print(error)

    except Exception as e:
        # Catch-all for unexpected errors during parsing or execution logic
        error_msg = f"Unexpected error during tool parsing/execution for action '{action_str}': {e}"
        import traceback
        print(f"{error_msg}\n{traceback.format_exc()}") # Print stack trace for debugging
        error = error_msg
        result = None  # Ensure result is None on error

    # Final structure
    return {
        "tool_name": tool_name if tool_name else "N/A",
        "argument": argument,  # Store the raw argument string received
        "result": result, # Store the result (should be string or None)
        "error": error,   # Store error message (string or None)
    }


# --- Example Usage (Updated) ---
if __name__ == "__main__":
    print("--- Initializing Workspace and Tools ---")
    print(f"Workspace directory: {WORKSPACE_DIR}")
    print("\nAvailable Tools:")
    print(json.dumps(get_available_tools_list(), indent=2, ensure_ascii=False))

    print("\n--- Testing Tool Execution ---")

    # Ensure workspace exists for tests
    WORKSPACE_DIR.mkdir(exist_ok=True)
    test_dir_path = WORKSPACE_DIR / "test_dir_tools"
    test_repo_path_str = "test_repo_tools" # Relative path string
    test_repo_path = WORKSPACE_DIR / test_repo_path_str # Path object

    # Clean up previous test runs first
    import shutil
    if test_dir_path.exists():
        print(f"Cleaning up previous test directory: {test_dir_path}")
        shutil.rmtree(test_dir_path)
    if test_repo_path.exists():
         print(f"Cleaning up previous test repository: {test_repo_path}")
         shutil.rmtree(test_repo_path)

    # --- Test Cases ---
    test_cases = [
        # --- Basic Functionality ---
        {"name": "List Tools", "action": "list_tools", "skip_if_no_libs": False},
        {"name": "List Tools (with brackets)", "action": "list_tools[]", "skip_if_no_libs": False},
        # --- Web Search ---
        {"name": "Web Search (DDG)", "action": "web_search[latest news about large language models]", "skip_if_no_libs": 'duckduckgo_search' not in sys.modules},
        # --- ArXiv Search ---
        {"name": "ArXiv Search", "action": "arxiv_search[explainable AI techniques]", "skip_if_no_libs": 'arxiv' not in sys.modules},
        # --- Wikipedia Search ---
        {"name": "Wikipedia Search", "action": "wikipedia_search[Alan Turing]", "skip_if_no_libs": 'wikipedia' not in sys.modules},
        {"name": "Wikipedia Search (Ambiguous)", "action": "wikipedia_search[Python]", "skip_if_no_libs": 'wikipedia' not in sys.modules},
        # --- Filesystem Operations ---
        {"name": "Create Directory", "action": f"create_directory[{test_dir_path.name}]", "skip_if_no_libs": False},
        {"name": "Create File", "action": f'create_file[{{"filepath": "{test_dir_path.name}/my_test_file.txt", "content": "Hello from test case!"}}]', "skip_if_no_libs": False},
        {"name": "Read File", "action": f"read_file[{test_dir_path.name}/my_test_file.txt]", "skip_if_no_libs": False},
        {"name": "Edit File (Append)", "action": f'edit_file[{{"filepath": "{test_dir_path.name}/my_test_file.txt", "action": "append", "text": "\\nSecond line added."}}]', "skip_if_no_libs": False},
        {"name": "Read File (After Append)", "action": f"read_file[{test_dir_path.name}/my_test_file.txt]", "skip_if_no_libs": False},
        {"name": "Edit File (Replace)", "action": f'edit_file[{{"filepath": "{test_dir_path.name}/my_test_file.txt", "action": "replace", "old_text": "Second line", "new_text": "Third line", "count": 1}}]', "skip_if_no_libs": False},
        {"name": "Read File (After Replace)", "action": f"read_file[{test_dir_path.name}/my_test_file.txt]", "skip_if_no_libs": False},
        {"name": "Edit File (Insert)", "action": f'edit_file[{{"filepath": "{test_dir_path.name}/my_test_file.txt", "action": "insert", "line_number": 1, "text": "Inserted first line.\\n"}}]', "skip_if_no_libs": False},
        {"name": "Read File (After Insert)", "action": f"read_file[{test_dir_path.name}/my_test_file.txt]", "skip_if_no_libs": False},
        {"name": "Edit File (Delete Line)", "action": f'edit_file[{{"filepath": "{test_dir_path.name}/my_test_file.txt", "action": "delete_line", "line_number": 2}}]', "skip_if_no_libs": False},
        {"name": "Read File (After Delete)", "action": f"read_file[{test_dir_path.name}/my_test_file.txt]", "skip_if_no_libs": False},
        # --- API Calls ---
        {"name": "API Call (GET)", "action": 'api_call[{"url": "https://httpbin.org/get", "method": "GET", "params": {"test": "true"}}]', "skip_if_no_libs": 'requests' not in sys.modules},
        {"name": "API Call (POST)", "action": 'api_call[{"url": "https://httpbin.org/post", "method": "POST", "json_body": {"value": 42}}]', "skip_if_no_libs": 'requests' not in sys.modules},
        {"name": "DataUSA API", "action": 'datausa_api[{"measures": "Population", "geo": "nation/us"}]', "skip_if_no_libs": 'requests' not in sys.modules},
        # --- Git Operations (Requires git executable in PATH) ---
        {"name": "Git Clone", "action": f'git_operation[{{"repo_path": "{test_repo_path_str}", "command": "clone", "url": "https://github.com/pallets/flask.git"}}]', "skip_if_no_libs": False}, # Assumes git is installed
        {"name": "Git Status", "action": f'git_operation[{{"repo_path": "{test_repo_path_str}", "command": "status"}}]', "skip_if_no_libs": False}, # Depends on clone success
        {"name": "Git Log", "action": f'git_operation[{{"repo_path": "{test_repo_path_str}", "command": "log", "options": ["-n", "2", "--oneline"]}}]', "skip_if_no_libs": False}, # Depends on clone success
        {"name": "Git Checkout (Branch)", "action": f'git_operation[{{"repo_path": "{test_repo_path_str}", "command": "checkout", "options": ["main"]}}]', "skip_if_no_libs": False}, # Depends on clone success
        # --- Agent/Human Interaction ---
        {"name": "Create Subgoal (Stub)", "action": "create_subgoal[Plan the next phase of testing]", "skip_if_no_libs": False},
        # {"name": "Request Human Input", "action": "request_human_input[Please enter 'test input' below:]", "skip_if_no_libs": False}, # Uncomment to test interactively
        {"name": "Request Tool Enhancement", "action": "request_tool_enhancement[Need a tool to summarize PDF documents]", "skip_if_no_libs": False},
        # --- Autonomy Stubs ---
        {"name": "Monitor Environment (Stub)", "action": 'monitor_environment[{"type": "time_schedule", "cron": "0 0 * * *"}]', "skip_if_no_libs": False},
        {"name": "Generate Hypothesis (Stub)", "action": "generate_hypothesis[Agent is idle, last task completed successfully.]", "skip_if_no_libs": False},
        {"name": "Evaluate Performance (Stub)", "action": "evaluate_self_performance[Completed goal 'research X', used web_search and read_file, took 3 steps.]", "skip_if_no_libs": False},
        # --- Error Handling ---
        {"name": "Non-existent Tool", "action": "non_existent_tool[some query]", "skip_if_no_libs": False},
        {"name": "Invalid Format (No Brackets)", "action": "read_file", "skip_if_no_libs": False},
        {"name": "Invalid Format (Bad Brackets)", "action": "read_file(test.txt)", "skip_if_no_libs": False},
        {"name": "Tool Requiring Arg (Missing)", "action": "read_file[]", "skip_if_no_libs": False},
        {"name": "Tool Not Requiring Arg (Provided)", "action": "list_tools[some argument]", "skip_if_no_libs": False},
        {"name": "Path Safety Violation (Read)", "action": "read_file[../outside_file.txt]", "skip_if_no_libs": False},
        {"name": "Path Safety Violation (Create Dir)", "action": "create_directory[../../unsafe_dir]", "skip_if_no_libs": False},
        {"name": "Path Safety Violation (Git Clone Parent)", "action": 'git_operation[{"repo_path": "../unsafe_repo", "command": "clone", "url": "https://example.com/repo.git"}]', "skip_if_no_libs": False},
        {"name": "Edit File (Invalid JSON)", "action": 'edit_file[this is not json]', "skip_if_no_libs": False},
        {"name": "API Call (Invalid Domain)", "action": 'api_call[{"url": "https://malicious.example.net/data"}]', "skip_if_no_libs": 'requests' not in sys.modules},
        {"name": "Git Operation (Disallowed Command)", "action": f'git_operation[{{"repo_path": "{test_repo_path_str}", "command": "push"}}]', "skip_if_no_libs": False},
        {"name": "Git Operation (Dangerous Option)", "action": f'git_operation[{{"repo_path": "{test_repo_path_str}", "command": "log", "options": ["--exec=echo hacked"]}}]', "skip_if_no_libs": False},
        {"name": "Nested Tool Call (Hallucination)", "action": "read_file[web_search[find filename]]", "skip_if_no_libs": False},
    ]

    results_summary = {}

    for test in test_cases:
        name = test["name"]
        action = test["action"]
        skip = test["skip_if_no_libs"]

        print(f"\n--- Running Test: {name} ---")
        print(f"Action: {action}")

        if skip:
            print("Skipping test due to missing library dependencies.")
            results_summary[name] = "SKIPPED"
            continue

        # Special handling for git status/log which depend on clone success
        if name in ["Git Status", "Git Log", "Git Checkout (Branch)"] and results_summary.get("Git Clone") != "PASSED":
             print("Skipping test because prerequisite 'Git Clone' did not pass or was skipped.")
             results_summary[name] = "SKIPPED (Prereq Failed)"
             continue

        output = execute_tool(action)
        print(f"Output:\n{output}")

        # Basic pass/fail check (can be made more sophisticated)
        passed = output.get("error") is None
        # Specific checks for errors we expect
        if name.startswith("Path Safety") or name.startswith("Non-existent") or name.startswith("Invalid Format") or \
           name.startswith("Tool Requiring Arg (Missing)") or name.startswith("API Call (Invalid Domain)") or \
           name.startswith("Git Operation (Disallowed") or name.startswith("Git Operation (Dangerous") or \
           name.startswith("Nested Tool Call"):
            passed = output.get("error") is not None # Expect an error for these
        elif name == "Tool Not Requiring Arg (Provided)":
             passed = output.get("error") is None # Expect no error, just a warning maybe

        results_summary[name] = "PASSED" if passed else "FAILED"
        print(f"Result: {results_summary[name]}")


    print("\n\n--- Test Summary ---")
    for name, result in results_summary.items():
        print(f"{name:<40}: {result}")

    # Optional: Final cleanup
    # print("\n--- Cleaning up test artifacts ---")
    # if test_dir_path.exists():
    #     shutil.rmtree(test_dir_path)
    #     print(f"Removed: {test_dir_path}")
    # if test_repo_path.exists():
    #      shutil.rmtree(test_repo_path)
    #      print(f"Removed: {test_repo_path}")
    # enhancement_log = WORKSPACE_DIR / "tool_enhancement_requests.log"
    # if enhancement_log.exists():
    #      enhancement_log.unlink()
    #      print(f"Removed: {enhancement_log}")
