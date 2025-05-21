import os
import uuid
import json  # Import json for potential result parsing
import re  # Import re for parsing reflection
import datetime  # Import datetime for timestamps
from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import logging
import threading
import time

# Import utilities
from qdrant_utils import (
    get_qdrant_client,
    initialize_qdrant_collections,
    add_memory,
    search_memory,
)
from llm_utils import (
    get_main_model,  # Import main model getter
    get_assistant_model,  # Import assistant model getter
    generate_text,
    generate_embeddings,
    MAIN_MODEL_NAME,  # Import the constant
    ASSISTANT_MODEL_NAME,  # Import the constant
)

# Import tool utilities
from tools import execute_tool, get_available_tools_list

# Load environment variables (especially for LangSmith)
load_dotenv()

# Create a logger instance for this file
logger = logging.getLogger(__name__)

# --- Global Agent State for API ---
_current_agent_state: Dict[str, Any] = {
    "status": "initializing",
    "current_node": "N/A",
    "active_goal": "N/A",
    "current_plan": [],
    "executed_actions": [],
    "reflection_insights": [],
    "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
    "log_messages": [], # Stores recent log messages for the API
    "run_id": None,
}
_agent_state_lock = threading.Lock() # To ensure thread-safe updates to _current_agent_state

# --- Initialize Clients ---
qdrant_client = get_qdrant_client()
# Initialize both LLM models
try:
    main_llm = get_main_model()  # More powerful model
    assistant_llm = get_assistant_model()  # Faster/cheaper model
except Exception as e:
    logger.exception("Fatal Error: Could not initialize LLM models. Exiting.")
    # Update state before exiting if possible
    with _agent_state_lock:
        _current_agent_state["status"] = "error"
        _current_agent_state["error_message"] = "LLM initialization failed."
        _current_agent_state["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
    exit()  # Exit if models can't be loaded

# Ensure Qdrant collections exist
initialize_qdrant_collections(qdrant_client)

# --- Get Available Tools ---
# Fetch the list of tools once at the start
available_tools_desc = get_available_tools_list()
logger.info(f"Available tools: {list(available_tools_desc.keys())}")

# --- State Definition ---
# Based on the components and loop described in Goal.md


class AgentState(TypedDict):
    """
    Represents the state of the agent graph.
    """

    initial_goal: Optional[str]  # The initial request or trigger
    active_goal: Optional[str]  # The currently focused goal
    goal_history: List[str]  # Track achieved/failed goals if needed
    environment_observations: List[str]  # What the agent perceives
    retrieved_memories: List[Dict[str, Any]]  # Data pulled from Qdrant
    current_plan: List[str]  # Sequence of actions to achieve the goal
    executed_actions: List[
        Dict[str, Any]
    ]  # History of actions taken {action: ..., result: ...}
    reflection_insights: List[str]  # Learnings from reflection
    last_reflection_decision: Optional[
        str
    ]  # Decision from reflection (continue, replan, achieved, failed)
    error_message: Optional[str]  # To capture failures
    # Add a unique run ID for memory association
    run_id: str


# --- Node Placeholder Functions ---
# These will be replaced with actual logic later


def start_run(state: AgentState) -> AgentState:
    """Initializes the run_id."""
    logger.info("--- Starting Run ---")
    run_id = str(uuid.uuid4())
    logger.info(f"Run ID: {run_id}")
    with _agent_state_lock:
        _current_agent_state["run_id"] = run_id
        _current_agent_state["status"] = "starting_run"
        _current_agent_state["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return {"run_id": run_id}


def goal_evaluation(state: AgentState) -> AgentState:
    logger.info("--- Evaluating Goal ---")
    with _agent_state_lock:
        _current_agent_state["current_node"] = "goal_evaluation"
        _current_agent_state["status"] = "evaluating_goal"
        _current_agent_state["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")

    initial_goal = state.get("initial_goal", "No initial goal provided.")
    run_id = state["run_id"]

    # Use Assistant LLM for potentially simpler goal refinement
    prompt = f"""Given the initial user request: '{initial_goal}', analyze it and refine it into a clear, actionable primary goal for an autonomous AI agent.
Focus on the core intent. If the request is simple and clear, the refined goal might be the same.
Example:
Request: 'Tell me about the weather in London.'
Refined Goal: 'Get the current weather forecast for London, UK.'
Request: 'Summarize the main points of the latest paper on LLM scaling laws.'
Refined Goal: 'Find and summarize the main points of the most recent research paper discussing scaling laws for Large Language Models.'

Refined Goal:"""
    logger.info("Attempting to refine goal using Assistant LLM...")
    refined_goal = generate_text(prompt, assistant_llm)  # Use assistant_llm

    if refined_goal:
        active_goal = refined_goal.strip()
        logger.info(f"Refined Goal: {active_goal}")
    else:
        logger.warning("Goal refinement failed or returned empty, using initial goal.")
        active_goal = initial_goal  # Fallback to initial

    # Store the goal in goal_memory
    goal_doc = {
        "id": f"goal_{run_id}",
        "goal": active_goal,
        "status": "active",
        "run_id": run_id,
    }
    add_memory(qdrant_client, "goal_memory", [goal_doc], [active_goal])

    with _agent_state_lock:
        _current_agent_state["active_goal"] = active_goal
    return {"active_goal": active_goal, "error_message": None}


def sense_environment(state: AgentState) -> AgentState:
    logger.info("--- Sensing Environment ---")
    with _agent_state_lock:
        _current_agent_state["current_node"] = "sense_environment"
        _current_agent_state["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
    # Placeholder: In a real scenario, this could involve API calls, sensor readings etc.
    # For now, let's just add a timestamp as an observation.
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    observations = [f"Current time is {now}."]
    # Potentially add results from previous actions if relevant?
    # executed_actions = state.get("executed_actions", [])
    # if executed_actions:
    #     last_result = executed_actions[-1].get("result", "No result")
    #     observations.append(f"Result of last action: {str(last_result)[:100]}...") # Truncated

    logger.info(f"Observations: {observations}")
    return {"environment_observations": observations}


def memory_retrieval(state: AgentState) -> AgentState:
    logger.info("--- Retrieving Memory ---")
    with _agent_state_lock:
        _current_agent_state["current_node"] = "memory_retrieval"
        _current_agent_state["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
    active_goal = state["active_goal"]
    observations = state.get("environment_observations", [])
    # executed_actions = state.get("executed_actions", []) # Consider adding context from recent actions if needed
    run_id = state["run_id"]

    if not active_goal:
        logger.error("Active goal is not set for memory retrieval.")
        with _agent_state_lock:
            _current_agent_state["error_message"] = "Active goal not set for memory retrieval."
        return {"error_message": "Active goal not set for memory retrieval."}

    # --- Generate a focused query using Assistant LLM ---
    query_generation_prompt = f"""
    Given the current goal and recent observations, generate a concise search query to retrieve relevant information from memory (past experiences, learned facts) that would help in planning the next steps.

    Current Goal: {active_goal}
    Recent Observations: {' '.join(observations) if observations else 'None'}

    Focus the query on the core task or information needed to progress towards the goal based on the current situation.
    Example: If the goal is 'Research LLM scaling laws' and observation is 'Found paper X', the query might be 'Key findings of LLM scaling law paper X' or 'Alternative approaches to LLM scaling'.

    Search Query:"""

    logger.info("Generating memory search query using Assistant LLM...")
    generated_query = generate_text(
        query_generation_prompt, assistant_llm
    )  # Use assistant_llm

    if not generated_query or generated_query.strip().lower() == "none":
        logger.warning(
            "LLM failed to generate a specific search query. Using goal as fallback."
        )
        query_text = active_goal  # Fallback to using the goal directly
    else:
        query_text = generated_query.strip()
        logger.info(f"Generated Query: {query_text}")
    # --- End Query Generation ---

    # Search relevant memory types using the generated query
    episodic_mem = search_memory(qdrant_client, "episodic_memory", query_text, limit=3)
    semantic_mem = search_memory(qdrant_client, "semantic_memory", query_text, limit=2)

    # Retrieve tool descriptions (keep using the live list for now)
    tool_mem_desc = [
        {"source": "tool_description", "payload": {"name": name, "description": desc}}
        for name, desc in available_tools_desc.items()
    ]

    retrieved_memories = (
        [{"source": "episodic", **mem} for mem in episodic_mem]
        + [{"source": "semantic", **mem} for mem in semantic_mem]
        + tool_mem_desc
    )

    logger.info(
        f"Retrieved {len(retrieved_memories)} memories/tool descriptions using query: '{query_text}'"
    )
    with _agent_state_lock:
        _current_agent_state["retrieved_memories_count"] = len(retrieved_memories)
    # Clear any potential error from previous steps if memory retrieval itself succeeded
    return {"retrieved_memories": retrieved_memories, "error_message": None}


def planning(state: AgentState) -> AgentState:
    logger.info("--- Planning ---")
    with _agent_state_lock:
        _current_agent_state["current_node"] = "planning"
        _current_agent_state["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
    active_goal = state["active_goal"]
    memories = state["retrieved_memories"]
    observations = state["environment_observations"]
    run_id = state["run_id"]

    if not active_goal:
        logger.error("Active goal is not set for planning.")
        with _agent_state_lock:
            _current_agent_state["error_message"] = "Active goal not set for planning."
        return {"error_message": "Active goal not set for planning."}

    # Extract tool descriptions from retrieved memories/list
    tool_info = "\n".join(
        [
            f"- {mem['payload']['name']}: {mem['payload']['description']}"
            for mem in memories
            if mem.get("source") == "tool_description"
        ]
    )
    if not tool_info:
        tool_info = "No tools available."

    # Filter out tool descriptions from general memories passed to LLM
    general_memories = [
        mem for mem in memories if mem.get("source") != "tool_description"
    ]

    # Construct prompt for the LLM, explicitly mentioning tools
    prompt = f"""
    Objective: Create a step-by-step plan for an AI agent to achieve the following goal.
    Goal: {active_goal}

    Current Situation:
    {observations if observations else "No specific observations."}

    Available Tools:
    {tool_info}

    Relevant Information from Memory (use cautiously):
    {general_memories if general_memories else "No relevant memories retrieved."}

    Instructions:
    - Generate a concise, numbered list of actions for the agent.
    - If using a tool, format the step as: TOOL_NAME[argument]
      (e.g., web_search[latest AI research], placeholder_tool[data to process])
    - If not using a tool, describe the internal action (e.g., "Analyze search results", "Summarize findings").
    - Focus on the immediate next steps. The plan might be revised later.
    - If the goal seems impossible or requires clarification, state that clearly as the plan.

    Plan:
    1. [First action step]
    2. [Second action step]
    ...
    """
    logger.info("Generating plan using Main LLM...")
    plan_str = generate_text(prompt, main_llm)  # Use main_llm for planning

    if not plan_str:
        logger.error("LLM failed to generate a plan.")
        return {"error_message": "LLM failed to generate a plan."}

    # Basic parsing of the numbered list plan
    plan = [
        line.strip().split(". ", 1)[-1]
        for line in plan_str.split("\n")
        if line.strip() and line.strip()[0].isdigit()
    ]
    if not plan:
        logger.warning(f"Could not parse numbered plan, using raw output: {plan_str}")
        plan = [plan_str.strip()]  # Fallback

    logger.info(f"Generated Plan: {plan}")
    with _agent_state_lock:
        _current_agent_state["current_plan"] = plan
    return {"current_plan": plan}


def action_execution(state: AgentState) -> AgentState:
    logger.info("--- Executing Action ---")
    with _agent_state_lock:
        _current_agent_state["current_node"] = "action_execution"
        _current_agent_state["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
    plan = state["current_plan"]
    run_id = state["run_id"]
    executed_actions_history = state.get("executed_actions", [])

    if not plan:
        logger.info("Notice: No plan to execute.")
        # No error here, could be end of plan. Let should_continue decide.
        return {"error_message": None}  # Clear any previous transient error

    action_step = plan[0]
    remaining_plan = plan[1:]
    logger.info(f"Attempting action: {action_step}")
    with _agent_state_lock:
        _current_agent_state["current_action_step"] = action_step


    # Check if it looks like a tool call
    if "[" in action_step and action_step.endswith("]"):
        # Execute using the tool executor
        execution_result = execute_tool(action_step)
        action_outcome = execution_result["result"]
        action_error = execution_result["error"]
        tool_name = execution_result["tool_name"]
        argument = execution_result["argument"]

        if action_error:
            logger.error(f"Error executing tool '{tool_name}': {action_error}")
            # Decide how to handle tool errors - stop run? try to replan?
            # For now, record the error and let reflection/loop decide.
            action_outcome = f"Error: {action_error}"  # Store error as outcome
            # Optionally set a graph-level error to force end/replan
            # return {"error_message": f"Tool execution failed: {action_error}"}
    else:
        # Assume it's an internal step (e.g., analysis, summarization)
        # For now, we just acknowledge it. Later, this could involve LLM calls.
        logger.info(f"Executing internal step (placeholder): {action_step}")
        tool_name = "internal"
        argument = None
        action_outcome = f"Internal step '{action_step}' acknowledged."
        action_error = None

    logger.info(f"Action Outcome: {str(action_outcome)[:200]}...")  # Log truncated outcome

    # Record the execution attempt
    executed_actions_history.append(
        {
            "action": action_step,
            "tool_name": tool_name,
            "argument": argument,
            "result": action_outcome,  # Store outcome (could be result or error message)
            "error": action_error,  # Store specific error if one occurred
            "run_id": run_id,
        }
    )
    with _agent_state_lock:
        _current_agent_state["executed_actions"] = executed_actions_history
        _current_agent_state["last_action_result"] = action_outcome
        _current_agent_state["last_action_error"] = action_error

    # Update state: remove executed step, add record, clear transient error
    return {
        "current_plan": remaining_plan,
        "executed_actions": executed_actions_history,
        "error_message": None,  # Clear graph error if action attempted
    }


def reflection(state: AgentState) -> AgentState:
    logger.info("--- Reflecting ---")
    with _agent_state_lock:
        _current_agent_state["current_node"] = "reflection"
        _current_agent_state["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
    executed_actions = state.get("executed_actions", [])
    active_goal = state["active_goal"]
    current_plan = state.get("current_plan", [])
    run_id = state["run_id"]

    if not executed_actions:
        logger.info("No actions executed yet, skipping reflection.")
        # If no actions, assume we continue with the plan if one exists
        decision = (
            "continue" if current_plan else "achieved"
        )  # Or 'failed' if no plan and no actions?
        with _agent_state_lock:
            _current_agent_state["last_reflection_decision"] = decision
        return {"last_reflection_decision": decision}

    last_action_record = executed_actions[-1]

    # Construct prompt for reflection, asking for a specific decision keyword
    prompt = f"""
    Goal: {active_goal}
    Current Plan Remaining: {current_plan if current_plan else 'None'}
    Last Action Taken: {last_action_record['action']}
    Tool Used: {last_action_record['tool_name']}
    Argument: {last_action_record['argument']}
    Result/Outcome: {last_action_record['result']}
    Error During Execution: {last_action_record['error'] if last_action_record['error'] else 'None'}

    Reflect on the progress towards the goal based on the last action's outcome.
    - Was the action successful (or did it encounter an error)?
    - Did the result (or error) provide useful information or move closer to the goal?
    - Are there any key learnings or insights from this step?
    - Based *only* on the outcome of the *last action* and the remaining plan, what should be the next step? Choose *one* keyword from the following options:
        - ACHIEVED: If the last action's result clearly indicates the overall goal is complete.
        - FAILED: If the last action encountered an unrecoverable error or the result indicates the goal is impossible.
        - REPLAN: If the last action's result suggests the current plan is no longer optimal or needs adjustment based on new information.
        - CONTINUE: If the last action was successful (or failed benignly) and the plan should proceed to the next step.

    Reflection:
    [Your detailed reflection here]

    Decision: [Keyword: ACHIEVED|FAILED|REPLAN|CONTINUE]
    """
    logger.info("Generating reflection using Main LLM...")
    reflection_text = generate_text(prompt, main_llm)  # Use main_llm for reflection

    if not reflection_text:
        reflection_text = "Reflection failed."
        decision = "failed"  # If reflection fails, assume failure
        logger.error("LLM failed to generate reflection.")
    else:
        logger.info(f"Reflection Output (raw): {reflection_text}")
        # Attempt to parse the decision keyword
        match = re.search(
            r"Decision:\s*\[?Keyword:\s*(ACHIEVED|FAILED|REPLAN|CONTINUE)\]?",
            reflection_text,
            re.IGNORECASE | re.MULTILINE,
        )
        if match:
            decision = match.group(1).lower()
            logger.info(f"Parsed Decision: {decision}")
        else:
            logger.warning(
                "Could not parse decision keyword from reflection. Defaulting to 'continue'."
            )
            decision = "continue"  # Default if parsing fails

    insights = state.get("reflection_insights", [])
    insights.append(reflection_text)  # Store the full reflection text

    with _agent_state_lock:
        _current_agent_state["reflection_insights"] = insights
        _current_agent_state["last_reflection_decision"] = decision
    # Store the parsed decision in the state
    return {"reflection_insights": insights, "last_reflection_decision": decision}


def memory_update(state: AgentState) -> AgentState:
    logger.info("--- Updating Memory ---")
    with _agent_state_lock:
        _current_agent_state["current_node"] = "memory_update"
        _current_agent_state["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
    insights = state.get("reflection_insights", [])
    executed_actions = state.get("executed_actions", [])
    run_id = state["run_id"]

    if not executed_actions:
        logger.info("No actions executed, nothing to store in memory.")
        return {}

    last_action_record = executed_actions[-1]
    last_insight = insights[-1] if insights else "No specific insight generated."
    now_iso = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()  # Get current UTC timestamp

    # Create documents to store
    # Episodic memory: What happened? Include tool info and error status
    episodic_doc = {
        "id": f"ep_{run_id}_{len(executed_actions)}",
        "run_id": run_id,
        "action": last_action_record["action"],
        "tool_name": last_action_record["tool_name"],
        "argument": last_action_record["argument"],
        "result": last_action_record[
            "result"
        ],  # Store the outcome (result or error message)
        "error": last_action_record["error"],  # Explicitly store error
        "timestamp": now_iso,  # Use the actual timestamp
    }
    # Text for embedding should represent the key event
    episodic_text = f"Action taken: {last_action_record['action']}. Tool: {last_action_record['tool_name']}. Outcome: {last_action_record['result'][:100]}..."  # Truncate long results
    if last_action_record["error"]:
        episodic_text += f" Error: {last_action_record['error']}"
    episodic_text += f" Timestamp: {now_iso}"  # Optionally include timestamp in text

    # Semantic memory: What was learned?
    semantic_doc = {
        "id": f"sem_{run_id}_{len(insights)}",
        "run_id": run_id,
        "insight": last_insight,
        "related_action": last_action_record["action"],
        "goal": state.get("active_goal", "Unknown"),  # Associate insight with goal
        "timestamp": now_iso,  # Also timestamp the insight
    }
    semantic_text = f"Insight regarding goal '{state.get('active_goal', 'Unknown')}' at {now_iso}: {last_insight}"

    # Add to Qdrant
    add_memory(qdrant_client, "episodic_memory", [episodic_doc], [episodic_text])
    add_memory(qdrant_client, "semantic_memory", [semantic_doc], [semantic_text])

    logger.info(
        f"Stored episodic and semantic memory for action: {last_action_record['action']}"
    )

    # Optional: Update tool memory if reflection provides insights about tool usage/failure
    # if "tool usage insight" in last_insight:
    #    tool_insight_doc = {...}
    #    add_memory(qdrant_client, "tool_memory", [tool_insight_doc], [insight_text])

    return {}


# --- Graph Definition ---

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("start_run", start_run)  # New entry node
workflow.add_node("goal_evaluation", goal_evaluation)
workflow.add_node("sense_environment", sense_environment)
workflow.add_node("memory_retrieval", memory_retrieval)
workflow.add_node("planning", planning)
workflow.add_node("action_execution", action_execution)
workflow.add_node("reflection", reflection)
workflow.add_node("memory_update", memory_update)

# Define edges
workflow.set_entry_point("start_run")
workflow.add_edge("start_run", "goal_evaluation")  # Start with goal eval after setup
workflow.add_edge("goal_evaluation", "sense_environment")
workflow.add_edge("sense_environment", "memory_retrieval")
workflow.add_edge("memory_retrieval", "planning")
workflow.add_edge("planning", "action_execution")
workflow.add_edge("action_execution", "reflection")
workflow.add_edge("reflection", "memory_update")

# --- Conditional Edges / Loop Logic ---


def should_continue(state: AgentState) -> str:
    logger.info("--- Checking Loop Condition ---")
    with _agent_state_lock:
        _current_agent_state["current_node"] = "should_continue"
        _current_agent_state["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
    error = state.get("error_message")
    decision = state.get("last_reflection_decision")
    plan = state.get("current_plan", [])

    # Prioritize hard errors from the graph execution itself
    if error:
        logger.error(f"Graph error detected: {error}. Ending run.")
        with _agent_state_lock:
            _current_agent_state["status"] = "error"
            _current_agent_state["error_message"] = error
        return "end"  # Route to END state

    logger.info(f"Reflection Decision: {decision}")
    logger.info(f"Remaining Plan Steps: {len(plan)}")

    # Use the decision from the reflection node
    final_decision = "end" # Default to end
    if decision == "achieved":
        logger.info("Reflection indicates goal achieved. Ending run.")
        final_decision = "end"
        with _agent_state_lock: _current_agent_state["status"] = "achieved"
    elif decision == "failed":
        logger.info("Reflection indicates goal failed or unrecoverable error. Ending run.")
        final_decision = "end"
        with _agent_state_lock: _current_agent_state["status"] = "failed"
    elif decision == "replan":
        logger.info("Reflection suggests replanning is needed.")
        final_decision = "replan"  # Route back to planning
        with _agent_state_lock: _current_agent_state["status"] = "replanning"
    elif decision == "continue":
        # If reflection says continue, check if there's actually a plan left
        if not plan:
            logger.info(
                "Plan complete (reflection said continue, but no steps left). Ending run."
            )
            final_decision = "end"
            with _agent_state_lock: _current_agent_state["status"] = "plan_complete"
        else:
            logger.info(
                "Plan has remaining steps and reflection indicates continue. Continuing execution."
            )
            final_decision = "continue_execution"  # Route to next action
            with _agent_state_lock: _current_agent_state["status"] = "continuing"
    else:
        # Fallback if decision is missing or invalid (shouldn't happen ideally)
        logger.warning("Invalid or missing reflection decision. Checking plan status.")
        if not plan:
            logger.info("Plan complete (fallback). Ending run.")
            final_decision = "end"
            with _agent_state_lock: _current_agent_state["status"] = "plan_complete_fallback"
        else:
            logger.info("Plan has remaining steps (fallback). Continuing execution.")
            final_decision = "continue_execution"
            with _agent_state_lock: _current_agent_state["status"] = "continuing_fallback"
    return final_decision


# Add conditional edge from memory_update
workflow.add_conditional_edges(
    "memory_update",
    should_continue,
    {
        "continue_execution": "action_execution",  # Loop back to execute next step
        "replan": "planning",  # Route to planning if reflection suggests it
        "end": END,  # End the graph execution
    },
)

# Compile the graph
app = workflow.compile()


# --- API Server Integration Functions ---
def get_current_agent_status() -> Dict[str, Any]:
    """Returns a copy of the current agent state for the API."""
    with _agent_state_lock:
        # Add recent log messages to the status
        # This simple approach keeps the last N messages. A more robust solution
        # might use a proper logging handler that updates this list.
        # For now, assuming other parts of the code might append to _current_agent_state['log_messages']
        # If not, this list will remain empty unless explicitly populated.
        # A better way: use a custom logging handler.
        # For now, let's add a placeholder if no logs are captured.
        if not _current_agent_state.get("log_messages"):
             _current_agent_state["log_messages"] = ["No log messages captured in agent state yet."]
        return _current_agent_state.copy()

def start_agent_loop_in_thread_from_server(initial_goal: str, new_goal_event: Optional[threading.Event] = None):
    """
    Runs the agent's main processing loop.
    This function is intended to be called by the API server, potentially in a thread.
    """
    logger.info(f"AGENT_GRAPH: Thread started for goal: {initial_goal}")
    current_goal = initial_goal
    max_iterations = 50 # Safety break for the loop
    iteration = 0

    while iteration < max_iterations :
        iteration += 1
        logger.info(f"AGENT_GRAPH: Starting/Continuing agent stream with goal: {current_goal} (Iteration {iteration})")
        inputs = {"initial_goal": current_goal}
        with _agent_state_lock:
            _current_agent_state["status"] = "running"
            _current_agent_state["active_goal"] = current_goal
            _current_agent_state["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _current_agent_state["run_id"] = None # Reset run_id for new goal processing cycle
            _current_agent_state["log_messages"] = [] # Clear logs for new run


        # Stream agent execution
        for event in app.stream(inputs, {"recursion_limit": 25}): # Increased recursion limit
            with _agent_state_lock:
                _current_agent_state["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
                for key, value in event.items(): # key is the node name
                    _current_agent_state["current_node"] = key
                    if isinstance(value, dict):
                        # Update specific fields based on the event from the node
                        if 'active_goal' in value: _current_agent_state["active_goal"] = value['active_goal']
                        if 'current_plan' in value: _current_agent_state["current_plan"] = value['current_plan']
                        if 'executed_actions' in value: _current_agent_state["executed_actions"] = value['executed_actions']
                        if 'reflection_insights' in value: _current_agent_state["reflection_insights"] = value['reflection_insights']
                        if 'error_message' in value and value['error_message']:
                             _current_agent_state["error_message"] = value['error_message']
                             _current_agent_state["status"] = "error"
                        if 'run_id' in value: _current_agent_state["run_id"] = value['run_id']

                    # Simplified log capture: append string representation of the event
                    # This is very basic; a custom logging handler would be better.
                    log_entry = f"Event: {key} - Value: {str(value)[:200]}"
                    _current_agent_state["log_messages"].append(log_entry)
                    if len(_current_agent_state["log_messages"]) > 50: # Keep last 50 messages
                        _current_agent_state["log_messages"].pop(0)

            logger.debug(f"AGENT_GRAPH_STREAM: Event: {event}") # Log the event itself

            # Check for new goal signal from API
            if new_goal_event and new_goal_event.is_set():
                logger.info("AGENT_GRAPH: New goal event detected by agent. Restarting loop with new goal.")
                # The API server is expected to update `current_goal` or pass it
                # For this simple model, we assume the `initial_goal` arg of this function
                # is now the *new* goal if the event is set.
                # The API server needs to re-call this function or use a shared variable for the new goal.
                # Let's assume the API server will pass the new goal via the initial_goal parameter
                # if it calls this function again or if there's a state update mechanism.
                # For now, the loop will break, and the API server might restart it.
                # A more robust approach would be a queue for goals.
                new_goal_event.clear()
                # To get the *actual* new goal, the API server would need to update a shared variable
                # or `initial_goal` would need to be the new goal for the *next* call.
                # This is a simplification: we'll just log and the outer loop (if any) handles it.
                # The current structure implies this function is the main loop for ONE goal,
                # and the API server starts a new thread for each new goal.
                # If using a single persistent thread, goal management needs a queue.
                # For now, let's assume the API server will set the new goal if it re-invokes this or a similar function.
                # We'll just break this stream and the outer `while` loop can potentially pick up a new goal
                # if `current_goal` is updated by the API server through some other means.
                # This part needs careful design for continuous operation.
                # For this iteration, let's assume the API server starts a new thread/call for a new goal.
                # So, if event is set, this run for the *old* goal should probably terminate.
                logger.info("AGENT_GRAPH: New goal event processed. Current agent run for old goal will now end.")
                with _agent_state_lock:
                    _current_agent_state["status"] = "awaiting_new_goal_after_interrupt"
                return # Exit this function, letting the thread end. API server can start a new one.

        with _agent_state_lock:
            if _current_agent_state["status"] not in ["error", "achieved", "failed"]:
                 _current_agent_state["status"] = "completed_goal_run" # Or based on final node

        logger.info(f"AGENT_GRAPH: Finished agent stream for goal: {current_goal} (Iteration {iteration})")

        if new_goal_event:
            logger.info("AGENT_GRAPH: Waiting for new goal event or timeout...")
            # Wait for a new goal event or a short timeout to allow the loop to naturally exit if no new goal
            event_is_set = new_goal_event.wait(timeout=5) # Wait for 5 seconds
            if event_is_set:
                logger.info("AGENT_GRAPH: New goal event received. Loop will restart with new goal.")
                # The API server should have updated the goal. We need a way to get that new goal.
                # This is tricky with the current structure. For now, let's assume the API server
                # would call this function again in a new thread, or update a shared variable.
                # If current_goal is a shared mutable object or updated via another mechanism:
                # current_goal = get_latest_goal_from_api_shared_state()
                new_goal_event.clear()
                # The 'initial_goal' for the *next* iteration of this loop needs to be the *new* goal.
                # This part is still a bit conceptual with the current direct arg passing.
                # A better model: this function is called once per goal.
                # So, if a new goal comes, this instance finishes, and API server calls again.
                # Let's stick to: if event is set, this run ends.
                return
            else:
                logger.info("AGENT_GRAPH: No new goal event after timeout. Thread will exit.")
                with _agent_state_lock:
                    _current_agent_state["status"] = "idle_timeout"
                return # Exit loop and thread if no new goal after timeout
        else:
            # If no new_goal_event object, this loop was likely for a single run.
            logger.info("AGENT_GRAPH: No new_goal_event provided. Loop will complete after one run.")
            return # Exit loop


# --- Main Execution Block (for standalone testing) ---
if __name__ == "__main__":
    # --- Root Logger Configuration ---
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    file_handler = logging.FileHandler("agent.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(file_handler)
    # --- End Root Logger Configuration ---

    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY not found.")
    elif not os.getenv("LANGCHAIN_API_KEY"):
        logger.warning("LANGCHAIN_API_KEY not found. LangSmith tracing will be disabled.")
    else:
        logger.info("API keys loaded. LangSmith tracing configured.")
    logger.info(f"Using Main Model: {MAIN_MODEL_NAME}")
    logger.info(f"Using Assistant Model: {ASSISTANT_MODEL_NAME}")

    # Example of direct invocation for testing
    logger.info("Starting agent graph for a test goal (standalone execution)...")
    test_goal = "What is the latest news about the Perseverance rover on Mars?"

    # This will run the agent for one goal and then exit, updating the global state.
    # In a server context, start_agent_loop_in_thread_from_server would be used.
    start_agent_loop_in_thread_from_server(test_goal, None)

    logger.info("--- Standalone Agent Run Complete ---")
    logger.info("Final Agent State (from global):")
    logger.info(json.dumps(get_current_agent_status(), indent=2))
