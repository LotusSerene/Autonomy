import os
import uuid
import json  # Import json for potential result parsing
import re  # Import re for parsing reflection
import datetime  # Import datetime for timestamps
from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

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

# --- Initialize Clients ---
qdrant_client = get_qdrant_client()
# Initialize both LLM models
try:
    main_llm = get_main_model()  # More powerful model
    assistant_llm = get_assistant_model()  # Faster/cheaper model
except Exception as e:
    print(f"Fatal Error: Could not initialize LLM models. Exiting. Error: {e}")
    exit()  # Exit if models can't be loaded

# Ensure Qdrant collections exist
initialize_qdrant_collections(qdrant_client)

# --- Get Available Tools ---
# Fetch the list of tools once at the start
available_tools_desc = get_available_tools_list()
print(f"Available tools: {list(available_tools_desc.keys())}")

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
    print("--- Starting Run ---")
    run_id = str(uuid.uuid4())
    print(f"Run ID: {run_id}")
    return {"run_id": run_id}


def goal_evaluation(state: AgentState) -> AgentState:
    print("--- Evaluating Goal ---")
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
    print("Attempting to refine goal using Assistant LLM...")
    refined_goal = generate_text(prompt, assistant_llm)  # Use assistant_llm

    if refined_goal:
        active_goal = refined_goal.strip()
        print(f"Refined Goal: {active_goal}")
    else:
        print("Goal refinement failed or returned empty, using initial goal.")
        active_goal = initial_goal  # Fallback to initial

    # Store the goal in goal_memory
    goal_doc = {
        "id": f"goal_{run_id}",
        "goal": active_goal,
        "status": "active",
        "run_id": run_id,
    }
    add_memory(qdrant_client, "goal_memory", [goal_doc], [active_goal])

    return {"active_goal": active_goal, "error_message": None}


def sense_environment(state: AgentState) -> AgentState:
    print("--- Sensing Environment ---")
    # Placeholder: In a real scenario, this could involve API calls, sensor readings etc.
    # For now, let's just add a timestamp as an observation.
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    observations = [f"Current time is {now}."]
    # Potentially add results from previous actions if relevant?
    # executed_actions = state.get("executed_actions", [])
    # if executed_actions:
    #     last_result = executed_actions[-1].get("result", "No result")
    #     observations.append(f"Result of last action: {str(last_result)[:100]}...") # Truncated

    print(f"Observations: {observations}")
    return {"environment_observations": observations}


def memory_retrieval(state: AgentState) -> AgentState:
    print("--- Retrieving Memory ---")
    active_goal = state["active_goal"]
    observations = state.get("environment_observations", [])
    # executed_actions = state.get("executed_actions", []) # Consider adding context from recent actions if needed
    run_id = state["run_id"]

    if not active_goal:
        print("Error: Active goal is not set.")
        return {"error_message": "Active goal not set for memory retrieval."}

    # --- Generate a focused query using Assistant LLM ---
    query_generation_prompt = f"""
    Given the current goal and recent observations, generate a concise search query to retrieve relevant information from memory (past experiences, learned facts) that would help in planning the next steps.

    Current Goal: {active_goal}
    Recent Observations: {' '.join(observations) if observations else 'None'}

    Focus the query on the core task or information needed to progress towards the goal based on the current situation.
    Example: If the goal is 'Research LLM scaling laws' and observation is 'Found paper X', the query might be 'Key findings of LLM scaling law paper X' or 'Alternative approaches to LLM scaling'.

    Search Query:"""

    print("Generating memory search query using Assistant LLM...")
    generated_query = generate_text(
        query_generation_prompt, assistant_llm
    )  # Use assistant_llm

    if not generated_query or generated_query.strip().lower() == "none":
        print(
            "Warning: LLM failed to generate a specific search query. Using goal as fallback."
        )
        query_text = active_goal  # Fallback to using the goal directly
    else:
        query_text = generated_query.strip()
        print(f"Generated Query: {query_text}")
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

    print(
        f"Retrieved {len(retrieved_memories)} memories/tool descriptions using query: '{query_text}'"
    )
    # Clear any potential error from previous steps if memory retrieval itself succeeded
    return {"retrieved_memories": retrieved_memories, "error_message": None}


def planning(state: AgentState) -> AgentState:
    print("--- Planning ---")
    active_goal = state["active_goal"]
    memories = state["retrieved_memories"]
    observations = state["environment_observations"]
    run_id = state["run_id"]

    if not active_goal:
        print("Error: Active goal is not set.")
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
    print("Generating plan using Main LLM...")
    plan_str = generate_text(prompt, main_llm)  # Use main_llm for planning

    if not plan_str:
        print("Error: LLM failed to generate a plan.")
        return {"error_message": "LLM failed to generate a plan."}

    # Basic parsing of the numbered list plan
    plan = [
        line.strip().split(". ", 1)[-1]
        for line in plan_str.split("\n")
        if line.strip() and line.strip()[0].isdigit()
    ]
    if not plan:
        plan = [plan_str.strip()]  # Fallback

    print(f"Generated Plan: {plan}")
    return {"current_plan": plan}


def action_execution(state: AgentState) -> AgentState:
    print("--- Executing Action ---")
    plan = state["current_plan"]
    run_id = state["run_id"]
    executed_actions_history = state.get("executed_actions", [])

    if not plan:
        print("Notice: No plan to execute.")
        # No error here, could be end of plan. Let should_continue decide.
        return {"error_message": None}  # Clear any previous transient error

    action_step = plan[0]
    remaining_plan = plan[1:]
    print(f"Attempting action: {action_step}")

    # Check if it looks like a tool call
    if "[" in action_step and action_step.endswith("]"):
        # Execute using the tool executor
        execution_result = execute_tool(action_step)
        action_outcome = execution_result["result"]
        action_error = execution_result["error"]
        tool_name = execution_result["tool_name"]
        argument = execution_result["argument"]

        if action_error:
            print(f"Error executing tool: {action_error}")
            # Decide how to handle tool errors - stop run? try to replan?
            # For now, record the error and let reflection/loop decide.
            action_outcome = f"Error: {action_error}"  # Store error as outcome
            # Optionally set a graph-level error to force end/replan
            # return {"error_message": f"Tool execution failed: {action_error}"}
    else:
        # Assume it's an internal step (e.g., analysis, summarization)
        # For now, we just acknowledge it. Later, this could involve LLM calls.
        print(f"Executing internal step (placeholder): {action_step}")
        tool_name = "internal"
        argument = None
        action_outcome = f"Internal step '{action_step}' acknowledged."
        action_error = None

    print(f"Action Outcome: {action_outcome[:200]}...")  # Log truncated outcome

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

    # Update state: remove executed step, add record, clear transient error
    return {
        "current_plan": remaining_plan,
        "executed_actions": executed_actions_history,
        "error_message": None,  # Clear graph error if action attempted
    }


def reflection(state: AgentState) -> AgentState:
    print("--- Reflecting ---")
    executed_actions = state.get("executed_actions", [])
    active_goal = state["active_goal"]
    current_plan = state.get("current_plan", [])
    run_id = state["run_id"]

    if not executed_actions:
        print("No actions executed yet, skipping reflection.")
        # If no actions, assume we continue with the plan if one exists
        decision = (
            "continue" if current_plan else "achieved"
        )  # Or 'failed' if no plan and no actions?
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
    print("Generating reflection using Main LLM...")
    reflection_text = generate_text(prompt, main_llm)  # Use main_llm for reflection

    if not reflection_text:
        reflection_text = "Reflection failed."
        decision = "failed"  # If reflection fails, assume failure
        print("Error: LLM failed to generate reflection.")
    else:
        print(f"Reflection Output (raw): {reflection_text}")
        # Attempt to parse the decision keyword
        match = re.search(
            r"Decision:\s*\[?Keyword:\s*(ACHIEVED|FAILED|REPLAN|CONTINUE)\]?",
            reflection_text,
            re.IGNORECASE | re.MULTILINE,
        )
        if match:
            decision = match.group(1).lower()
            print(f"Parsed Decision: {decision}")
        else:
            print(
                "Warning: Could not parse decision keyword from reflection. Defaulting to 'continue'."
            )
            decision = "continue"  # Default if parsing fails

    insights = state.get("reflection_insights", [])
    insights.append(reflection_text)  # Store the full reflection text

    # Store the parsed decision in the state
    return {"reflection_insights": insights, "last_reflection_decision": decision}


def memory_update(state: AgentState) -> AgentState:
    print("--- Updating Memory ---")
    insights = state.get("reflection_insights", [])
    executed_actions = state.get("executed_actions", [])
    run_id = state["run_id"]

    if not executed_actions:
        print("No actions executed, nothing to store in memory.")
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

    print(
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
    print("--- Checking Loop Condition ---")
    error = state.get("error_message")
    decision = state.get("last_reflection_decision")
    plan = state.get("current_plan", [])

    # Prioritize hard errors from the graph execution itself
    if error:
        print(f"Graph error detected: {error}. Ending run.")
        return "end"  # Route to END state

    print(f"Reflection Decision: {decision}")
    print(f"Remaining Plan Steps: {len(plan)}")

    # Use the decision from the reflection node
    if decision == "achieved":
        print("Reflection indicates goal achieved. Ending run.")
        return "end"
    elif decision == "failed":
        print("Reflection indicates goal failed or unrecoverable error. Ending run.")
        return "end"
    elif decision == "replan":
        print("Reflection suggests replanning is needed.")
        return "replan"  # Route back to planning
    elif decision == "continue":
        # If reflection says continue, check if there's actually a plan left
        if not plan:
            print(
                "Plan complete (reflection said continue, but no steps left). Ending run."
            )
            return "end"
        else:
            print(
                "Plan has remaining steps and reflection indicates continue. Continuing execution."
            )
            return "continue_execution"  # Route to next action
    else:
        # Fallback if decision is missing or invalid (shouldn't happen ideally)
        print("Warning: Invalid or missing reflection decision. Checking plan status.")
        if not plan:
            print("Plan complete (fallback). Ending run.")
            return "end"
        else:
            print("Plan has remaining steps (fallback). Continuing execution.")
            return "continue_execution"


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

# --- Running the Agent (Example) ---
if __name__ == "__main__":
    # Ensure API keys are loaded (redundant if load_dotenv() called globally, but safe)
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found. Please set it in your .env file.")
    elif not os.getenv("LANGCHAIN_API_KEY"):
        print(
            "Warning: LANGCHAIN_API_KEY not found. LangSmith tracing will be disabled."
        )
        # You might want to disable tracing explicitly if the key is missing
        # os.environ["LANGCHAIN_TRACING_V2"] = "false"
    else:
        print(
            "API keys loaded. LangSmith tracing is configured (if LANGCHAIN_TRACING_V2 is 'true')."
        )
        print(f"Using Main Model: {MAIN_MODEL_NAME}")  # Log model names - NOW DEFINED
        print(
            f"Using Assistant Model: {ASSISTANT_MODEL_NAME}"
        )  # Log model names - NOW DEFINED

    inputs = {
        "initial_goal": "What is the latest news about the Perseverance rover on Mars?"
    }
    # Use stream to see events step-by-step
    for event in app.stream(
        inputs, {"recursion_limit": 15}
    ):  # Increase recursion limit slightly
        for key, value in event.items():
            print(f"\n--- Node: {key} ---")
            # Print specific state keys for clarity
            print(f" Active Goal: {value.get('active_goal')}")
            print(f" Plan: {value.get('current_plan')}")
            print(f" Executed Actions: {len(value.get('executed_actions', []))}")
            if value.get("executed_actions"):
                print(f" Last Action: {value['executed_actions'][-1]['action']}")
                print(
                    f" Last Result: {str(value['executed_actions'][-1]['result'])[:200]}..."
                )  # Truncate long results
                print(f" Last Error: {value['executed_actions'][-1]['error']}")
            if value.get("reflection_insights"):
                print(f" Last Insight: {value['reflection_insights'][-1][:200]}...")
            print(f" Graph Error: {value.get('error_message')}")
            print("-" * 20)

    # Or use invoke for final state
    # final_state = app.invoke(inputs, {"recursion_limit": 10})
    # print("\n--- Agent Run Complete ---")
    # print("Final State:")
    # print(final_state)
