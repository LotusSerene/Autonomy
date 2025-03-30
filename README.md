# Autonomous Intent Framework (Concept Project)

## Introduction

This project explores a conceptual framework for building AI agents with a degree of "autonomous intent." The core idea is to create systems capable of proactively pursuing high-level goals, adapting their plans based on memory, environmental context, and self-reflection. This framework integrates concepts similar to Chain-of-Thought (CoT) reasoning through its planning and reflection cycles.

**Important Clarification:** This framework aims to _mimic_ aspects of intent and proactive behavior within the current limitations of AI. It does **not** aim to create consciousness, sentience, or true self-awareness, which are far beyond the scope of this project and current AI capabilities. The term "intent" is used conceptually to describe the agent's ability to formulate and pursue goals based on its internal state and external triggers.

## Core Concept: Autonomous Intent + CoT

The framework combines several key ideas:

1.  **Goal-Driven Operation:** The agent starts with an initial goal (or generates one proactively) and works iteratively to achieve it.
2.  **Memory Integration:** Utilizes multiple memory types (episodic, semantic, goal) stored in a vector database (Qdrant) to provide context, recall past experiences, and store learned insights.
3.  **Environmental Interaction:** Leverages a suite of tools (`tools.py`) to interact with the external environment (web search, file system, APIs, Git).
4.  **Planning & Execution Cycle:** Dynamically creates plans based on the current goal, observations, and retrieved memories. Executes plan steps using available tools or internal processing.
5.  **Reflection & Adaptation:** After actions, the agent reflects on the outcome, updates its memory with new insights (semantic memory), and adapts its plan if necessary (replan, mark goal as achieved/failed). This cycle mirrors aspects of CoT reasoning by evaluating progress and adjusting strategy.
6.  **Proactive Hypothesis Generation:** Includes tools (`generate_hypothesis`) allowing the agent to potentially propose new goals or tasks based on its context and capabilities when idle.

## Project Structure & Components

The framework is built upon several key Python modules:

- **`agent_graph.py`:** The heart of the agent. It uses LangGraph to define and orchestrate the main operational loop:
  - Goal Evaluation -> Sense Environment -> Memory Retrieval -> Planning -> Action Execution -> Reflection -> Memory Update -> Loop/End.
- **`llm_utils.py`:** Handles all interactions with the underlying Large Language Models (specifically Google Generative AI models like Gemini). Provides functions for text generation and creating vector embeddings for memory.
- **`qdrant_utils.py`:** Manages the agent's memory. It interfaces with the Qdrant vector database to store and retrieve different types of memories (episodic, semantic, goal, potentially tool-related) based on vector similarity.
- **`tools.py`:** Defines the agent's capabilities. It includes a registry of functions the agent can call to perform actions like searching the web (DuckDuckGo, ArXiv, Wikipedia), reading/writing/editing files within a secure workspace, making API calls (to allowed domains), interacting with Git repositories, managing its own goals, requesting human input, and performing self-reflection or generating new hypotheses using the LLM.

## How it Works (Conceptual Flow)

The agent operates in a cycle orchestrated by `agent_graph.py`:

1.  **Start:** Receives an initial goal or trigger.
2.  **Goal Evaluation:** Refines the initial request into a clear, actionable goal.
3.  **Sense:** Gathers information about the current state (e.g., time, results of the last action).
4.  **Memory Retrieval:** Queries the Qdrant memory stores (episodic, semantic) based on the current goal and observations to find relevant past experiences or learned knowledge. Retrieves available tool descriptions.
5.  **Planning:** Uses an LLM (main model) to generate a step-by-step plan to achieve the active goal, considering the retrieved memories, observations, and available tools.
6.  **Action Execution:** Executes the next step in the plan. This might involve calling a specific tool (like `web_search` or `edit_file`) or performing an internal analysis step.
7.  **Reflection:** Uses an LLM (main model) to evaluate the outcome of the executed action against the goal. It determines whether to continue the plan, replan, or mark the goal as achieved or failed. Generates insights.
8.  **Memory Update:** Stores the executed action (episodic memory) and any generated insights (semantic memory) back into Qdrant.
9.  **Loop/End:** Based on the reflection decision, either loops back to execute the next action, replans, or terminates the run.

## Key Features

- **Modular Design:** Components for LLM interaction, memory, tools, and orchestration are separated.
- **Vector Memory:** Leverages Qdrant for efficient similarity-based retrieval of relevant context.
- **Extensible Toolset:** Designed to easily incorporate new tools and capabilities.
- **Planning & Reflection:** Implements a core loop for adaptive behavior and learning.
- **Security Considerations:** Basic measures like workspace restrictions and allowed API domains are included in `tools.py`.
- **Potential for Proactivity:** Includes stubs and tools for hypothesis generation and environment monitoring (though monitoring requires further implementation).

## Setup (Conceptual)

As this is a conceptual project, setup involves:

1.  **Python Environment:** Python 3.x.
2.  **Dependencies:** Install required libraries (e.g., `langgraph`, `qdrant-client`, `google-generativeai`, `python-dotenv`, `duckduckgo-search`, `arxiv`, `wikipedia`, `requests`). A `requirements.txt` file would typically be included.
3.  **Environment Variables:** Create a `.env` file and populate it with necessary API keys and configurations:
    - `GOOGLE_API_KEY`: For Google Generative AI access.
    - `QDRANT_URL`: URL for your Qdrant instance (e.g., `http://localhost:6333`).
    - `QDRANT_API_KEY`: (Optional) API key if your Qdrant instance requires authentication.
    - `LANGCHAIN_API_KEY`: (Optional) For LangSmith tracing.
    - `LANGCHAIN_TRACING_V2="true"`: (Optional) To enable LangSmith tracing.
    - `AGENT_WORKSPACE`: (Optional) Path to the directory where the agent can perform file operations (defaults to `./agent_workspace`).
4.  **Qdrant Instance:** Ensure a Qdrant vector database instance is running and accessible at the specified `QDRANT_URL`.

## Running the Example

The `agent_graph.py` script includes a `if __name__ == "__main__":` block that demonstrates how to run the agent graph with an example initial goal. You can execute it directly:

```bash
python agent_graph.py
```

This will trigger the agent loop and print output from each node as it progresses.

## Why Gemini?

Because it's API is free to use, while I can use other models like Claude-3.7 or GPT-o3-mini/gpt4o it's not that important to be because Gemini Models work well enough, I'll think about implementing more models as I move forward, for now this is just a conceptual project that I will work on when I have free time.

## Future Work & Concept Status

This project serves as a foundation and exploration of autonomous agent concepts. Potential future directions include:

- **Sophisticated Environment Monitoring:** Implementing robust background processes for the `monitor_environment` tool.
- **Enhanced Error Handling:** More granular error detection and recovery strategies within the graph.
- **Deeper CoT Integration:** Explicitly prompting for and utilizing detailed reasoning steps within the planning and reflection nodes.
- **Improved Tool Argument Handling:** More robust parsing and validation for complex tool arguments.
- **User Interface:** Building a UI for interaction instead of relying solely on console input/output.
- **Advanced Memory Management:** Implementing memory summarization, relevance filtering, and decay mechanisms.
- **Multi-Agent Collaboration:** Exploring interactions between multiple instances of this framework.
