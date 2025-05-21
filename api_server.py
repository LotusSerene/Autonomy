import threading
import time # For placeholder data updates
from flask import Flask, jsonify, request
import os # For reading log file
import logging

# Configure basic logging for the API server itself
api_logger = logging.getLogger(__name__ + '_api') # Unique name for API logger
# Ensure the root logger is configured if not already by agent_graph or other modules
# This basicConfig will only have an effect if no handlers are already configured on the root logger.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# Attempt to import agent graph components
try:
    # Assuming agent_graph.py will provide these
    from agent_graph import get_current_agent_status, start_agent_loop_in_thread_from_server
    AGENT_GRAPH_AVAILABLE = True
    api_logger.info("Successfully imported functions from agent_graph.")
except ImportError as e:
    api_logger.warning(f"Could not import from agent_graph: {e}. API will use placeholder functions.")
    AGENT_GRAPH_AVAILABLE = False
    # Define placeholder functions if agent_graph.py is not ready
    _placeholder_status_data = {
        "status": "simulated_initializing",
        "current_node": "N/A",
        "active_goal": "N/A",
        "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
        "agent_graph_module_loaded": AGENT_GRAPH_AVAILABLE,
        "log_messages": ["Placeholder logs: Agent graph module not loaded."]
    }
    def get_current_agent_status():
        _placeholder_status_data["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _placeholder_status_data["current_node"] = "simulated_node_" + str(int(time.time()) % 5) # Simulate change
        return _placeholder_status_data

    def start_agent_loop_in_thread_from_server(initial_goal, new_goal_event=None):
        api_logger.info(f"AGENT_GRAPH PLACEHOLDER: Called to start/handle goal: {initial_goal}")
        _placeholder_status_data["active_goal"] = initial_goal
        _placeholder_status_data["status"] = "simulated_running"
        if new_goal_event:
            new_goal_event.set() # Signal that the goal has been "processed"
        pass

app = Flask(__name__)

# --- Agent State (global, to be updated by agent_graph.py or placeholders) ---
# This is accessed by get_current_agent_status

def update_placeholder_status_periodically():
    ''' Periodically updates status if agent_graph is not available by calling its placeholder '''
    if not AGENT_GRAPH_AVAILABLE:
        while True:
            time.sleep(10) # Update less frequently
            get_current_agent_status() # This updates the shared _placeholder_status_data
            api_logger.info("API_SERVER (Placeholder): Updated placeholder status.")


@app.route('/api/agent/status', methods=['GET'])
def get_status():
    status = get_current_agent_status()
    return jsonify(status)

@app.route('/api/agent/logs', methods=['GET'])
def get_logs():
    log_file_path = 'agent.log' # Assuming agent_graph.py logs here
    max_lines = request.args.get('lines', 50, type=int)
    try:
        if AGENT_GRAPH_AVAILABLE: # Try to read from actual agent state if possible
            status = get_current_agent_status()
            log_entries = status.get("log_messages", [])[-max_lines:]
            if not log_entries and not os.path.exists(log_file_path) : # Fallback if no logs in state and no file
                 return jsonify({"logs": ["No log messages available in agent state and 'agent.log' not found."], "message": "Log file 'agent.log' not found and no logs in state."}), 404
        elif os.path.exists(log_file_path): # Fallback for placeholder or if state doesn't have logs
            with open(log_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            log_entries = [line.strip() for line in lines[-max_lines:]]
        else: # Placeholder mode and no log file
            return jsonify({"logs": _placeholder_status_data.get("log_messages",[]), "message": "Agent graph not loaded, showing placeholder logs. 'agent.log' not found."}), 200


        return jsonify({"logs": log_entries})
    except Exception as e:
        api_logger.error(f"Error reading log file/state: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Event to signal that a new goal has been submitted via API
# This is a more robust way to handle goal submission to a running agent thread
new_goal_event_for_agent = None
if AGENT_GRAPH_AVAILABLE:
    new_goal_event_for_agent = threading.Event()


@app.route('/api/agent/goal', methods=['POST'])
def submit_goal():
    data = request.get_json()
    if not data or 'goal' not in data:
        return jsonify({"error": "Invalid request. 'goal' field is required."}), 400

    goal = data['goal']
    api_logger.info(f"API_SERVER: Received new goal '{goal}'.")

    try:
        # This function should now handle the logic of passing the goal to the running agent thread
        # or starting the agent if it's not running.
        # The `start_agent_loop_in_thread_from_server` in agent_graph.py will need to be designed for this.
        start_agent_loop_in_thread_from_server(goal, new_goal_event_for_agent)
        return jsonify({"message": "Goal submitted to agent.", "goal": goal})
    except Exception as e:
        api_logger.error(f"Error submitting goal to agent: {e}", exc_info=True)
        return jsonify({"error": f"Failed to submit goal: {str(e)}"}), 500

if __name__ == '__main__':
    api_logger.info("API_SERVER: Initializing Flask API server...")

    if not AGENT_GRAPH_AVAILABLE:
        api_logger.info("API_SERVER: Running with placeholder agent status due to import issues.")
        threading.Thread(target=update_placeholder_status_periodically, daemon=True).start()
    else:
        api_logger.info("API_SERVER: Attempting to start agent graph's main processing loop in a background thread.")
        initial_agent_goal = "Initialize and await new goals via API."
        # The agent_graph.py's start_agent_loop_in_thread_from_server should be designed to run continuously
        # and potentially accept new goals through a shared mechanism (e.g., a queue or event).
        thread = threading.Thread(target=start_agent_loop_in_thread_from_server, args=(initial_agent_goal, new_goal_event_for_agent), daemon=True)
        thread.start()
        api_logger.info(f"API_SERVER: Agent graph thread started: {thread.name}")


    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
