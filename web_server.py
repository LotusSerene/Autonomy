import os
import threading  # Import threading
import time  # Import time for sleep
from flask import Flask, render_template, send_from_directory, request, jsonify
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv

# Import the compiled agent graph
from agent_graph import app as agent_app  # Rename to avoid conflict

# Load environment variables (optional, but good practice)
load_dotenv()

flask_app = Flask(__name__)  # Rename Flask app instance
# Secret key is needed for Flask session management, used by SocketIO
flask_app.config["SECRET_KEY"] = os.getenv(
    "FLASK_SECRET_KEY", "your_fallback_secret_key_here!"
)

# Initialize SocketIO
# async_mode='threading' is often simpler for background tasks without complex async frameworks
# Adjust cors_allowed_origins if your frontend is served from a different domain/port
socketio = SocketIO(flask_app, async_mode="threading", cors_allowed_origins="*")

# Global flag/variable to track agent run status
agent_running = False
agent_thread = None


@flask_app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html")


# Removed the /static route as Flask handles it automatically
# when using url_for in templates and the static folder exists.

# --- Agent Execution Function ---


def run_agent_graph(initial_goal: str):
    """Runs the agent graph stream and emits updates via SocketIO."""
    global agent_running
    print("****** Entered run_agent_graph function ******")
    agent_running = True
    print(f"--- Starting agent graph with goal: {initial_goal} ---")
    socketio.emit(
        "agent_update", {"status": f'Agent starting with goal: "{initial_goal}"...'}
    )
    try:
        inputs = {"initial_goal": initial_goal}
        print(f"****** Calling agent_app.stream with inputs: {inputs} ******")
        # Use stream to see events step-by-step
        for event in agent_app.stream(inputs, {"recursion_limit": 15}):
            print(f"****** Processing stream event: {list(event.keys())} ******")
            if not agent_running:  # Check flag to allow stopping
                print("Agent run stopped externally.")
                break
            # Process event and emit updates
            for node_name, state_update in event.items():
                # General status update for any node entry/update
                status_message = f"Processing node: {node_name}"
                socketio.emit(
                    "agent_update", {"status": status_message, "node": node_name}
                )

                # Detailed brain state update
                # We can customize the data sent based on the node
                brain_data = {
                    "node": node_name,
                    "status": "Processing",  # Default status
                    "data": {},  # Placeholder for specific data
                }
                if isinstance(state_update, dict):
                    # Send selective state info, avoid sending large objects like full memory
                    brain_data["status"] = state_update.get(
                        "last_reflection_decision", "Processing"
                    )  # Use reflection decision if available
                    brain_data["data"] = {
                        "active_goal": state_update.get("active_goal"),
                        "plan": state_update.get("current_plan"),
                        "last_action": (
                            state_update.get("executed_actions", [{}])[-1].get("action")
                            if state_update.get("executed_actions")
                            else None
                        ),
                        "error": state_update.get("error_message"),
                    }
                    # Clean None values for cleaner UI display
                    brain_data["data"] = {
                        k: v for k, v in brain_data["data"].items() if v is not None
                    }

                # Emit the brain state update
                socketio.emit("brain_state", brain_data)
                print(f"Brain state update: {brain_data}")  # Log brain state updates

                # Add a small delay to make updates visible in UI
                socketio.sleep(0.1)

        socketio.emit("agent_update", {"status": "Agent run finished."})
        socketio.emit("brain_state", {"node": "END", "status": "Finished"})
        print("--- Agent graph run finished ---")

    except Exception as e:
        error_msg = f"Error during agent execution: {e}"
        print(error_msg)
        socketio.emit("agent_update", {"status": error_msg, "error": True})
        socketio.emit(
            "brain_state",
            {"node": "ERROR", "status": "Error", "data": {"error": error_msg}},
        )
    finally:
        agent_running = False


# --- SocketIO Event Handlers ---


@socketio.on("connect")
def handle_connect():
    """Handle new client connections."""
    print("Client connected")
    emit(
        "agent_update",
        {
            "status": "Server connected. Ready for agent start.",
            "agent_running": agent_running,
        },
    )


@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnections."""
    print("Client disconnected")


@socketio.on("start_agent")
def handle_start_agent(data):
    """Handle request from client to start the agent."""
    global agent_running, agent_thread
    print("****** Received 'start_agent' event from client ******")
    if not agent_running:
        initial_goal = data.get("goal", "Default goal: Summarize recent AI news.")
        print(f"Received start request with goal: {initial_goal}")
        print(
            f"****** Creating and starting agent thread for goal: {initial_goal} ******"
        )
        agent_thread = threading.Thread(target=run_agent_graph, args=(initial_goal,))
        agent_thread.daemon = True  # Allow server to exit even if thread is running
        agent_thread.start()
        emit(
            "agent_update",
            {"status": "Agent start requested...", "agent_running": True},
        )
    else:
        print("Agent is already running.")
        emit(
            "agent_update",
            {"status": "Agent is already running.", "agent_running": True},
        )


@socketio.on("stop_agent")
def handle_stop_agent():
    """Handle request from client to stop the agent."""
    global agent_running
    if agent_running:
        print("Received stop request.")
        agent_running = False  # Signal the thread to stop
        # Note: The thread will stop gracefully after its current stream event
        emit(
            "agent_update",
            {"status": "Agent stop requested...", "agent_running": False},
        )
    else:
        print("Agent is not running.")
        emit(
            "agent_update",
            {"status": "Agent is not currently running.", "agent_running": False},
        )


# --- Removed old helper functions send_agent_update, send_brain_state ---
# They are replaced by direct socketio.emit calls in run_agent_graph

# --- Main Execution ---

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    print(f"Starting Flask-SocketIO server on http://localhost:{port}")

    # Example: Start agent automatically on server start (Optional)
    # Comment out if you want to start via UI button
    # print("Starting agent automatically...")
    # initial_goal_auto = "What is the latest news about the Perseverance rover on Mars?"
    # agent_thread_auto = threading.Thread(target=run_agent_graph, args=(initial_goal_auto,))
    # agent_thread_auto.daemon = True
    # agent_thread_auto.start()

    # Setting debug=False to prevent potential issues with the reloader and background threads
    # You might need to manually restart the server when making code changes.
    # Set use_reloader=False explicitly as well for clarity.
    print("Running with debug=False and use_reloader=False for stability.")
    socketio.run(
        flask_app,
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )
