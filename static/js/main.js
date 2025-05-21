document.addEventListener("DOMContentLoaded", (event) => {
  // Connect to the Socket.IO server.
  // The connection URL might need adjustment based on your Flask setup (e.g., different host/port)
  // Default connection is to the same host and port the page is served from.
  const socket = io();

  // --- DOM Elements ---
  const statusContentElement = document.getElementById("agent-status-content");
  const brainContainer = document.getElementById("brain-container");
  const goalInputElement = document.getElementById("goal-input");
  const startButton = document.getElementById("start-agent-btn");
  const stopButton = document.getElementById("stop-agent-btn");

  let cy = null; // Cytoscape instance
  let agentRunning = false; // Track agent state on client

  // --- Cytoscape Initialization ---
  function initializeCytoscape() {
    if (!brainContainer) return;
    brainContainer.innerHTML = ""; // Clear placeholder
    cy = cytoscape({
      container: brainContainer,
      elements: [
        // Define nodes based on agent_graph.py nodes
        { data: { id: "start_run", label: "Start Run" } },
        { data: { id: "goal_evaluation", label: "Evaluate Goal" } },
        { data: { id: "sense_environment", label: "Sense Env" } },
        { data: { id: "memory_retrieval", label: "Retrieve Memory" } },
        { data: { id: "planning", label: "Plan" } },
        { data: { id: "action_execution", label: "Execute Action" } },
        { data: { id: "reflection", label: "Reflect" } },
        { data: { id: "memory_update", label: "Update Memory" } },
        { data: { id: "should_continue", label: "Check Loop" } },
        { data: { id: "END", label: "End" } }, // Implicit end node
        { data: { id: "ERROR", label: "Error" } }, // Error node

        // Define edges based on agent_graph.py
        {
          data: {
            id: "e_start_goal",
            source: "start_run",
            target: "goal_evaluation",
          },
        },
        {
          data: {
            id: "e_goal_sense",
            source: "goal_evaluation",
            target: "sense_environment",
          },
        },
        {
          data: {
            id: "e_sense_mem",
            source: "sense_environment",
            target: "memory_retrieval",
          },
        },
        {
          data: {
            id: "e_mem_plan",
            source: "memory_retrieval",
            target: "planning",
          },
        },
        {
          data: {
            id: "e_plan_exec",
            source: "planning",
            target: "action_execution",
          },
        },
        {
          data: {
            id: "e_exec_reflect",
            source: "action_execution",
            target: "reflection",
          },
        },
        {
          data: {
            id: "e_reflect_memupd",
            source: "reflection",
            target: "memory_update",
          },
        },
        {
          data: {
            id: "e_memupd_check",
            source: "memory_update",
            target: "should_continue",
          },
        },
        // Conditional edges represented by should_continue outcomes
        {
          data: {
            id: "e_check_continue",
            source: "should_continue",
            target: "action_execution",
            label: "Continue",
          },
        },
        {
          data: {
            id: "e_check_replan",
            source: "should_continue",
            target: "planning",
            label: "Replan",
          },
        },
        {
          data: {
            id: "e_check_end",
            source: "should_continue",
            target: "END",
            label: "End/Achieved/Failed",
          },
        },
        // Add edges to Error node (optional visualization)
        // { data: { id: 'e_any_error', source: '*', target: 'ERROR' } } // Can be complex
      ],
      style: [
        {
          selector: "node",
          style: {
            "background-color": "#666",
            label: "data(label)",
            width: "90px",
            height: "40px",
            padding: "10px",
            shape: "round-rectangle",
            "text-valign": "center",
            "text-halign": "center",
            "font-size": "10px",
            color: "white",
            "text-wrap": "wrap",
            "text-max-width": "80px",
          },
        },
        {
          selector: "edge",
          style: {
            width: 2,
            "line-color": "#ccc",
            "target-arrow-color": "#ccc",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            label: "data(label)", // Display edge labels
            "font-size": "8px",
            color: "#555",
            "text-rotation": "autorotate",
            "text-margin-y": -5,
          },
        },
        {
          selector: ".active",
          style: {
            "background-color": "#007bff", // Blue for active node
            "border-width": 2,
            "border-color": "#0056b3",
          },
        },
        {
          selector: ".success",
          style: {
            "background-color": "#28a745", // Green for success/end
            "border-width": 2,
            "border-color": "#1e7e34",
          },
        },
        {
          selector: ".error",
          style: {
            "background-color": "#dc3545", // Red for error
            "border-width": 2,
            "border-color": "#b02a37",
          },
        },
        {
          selector: "edge.active", // Style for active edges
          style: {
            "line-color": "#007bff",
            "target-arrow-color": "#007bff",
          },
        },
      ],
      layout: {
        name: "dagre", // Directed acyclic graph layout
        rankDir: "TB", // Top to bottom
        spacingFactor: 1.2,
      },
    });
    console.log("Cytoscape initialized.");

    // ---> ADDED: Check if cy object is valid and try a basic layout
    if (cy) {
      console.log("Cytoscape instance seems valid.");
      // Try applying layout again after a short delay, sometimes helps
      setTimeout(() => {
        cy.layout({ name: "dagre", rankDir: "TB", spacingFactor: 1.2 }).run();
        cy.fit(); // Fit the graph to the viewport
        console.log("Applied layout and fit.");
      }, 100);
    } else {
      console.error(
        "Cytoscape instance (cy) is null or undefined after initialization!"
      );
    }
  }

  // --- Cytoscape Update Function ---
  function updateBrainVisualization(data) {
    if (!cy) return;
    console.log("Updating brain visualization with:", data);

    // Stop any previous animations
    cy.elements().stop(true);
    // Remove previous classes from all elements
    cy.elements().removeClass("active error success incoming");

    const node_id = data.node; // The ID of the node from agent_graph
    if (node_id) {
      const nodeElement = cy.getElementById(node_id);
      if (nodeElement.length > 0) {
        let nodeClass = "active"; // Default class
        // Highlight the current node
        if (node_id === "ERROR" || data.status === "Error") {
          nodeClass = "error";
        } else if (
          node_id === "END" ||
          data.status === "Finished" ||
          data.status === "Goal Achieved"
        ) {
          nodeClass = "success";
        }
        nodeElement.addClass(nodeClass);

        // Highlight incoming edges to the active node
        const incomingEdge = nodeElement.incomers("edge");
        incomingEdge.addClass(nodeClass); // Use same class as node for edge color

        // --- Add Animation ---
        if (nodeClass === "active") {
          // Simple pulse effect for active node
          nodeElement.animate(
            {
              style: { "border-width": 4, "border-color": "#0056b3" },
            },
            {
              duration: 300,
              complete: function () {
                nodeElement.animate(
                  {
                    style: { "border-width": 2, "border-color": "#0056b3" },
                  },
                  { duration: 300 }
                );
              },
            }
          );
          // Animate edge width
          incomingEdge.animate(
            {
              style: { width: 4 },
            },
            {
              duration: 300,
              complete: function () {
                incomingEdge.animate(
                  { style: { width: 2 } },
                  { duration: 300 }
                );
              },
            }
          );
        }
        // --- End Animation ---

        // Optional: Zoom/pan to the active node
        // cy.animate({ center: { eles: nodeElement }, zoom: 1.5 }, { duration: 500 });
      } else {
        console.warn(`Node with ID '${node_id}' not found in Cytoscape graph.`);
      }
    } else if (data.status === "Finished" || data.status === "Error") {
      // Handle final states if node ID might be missing but status indicates end/error
      const finalNodeId = data.status === "Error" ? "ERROR" : "END";
      const finalNode = cy.getElementById(finalNodeId);
      if (finalNode.length > 0) {
        finalNode.addClass(data.status === "Error" ? "error" : "success");
      }
    }
    // Can extend this to show data.data details in a separate panel or node tooltips
  }

  // --- SocketIO Event Handlers ---
  socket.on("connect", () => {
    console.log("Connected to WebSocket server.");
    statusContentElement.textContent =
      "Connected to agent server. Ready to start.";
    initializeCytoscape(); // Initialize graph on connect
  });

  socket.on("disconnect", () => {
    console.log("Disconnected from WebSocket server.");
    statusContentElement.textContent = "Disconnected from agent server.";
    if (brainContainer)
      brainContainer.innerHTML = '<p style="color: red;">Connection Lost.</p>';
    startButton.disabled = false;
    stopButton.disabled = true;
    agentRunning = false;
  });

  socket.on("connect_error", (error) => {
    console.error("Connection Error:", error);
    statusContentElement.textContent = `Error connecting to agent server: ${error.message}`;
    if (brainContainer)
      brainContainer.innerHTML = '<p style="color: red;">Connection Error.</p>';
    startButton.disabled = false;
    stopButton.disabled = true;
    agentRunning = false;
  });

  socket.on("agent_update", (data) => {
    console.log("Agent Update Received:", data);
    // Update status log
    const timestamp = new Date().toLocaleTimeString();
    const newNode = document.createElement("div");
    newNode.textContent = `[${timestamp}] ${
      data.status || JSON.stringify(data)
    }`;
    if (data.error) newNode.style.color = "red";
    statusContentElement.appendChild(newNode);
    const maxLines = 50;
    while (statusContentElement.childNodes.length > maxLines) {
      statusContentElement.removeChild(statusContentElement.firstChild);
    }
    statusContentElement.scrollTop = statusContentElement.scrollHeight;

    // Update button states based on agent_running flag from server
    if (typeof data.agent_running === "boolean") {
      agentRunning = data.agent_running;
      startButton.disabled = agentRunning;
      stopButton.disabled = !agentRunning;
      goalInputElement.disabled = agentRunning;
    }
  });

  socket.on("brain_state", (data) => {
    // ---> ADDED: Log received data structure
    console.log("Received brain_state event:", JSON.stringify(data, null, 2));
    if (!data || typeof data.node !== "string") {
      console.error("Invalid brain_state data received:", data);
      return; // Stop processing if data is invalid
    }
    // Call the Cytoscape update function
    updateBrainVisualization(data);
  });

  // --- Control Button Listeners ---
  startButton.addEventListener("click", () => {
    const goal = goalInputElement.value.trim();
    if (!goal) {
      alert("Please enter an initial goal.");
      return;
    }
    if (!agentRunning) {
      console.log(`Requesting agent start with goal: ${goal}`);
      socket.emit("start_agent", { goal: goal });
      startButton.disabled = true; // Disable immediately, server will confirm state
      stopButton.disabled = false;
      goalInputElement.disabled = true;
      statusContentElement.textContent = `Requesting agent start with goal: "${goal}"...\n`; // Clear previous logs
    }
  });

  stopButton.addEventListener("click", () => {
    if (agentRunning) {
      console.log("Requesting agent stop...");
      socket.emit("stop_agent");
      stopButton.disabled = true; // Disable immediately, server will confirm state
    }
  });
});
