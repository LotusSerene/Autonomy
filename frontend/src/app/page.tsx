export default function HomePage() {
  return (
    <main className="flex min-h-screen flex-col items-center p-4 md:p-8 lg:p-12 bg-slate-900 text-white">
      <div className="w-full max-w-7xl mb-8">
        <h1 className="text-4xl md:text-5xl font-bold text-center text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-600 py-2">
          Agent Brain Visualization
        </h1>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full max-w-7xl">
        {/* Visualization Section (takes 2/3 width on medium screens and up) */}
        <section className="md:col-span-2 p-6 bg-slate-800 shadow-xl rounded-lg border border-slate-700">
          <h2 className="text-2xl font-semibold mb-4 text-purple-300">Cognitive Flow</h2>
          <div id="visualization-placeholder" className="w-full h-[60vh] min-h-[400px] bg-slate-700/50 rounded-md flex items-center justify-center border border-slate-600">
            <p className="text-slate-400 text-lg">Agent graph will be rendered here</p>
          </div>
        </section>

        {/* Status & Logs Section (takes 1/3 width on medium screens and up) */}
        <section className="md:col-span-1 p-6 bg-slate-800 shadow-xl rounded-lg border border-slate-700">
          <div className="mb-6">
            <h2 className="text-2xl font-semibold mb-4 text-teal-300">Current State</h2>
            <div id="status-placeholder" className="space-y-2 text-slate-300">
              <p><strong>Status:</strong> <span id="agent-status" className="text-yellow-400">Initializing...</span></p>
              <p><strong>Active Node:</strong> <span id="agent-node" className="text-sky-400">-</span></p>
              <p><strong>Goal:</strong> <span id="agent-goal" className="text-gray-400 text-sm">-</span></p>
            </div>
          </div>
          <div>
            <h2 className="text-2xl font-semibold mb-4 text-orange-300">Activity Log</h2>
            <div id="logs-placeholder" className="w-full h-96 bg-slate-900/70 rounded-md p-3 overflow-y-auto border border-slate-700 text-sm">
              <p className="text-slate-500">Log entries will appear here...</p>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
