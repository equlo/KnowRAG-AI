// index.js - the entry point: connect to MongoDB, then start the HTTP server.
// Run with `npm start` (or `npm run dev` to restart automatically on file changes).

// Builds the Express app.
import { createApp } from "./app.js";
// Settings from environment variables / .env.
import { config } from "./config.js";
// Database helpers.
import { connectDB, disconnectDB } from "./db.js";

// Connect to the database first: the API is useless without it.
try {
  // Top-level await works because package.json sets "type": "module".
  await connectDB(config.mongoUri);
} catch (error) {
  // Explain the most common cause, then stop with a failure exit code.
  console.error(`Could not connect to MongoDB at startup. Is it running?\n${error.message}`);
  process.exit(1);
}

// Create the app with the allowed browser origins.
const app = createApp({ clientOrigins: config.clientOrigins });

// Start listening for HTTP requests.
const server = app.listen(config.port, (error) => {
  // In Express 5 the callback receives an error if the port could not be opened.
  if (error) {
    // Usually "EADDRINUSE": another program already uses this port.
    console.error(`Could not start the server: ${error.message}`);
    process.exit(1);
  }
  // Tell the developer where to find the API.
  console.log(`API listening on http://localhost:${config.port}`);
});

// Shut down cleanly when the process is asked to stop (Ctrl+C or `docker stop`).
function shutdown(signal) {
  // Log why we are stopping.
  console.log(`${signal} received, shutting down...`);
  // Stop accepting new connections and wait for open requests to finish...
  server.close(async () => {
    // ...then close the database connection...
    await disconnectDB();
    // ...and exit successfully.
    process.exit(0);
  });
}

// Ctrl+C in the terminal sends SIGINT.
process.on("SIGINT", shutdown);
// Docker and most process managers send SIGTERM.
process.on("SIGTERM", shutdown);
