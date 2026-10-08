// app.js - builds the Express application (middleware + routes) WITHOUT starting it.
// Keeping this separate from index.js lets the tests use the app without opening a port.

// CORS middleware: lets a browser page on another origin call this API.
import cors from "cors";
// Express is the web framework.
import express from "express";
// Reports whether MongoDB is reachable, for the health check.
import { isDBConnected } from "./db.js";
// Our 404 and error handlers.
import { errorHandler, notFound } from "./middleware/errors.js";
// The /api/tasks routes.
import tasksRouter from "./routes/tasks.js";

// Create and configure a new app. `clientOrigins` comes from config.js.
export function createApp({ clientOrigins = [] } = {}) {
  // A fresh Express application.
  const app = express();

  // Don't advertise "X-Powered-By: Express" in every response.
  app.disable("x-powered-by");

  // Allow only the listed origins (e.g. the Vite dev server) to call the API from a browser.
  app.use(cors({ origin: clientOrigins }));

  // Parse JSON request bodies into req.body. The size limit blocks huge payloads.
  app.use(express.json({ limit: "10kb" }));

  // GET /api/health - quick check that the server and database are up.
  app.get("/api/health", (req, res) => {
    // Is the MongoDB connection open?
    const connected = isDBConnected();
    // 200 when healthy, 503 Service Unavailable when the database is down.
    res.status(connected ? 200 : 503).json({
      // Overall status.
      status: connected ? "ok" : "error",
      // Database status.
      database: connected ? "connected" : "disconnected",
    });
  });

  // Every /api/tasks/... request goes to the tasks router.
  app.use("/api/tasks", tasksRouter);

  // Nothing matched: reply 404.
  app.use(notFound);

  // Must be registered LAST so it catches errors from everything above.
  app.use(errorHandler);

  // Give the configured app back to the caller.
  return app;
}
