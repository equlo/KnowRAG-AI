// errors.js - turns every failure into a JSON response of the form { error: "..." }.

// Needed to recognise Mongoose's error classes.
import mongoose from "mongoose";

// Express calls this when no route matched the request.
export function notFound(req, res) {
  // Reply with 404 and say which route was missing.
  res.status(404).json({ error: `Route not found: ${req.method} ${req.originalUrl}` });
}

// Express recognises an error handler by its FOUR parameters (err, req, res, next).
// In Express 5, errors thrown inside async route handlers arrive here automatically.
export function errorHandler(err, req, res, next) {
  // A document failed schema validation (e.g. a missing title).
  if (err instanceof mongoose.Error.ValidationError) {
    // Collect one readable message per invalid field.
    const messages = Object.values(err.errors).map((fieldError) =>
      // Type mismatches (e.g. an object where a string belongs) get a friendlier message.
      fieldError instanceof mongoose.Error.CastError ? `${fieldError.path} has an invalid value` : fieldError.message,
    );
    // 400 Bad Request: the client sent bad data.
    return res.status(400).json({ error: messages.join("; ") });
  }

  // A value could not be converted to the schema type (e.g. completed: "maybe" in an update).
  if (err instanceof mongoose.Error.CastError) {
    // Also the client's fault.
    return res.status(400).json({ error: `${err.path} has an invalid value` });
  }

  // Errors from Express itself (e.g. malformed JSON) carry an HTTP status.
  const status = err.status || err.statusCode || 500;

  // 5xx errors are bugs or outages: log the details for the developer...
  if (status >= 500) console.error(err);

  // ...but never send internal details (stack traces, queries) to the browser.
  res.status(status).json({ error: status >= 500 ? "Something went wrong on the server" : err.message });
}
