// tasks.js - the REST API for tasks: the "E" (Express) side of MERN.
// Mounted at /api/tasks in app.js, so "/" here means "/api/tasks".

// A Router is a mini-app that groups related routes.
import { Router } from "express";
// Used to check that an id looks like a MongoDB ObjectId.
import mongoose from "mongoose";
// The Mongoose model we read and write.
import Task from "../models/Task.js";

// Create the router.
const router = Router();

// Copy only the fields clients are allowed to set. Anything else in the body
// (e.g. "_id" or "createdAt") is ignored, so clients can't overwrite it.
function pickTaskFields(body = {}) {
  // Start with an empty object.
  const fields = {};
  // Include the title only if the client sent one.
  if (body.title !== undefined) fields.title = body.title;
  // Include the completed flag only if the client sent one.
  if (body.completed !== undefined) fields.completed = body.completed;
  // Return what we kept.
  return fields;
}

// Runs before every route below that has an ":id" parameter.
router.param("id", (req, res, next, id) => {
  // A valid id is a 24-character hex string such as "66f1c2a9e4b0a1b2c3d4e5f6".
  if (!mongoose.isObjectIdOrHexString(id)) {
    // Reject obviously bad ids before touching the database.
    return res.status(400).json({ error: "Invalid task id" });
  }
  // The id looks fine: continue to the route handler.
  next();
});

// GET /api/tasks - list every task, newest first.
router.get("/", async (req, res) => {
  // sort({ createdAt: -1 }) means descending creation time.
  const tasks = await Task.find().sort({ createdAt: -1 });
  // Express turns the array into JSON.
  res.json(tasks);
});

// POST /api/tasks - create a task from { title }.
router.post("/", async (req, res) => {
  // Mongoose validates the data; a ValidationError goes to the error handler (400).
  const task = await Task.create(pickTaskFields(req.body));
  // 201 Created, with the saved document (including its new _id).
  res.status(201).json(task);
});

// GET /api/tasks/:id - fetch one task.
router.get("/:id", async (req, res) => {
  // Look the task up by its _id.
  const task = await Task.findById(req.params.id);
  // findById returns null when nothing matches.
  if (!task) return res.status(404).json({ error: "Task not found" });
  // Send the task back.
  res.json(task);
});

// PATCH /api/tasks/:id - change the title and/or completed flag.
router.patch("/:id", async (req, res) => {
  // Apply the changes in one database round trip.
  const task = await Task.findByIdAndUpdate(req.params.id, pickTaskFields(req.body), {
    // Return the document AFTER the update (the default is the old version).
    returnDocument: "after",
    // Updates skip schema validation unless we ask for it.
    runValidators: true,
  });
  // null means no task has that id.
  if (!task) return res.status(404).json({ error: "Task not found" });
  // Send the updated task back.
  res.json(task);
});

// DELETE /api/tasks/:id - remove a task.
router.delete("/:id", async (req, res) => {
  // Delete it and get the removed document (or null).
  const task = await Task.findByIdAndDelete(req.params.id);
  // Nothing was deleted: the id is unknown.
  if (!task) return res.status(404).json({ error: "Task not found" });
  // 204 No Content: success with an empty body.
  res.status(204).end();
});

// Export the router so app.js can mount it.
export default router;
