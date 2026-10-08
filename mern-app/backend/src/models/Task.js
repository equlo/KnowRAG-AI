// Task.js - the Mongoose model for one to-do item: the "M" (MongoDB) side of MERN.

// Mongoose turns this schema into a model class we can query.
import mongoose from "mongoose";

// A schema describes the shape of each document and the rules it must follow.
const taskSchema = new mongoose.Schema(
  {
    // The text of the task.
    title: {
      // Stored as a string.
      type: String,
      // Must be present; the message is what the API returns when it is missing.
      required: [true, "Title is required"],
      // Remove leading/trailing spaces, so "   " becomes "" and fails `required`.
      trim: true,
      // Keep titles short enough to display nicely.
      maxlength: [200, "Title must be 200 characters or fewer"],
    },
    // Whether the task is done.
    completed: {
      // Stored as true/false.
      type: Boolean,
      // New tasks start as not done.
      default: false,
    },
  },
  {
    // Add `createdAt` and `updatedAt` fields that Mongoose maintains automatically.
    timestamps: true,
    // Don't add the internal "__v" version field to documents.
    versionKey: false,
  },
);

// Compile the schema into a model. Documents go into the "tasks" collection
// (Mongoose lower-cases and pluralises the model name).
export default mongoose.model("Task", taskSchema);
