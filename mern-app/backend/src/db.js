// db.js - opens and closes the connection to MongoDB through Mongoose.

// Mongoose is an ODM (Object Document Mapper): it gives MongoDB documents a schema,
// validation, and convenient methods like Task.find().
import mongoose from "mongoose";

// Connect once at startup. Mongoose then reuses this connection for every query.
export async function connectDB(uri) {
  // Fail after 10 s instead of hanging if MongoDB is not running.
  await mongoose.connect(uri, { serverSelectionTimeoutMS: 10_000 });
  // Log which database we are using (never log the full URI: it may contain a password).
  console.log(`Connected to MongoDB database "${mongoose.connection.name}"`);
}

// Close the connection cleanly (used on shutdown and in tests).
export async function disconnectDB() {
  // Waits for in-flight operations, then closes all sockets.
  await mongoose.disconnect();
}

// True when the connection is open. readyState 1 means "connected".
export function isDBConnected() {
  // The other states are 0 disconnected, 2 connecting, 3 disconnecting.
  return mongoose.connection.readyState === 1;
}
