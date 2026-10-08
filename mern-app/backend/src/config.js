// config.js - every setting the server needs, read from environment variables.

// Load variables from a .env file (if one exists) into process.env.
// Built into Node.js, so no "dotenv" package is needed. Variables that are
// already set in the real environment win over the file.
try {
  // Path is relative to where you run `npm start` (the backend folder).
  process.loadEnvFile(".env");
} catch (error) {
  // No .env file is fine: the defaults below are used instead.
  // Any other problem (e.g. a file we can't read) should stop the server.
  if (error.code !== "ENOENT") throw error;
}

// One plain object holding all settings; other files import it.
export const config = {
  // Port the HTTP server listens on.
  port: Number(process.env.PORT) || 4000,
  // MongoDB connection string. The database name ("mern_tasks") is the last part.
  mongoUri: process.env.MONGODB_URI || "mongodb://127.0.0.1:27017/mern_tasks",
  // Browser origins allowed to call the API directly (comma-separated list).
  clientOrigins: (process.env.CLIENT_ORIGIN || "http://localhost:5173")
    // Split "a,b" into ["a", "b"]...
    .split(",")
    // ...remove stray spaces...
    .map((origin) => origin.trim())
    // ...and drop empty entries.
    .filter(Boolean),
};
