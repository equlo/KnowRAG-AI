// tasks.test.js - tests for the whole HTTP API, run with `npm test`.
// Uses Node's built-in test runner, Supertest to send requests, and
// mongodb-memory-server to start a throwaway MongoDB, so no setup is needed.

// Assertions that compare values strictly (=== and deep equality).
import assert from "node:assert/strict";
// Test structure and lifecycle hooks from Node's built-in runner.
import { after, before, beforeEach, describe, test } from "node:test";
// Starts a real, temporary MongoDB server for the tests.
import { MongoMemoryServer } from "mongodb-memory-server";
// Sends HTTP requests to an Express app without opening a port.
import request from "supertest";
// The code under test.
import { createApp } from "../src/app.js";
import { connectDB, disconnectDB } from "../src/db.js";
import Task from "../src/models/Task.js";

// The temporary database server and the app, shared by all tests.
let mongo;
let app;

// Once, before any test: start MongoDB, connect, build the app.
before(async () => {
  // Downloads a MongoDB binary the first time (cached afterwards).
  mongo = await MongoMemoryServer.create();
  // Connect Mongoose to a test database on that server.
  await connectDB(mongo.getUri("mern_tasks_test"));
  // Build the Express app (no CORS origins needed for tests).
  app = createApp();
});

// Once, after all tests: disconnect and stop MongoDB.
after(async () => {
  await disconnectDB();
  await mongo.stop();
});

// Before every test: start from an empty collection so tests can't affect each other.
beforeEach(async () => {
  await Task.deleteMany({});
});

// Helper: create a task through the API and return the response body.
async function createTask(title) {
  const res = await request(app).post("/api/tasks").send({ title }).expect(201);
  return res.body;
}

describe("health check", () => {
  test("reports the database as connected", async () => {
    const res = await request(app).get("/api/health").expect(200);
    assert.deepEqual(res.body, { status: "ok", database: "connected" });
  });
});

describe("POST /api/tasks", () => {
  test("creates a task with defaults and timestamps", async () => {
    // Surrounding spaces should be trimmed by the schema.
    const task = await createTask("  Buy milk  ");
    assert.equal(task.title, "Buy milk");
    assert.equal(task.completed, false);
    assert.ok(task._id);
    assert.ok(task.createdAt);
    // The internal version key is disabled.
    assert.equal(task.__v, undefined);
  });

  test("rejects a missing or blank title with 400", async () => {
    const missing = await request(app).post("/api/tasks").send({}).expect(400);
    assert.equal(missing.body.error, "Title is required");
    const blank = await request(app).post("/api/tasks").send({ title: "   " }).expect(400);
    assert.equal(blank.body.error, "Title is required");
  });

  test("rejects a title longer than 200 characters", async () => {
    const res = await request(app).post("/api/tasks").send({ title: "x".repeat(201) }).expect(400);
    assert.equal(res.body.error, "Title must be 200 characters or fewer");
  });

  test("rejects a non-string title such as a query operator", async () => {
    const res = await request(app).post("/api/tasks").send({ title: { $gt: "" } }).expect(400);
    assert.equal(res.body.error, "title has an invalid value");
  });

  test("ignores fields clients may not set", async () => {
    const res = await request(app)
      .post("/api/tasks")
      .send({ title: "Read", createdAt: "2000-01-01T00:00:00.000Z" })
      .expect(201);
    assert.notEqual(res.body.createdAt, "2000-01-01T00:00:00.000Z");
  });

  test("returns 400 for malformed JSON", async () => {
    const res = await request(app)
      .post("/api/tasks")
      .set("Content-Type", "application/json")
      .send('{"title": ')
      .expect(400);
    assert.ok(res.body.error);
  });
});

describe("GET /api/tasks", () => {
  test("lists tasks newest first", async () => {
    await createTask("first");
    await createTask("second");
    const res = await request(app).get("/api/tasks").expect(200);
    assert.deepEqual(
      res.body.map((task) => task.title),
      ["second", "first"],
    );
  });
});

describe("GET /api/tasks/:id", () => {
  test("returns one task", async () => {
    const created = await createTask("Walk the dog");
    const res = await request(app).get(`/api/tasks/${created._id}`).expect(200);
    assert.equal(res.body.title, "Walk the dog");
  });

  test("returns 400 for a malformed id and 404 for an unknown one", async () => {
    await request(app).get("/api/tasks/not-an-id").expect(400);
    await request(app).get("/api/tasks/000000000000000000000000").expect(404);
  });
});

describe("PATCH /api/tasks/:id", () => {
  test("updates completed and title", async () => {
    const created = await createTask("Draft");
    const res = await request(app)
      .patch(`/api/tasks/${created._id}`)
      .send({ completed: true, title: "Final" })
      .expect(200);
    assert.equal(res.body.completed, true);
    assert.equal(res.body.title, "Final");
  });

  test("validates updates", async () => {
    const created = await createTask("Keep me");
    const blank = await request(app).patch(`/api/tasks/${created._id}`).send({ title: "" }).expect(400);
    assert.equal(blank.body.error, "Title is required");
    const badFlag = await request(app).patch(`/api/tasks/${created._id}`).send({ completed: "maybe" }).expect(400);
    assert.equal(badFlag.body.error, "completed has an invalid value");
    // The stored task is unchanged.
    const stored = await Task.findById(created._id);
    assert.equal(stored.title, "Keep me");
  });

  test("returns 404 for an unknown id", async () => {
    await request(app).patch("/api/tasks/000000000000000000000000").send({ completed: true }).expect(404);
  });
});

describe("DELETE /api/tasks/:id", () => {
  test("deletes a task, then reports it missing", async () => {
    const created = await createTask("Temporary");
    await request(app).delete(`/api/tasks/${created._id}`).expect(204);
    await request(app).delete(`/api/tasks/${created._id}`).expect(404);
    assert.equal(await Task.countDocuments(), 0);
  });
});

describe("unknown routes", () => {
  test("return a JSON 404", async () => {
    const res = await request(app).get("/api/nope").expect(404);
    assert.equal(res.body.error, "Route not found: GET /api/nope");
  });
});
