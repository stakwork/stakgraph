import request from "supertest";
import { app } from "../api/routes";

describe("Users API Integration Tests", () => {
  test("GET /api/users returns user list", async () => {
    const response = await request(app)
      .get("/api/users")
      .expect("Content-Type", /json/)
      .expect(200);

    expect(Array.isArray(response.body)).toBe(true);
  });

  test("POST /api/users creates new user", async () => {
    const newUser = { name: "Test User", email: "test@example.com" };

    const response = await request(app)
      .post("/api/users")
      .send(newUser)
      .expect(201);

    expect(response.body.name).toBe(newUser.name);
    expect(response.body.email).toBe(newUser.email);
  });

  it("PUT /api/users/:id updates user", async () => {
    const updatedData = { name: "Updated Name", email: "updated@example.com" };

    const response = await request(app)
      .put("/api/users/1")
      .send(updatedData)
      .expect(200);

    expect(response.body.name).toBe(updatedData.name);
  });

  it("DELETE /api/users/:id removes user", async () => {
    await request(app).delete("/api/users/1").expect(200);
  });
});

describe("API Error Handling", () => {
  test("returns 404 for non-existent user", async () => {
    await request(app).get("/api/users/99999").expect(404);
  });

  test("returns 400 for invalid request body", async () => {
    await request(app).post("/api/users").send({}).expect(400);
  });
});
