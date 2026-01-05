// @ts-nocheck
import { registerRoutes } from "../../src/routes";

describe("integration: /person endpoint", () => {
  it("POST creates person via API", async () => {
    const res = await fetch("http://localhost:3000/person", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: "Test", email: "test@test.com" }),
    });
    expect(res.status).toBe(201);
  });

  it("GET retrieves person by id", async () => {
    const res = await fetch("http://localhost:3000/person/1");
    expect(res.status).toBe(200);
  });
});

describe("integration: /api/admin endpoints", () => {
  it("GET lists users", async () => {
    const res = await fetch("http://localhost:3000/api/admin/users");
    expect(res.status).toBe(200);
  });
});
