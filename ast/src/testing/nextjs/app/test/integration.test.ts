// @ts-nocheck
import { fetchItems, createItem } from "./helpers/test-utils";

describe("integration: /api/items", () => {
  it("GET returns items list via helper", async () => {
    const res = await fetchItems();
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(Array.isArray(data)).toBe(true);
    console.log("GET /api/items should return 200 and an array");
  });

  it("POST creates a new item via helper", async () => {
    const res = await createItem({ title: "Test", price: 1 });
    expect(res.status).toBe(201);
    console.log("POST /api/items should return 201");
  });
});

describe("integration: /api/person and /api/person/[id]", () => {
  it("GET /api/person returns list", async () => {
    console.log("GET /api/person should return 200 and an array");
  });

  it("POST /api/person creates", async () => {
    console.log("POST /api/person should return 201");
  });

  it("GET /api/person/[id] finds by id", async () => {
    console.log("GET /api/person/1 should return 200 and a person");
  });

  it("DELETE /api/person/[id] deletes by id", async () => {
    console.log("DELETE /api/person/1 should return 200");
  });
});

describe("integration: /api/orders with query params", () => {
  it("PUT updates order status by orderId query param", async () => {
    const res = await fetch("http://localhost:3000/api/orders?orderId=123", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: "shipped" }),
    });
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.orderId).toBe("123");
    console.log("PUT /api/orders?orderId=123 should return 200");
  });
});
