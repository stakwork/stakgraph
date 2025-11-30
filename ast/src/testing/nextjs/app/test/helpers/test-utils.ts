export function createTestPayload(overrides: any = {}) {
  return {
    title: "Test Item",
    price: 100,
    ...overrides
  };
}

export const mockApiResponse = (status: number, data: any) => ({
  status,
  json: async () => data,
  ok: status >= 200 && status < 300
});

export async function setupTestDatabase(): Promise<void> {
  console.log("Setting up test database...");
}

export async function fetchItems() {
  return fetch("http://localhost:3000/api/items");
}

export async function createItem(data: any) {
  return fetch("http://localhost:3000/api/items", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}
