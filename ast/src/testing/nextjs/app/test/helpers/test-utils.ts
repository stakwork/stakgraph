// @ast node: Function "createTestPayload"
// @ast edge: Contains <- File "test-utils.ts" "src/testing/nextjs/app/test/helpers/test-utils.ts"
export function createTestPayload(overrides: any = {}) {
  return {
    title: "Test Item",
    price: 100,
    ...overrides,
  };
}

// @ast node: Function "mockApiResponse"
// @ast edge: Contains <- File "test-utils.ts" "src/testing/nextjs/app/test/helpers/test-utils.ts"
export const mockApiResponse = (status: number, data: any) => ({
  status,
  json: async () => data,
  ok: status >= 200 && status < 300,
});

// @ast node: Function "setupTestDatabase"
// @ast edge: Contains <- File "test-utils.ts" "src/testing/nextjs/app/test/helpers/test-utils.ts"
export async function setupTestDatabase(): Promise<void> {
  console.log("Setting up test database...");
}

// @ast node: Function "fetchItems"
// @ast edge: Contains <- File "test-utils.ts" "src/testing/nextjs/app/test/helpers/test-utils.ts"
export async function fetchItems() {
  return fetch("http://localhost:3000/api/items");
}

// @ast node: Function "createItem"
// @ast edge: Contains <- File "test-utils.ts" "src/testing/nextjs/app/test/helpers/test-utils.ts"
export async function createItem(data: any) {
  return fetch("http://localhost:3000/api/items", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}
// @ast node: Request "http://localhost:3000/api/items"
// @ast node: Request "http://localhost:3000/api/items"
