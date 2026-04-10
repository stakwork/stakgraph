// @ts-nocheck
import { convertSatsToUSD } from "../../lib/currency";

// Pattern 1: Fetch response mock with inline object literal.
// mockBody is a pair arrow function that returns a literal — no real outgoing Calls,
// and nothing in this file calls res.mockBody() — so it is pure noise.
// It should be pruned from the graph.
describe("notifications api client", () => {
  const mockNotification = { id: 1, title: "New Bounty", amount: 5000 };

  beforeEach(() => {
    global.fetch = jest.fn().mockResolvedValue({
      mockBody: async () => mockNotification,
      ok: true,
    });
  });

  it("fetches notification and formats amount", () => {
    const formatted = convertSatsToUSD(mockNotification.amount);
    expect(formatted).toBeDefined();
  });
});

// Pattern 2: Mock object assigned to a local Var.
// get/post are pair arrow functions with no real outgoing Calls.
// Even though they live inside a Var (mockApiClient), they should be pruned
// because they do not call any tracked function and are never called themselves.
const mockApiClient = {
  get: async () => ({ data: [], total: 0 }),
  post: async () => ({ data: {}, created: true }),
};

describe("mocked api client usage", () => {
  it("confirms mock structure", () => {
    expect(mockApiClient).toBeDefined();
  });
});

// Pattern 3: Deeply nested mock factory passed as argument.
// status is a pair arrow function buried in an argument object — no real Calls.
// Should be pruned.
jest.mock("../../lib/api/apiClient", () => ({
  status: async () => ({ ok: true }),
}));

describe("mocked module factory", () => {
  it("uses mocked module", () => {
    expect(true).toBe(true);
  });
});
