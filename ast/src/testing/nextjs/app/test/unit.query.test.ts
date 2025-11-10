// @ts-nocheck
import { useUserQuery } from "../../lib/hooks/useUserQuery";

describe("unit: user query hook", () => {
  it("refetches user data", async () => {
    const query = useUserQuery("user-123");

    await query.refetch();

    expect(query.isLoading).toBe(false);
    console.log("query.refetch called:", query.data);
  });

  it("invalidates query cache", () => {
    const query = useUserQuery("user-123");

    query.invalidate();

    expect(query.data).toBeNull();
    console.log("query.invalidate called: cache cleared");
  });

  it("resets query state", async () => {
    const query = useUserQuery("user-123");

    await query.refetch();

    query.reset();

    expect(query.data).toBeNull();
    expect(query.error).toBeNull();
    expect(query.isLoading).toBe(false);
    console.log("query.reset called: state reset");
  });

  it("loads initial data", async () => {
    const query = useUserQuery("user-456");

    await new Promise((resolve) => setTimeout(resolve, 100));

    expect(query.data).toBeDefined();
    console.log("Initial data loaded:", query.data);
  });
});

describe("unit: query hook workflows", () => {
  it("handles refetch after invalidate", async () => {
    const query = useUserQuery("user-123");

    await query.refetch();
    expect(query.data).toBeDefined();

    query.invalidate();
    expect(query.data).toBeNull();

    await query.refetch();
    expect(query.data).toBeDefined();

    console.log("Refetch after invalidate workflow completed");
  });

  it("handles reset and refetch", async () => {
    const query = useUserQuery("user-123");

    await query.refetch();
    expect(query.data).toBeDefined();

    query.reset();
    expect(query.data).toBeNull();

    await query.refetch();
    expect(query.data).toBeDefined();

    console.log("Reset and refetch workflow completed");
  });

  it("chains multiple query operations", async () => {
    const query = useUserQuery("user-789");

    await query.refetch();
    expect(query.data).toBeDefined();

    await query.refetch();
    expect(query.data).toBeDefined();

    query.invalidate();
    expect(query.data).toBeNull();

    await query.refetch();
    expect(query.data).toBeDefined();

    query.reset();
    expect(query.data).toBeNull();

    console.log("Chained query operations completed");
  });
});

describe("unit: multiple query instances", () => {
  it("manages separate query instances", async () => {
    const query1 = useUserQuery("user-001");
    const query2 = useUserQuery("user-002");

    await query1.refetch();
    await query2.refetch();

    expect(query1.data).toBeDefined();
    expect(query2.data).toBeDefined();

    query1.invalidate();
    expect(query1.data).toBeNull();
    expect(query2.data).toBeDefined();

    query2.reset();
    expect(query2.data).toBeNull();

    console.log("Multiple query instances managed separately");
  });

  it("refetches multiple queries", async () => {
    const queries = [
      useUserQuery("user-1"),
      useUserQuery("user-2"),
      useUserQuery("user-3"),
    ];

    for (const query of queries) {
      await query.refetch();
      expect(query.data).toBeDefined();
    }

    queries[0].invalidate();
    queries[1].reset();

    await queries[0].refetch();
    await queries[1].refetch();

    expect(queries[0].data).toBeDefined();
    expect(queries[1].data).toBeDefined();
    expect(queries[2].data).toBeDefined();

    console.log("Multiple queries refetched successfully");
  });
});

describe("unit: query error handling", () => {
  it("handles errors and resets", async () => {
    const query = useUserQuery("invalid-user");

    await query.refetch();

    if (query.error) {
      query.reset();
      expect(query.error).toBeNull();
    }

    console.log("Error handling and reset completed");
  });

  it("refetches after error", async () => {
    const query = useUserQuery("error-user");

    await query.refetch();

    query.reset();

    await query.refetch();

    console.log("Refetch after error completed");
  });
});
