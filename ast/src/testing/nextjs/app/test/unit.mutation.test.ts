// @ts-nocheck
import { useMutation } from "../../lib/hooks/useMutation";

interface TestData {
  id: number;
  name: string;
}

describe("unit: mutation hook", () => {
  it("performs mutation successfully", async () => {
    const mutation = useMutation<TestData>("/api/test");
    const testData: TestData = { id: 1, name: "Test" };

    await mutation.mutate(testData);

    expect(mutation.isLoading).toBe(false);
    console.log("Mutation completed:", mutation.data);
  });

  it("resets mutation state", async () => {
    const mutation = useMutation<TestData>("/api/test");
    const testData: TestData = { id: 2, name: "Reset Test" };

    await mutation.mutate(testData);
    expect(mutation.data).not.toBeNull();

    mutation.reset();

    expect(mutation.data).toBeNull();
    expect(mutation.error).toBeNull();
    expect(mutation.isLoading).toBe(false);
    console.log("Mutation state reset successfully");
  });

  it("handles multiple mutations", async () => {
    const mutation = useMutation<TestData>("/api/test");

    await mutation.mutate({ id: 1, name: "First" });
    expect(mutation.data).toBeDefined();

    await mutation.mutate({ id: 2, name: "Second" });
    expect(mutation.data).toBeDefined();

    await mutation.mutate({ id: 3, name: "Third" });
    expect(mutation.data).toBeDefined();

    console.log("Multiple mutations executed");
  });
});

describe("unit: mutation with different endpoints", () => {
  it("creates user via mutation", async () => {
    const userMutation = useMutation<{ name: string; email: string }>(
      "/api/users"
    );

    await userMutation.mutate({
      name: "John Doe",
      email: "john@example.com",
    });

    expect(userMutation.isLoading).toBe(false);
    console.log("User mutation completed");
  });

  it("creates item via mutation", async () => {
    const itemMutation = useMutation<{ title: string; price: number }>(
      "/api/items"
    );

    await itemMutation.mutate({
      title: "Test Item",
      price: 99.99,
    });

    expect(itemMutation.isLoading).toBe(false);
    console.log("Item mutation completed");
  });

  it("handles mutation workflow", async () => {
    const mutation = useMutation<TestData>("/api/workflow");

    await mutation.mutate({ id: 1, name: "Start" });
    expect(mutation.data).toBeDefined();

    mutation.reset();
    expect(mutation.data).toBeNull();

    await mutation.mutate({ id: 2, name: "Restart" });
    expect(mutation.data).toBeDefined();

    console.log("Mutation workflow completed");
  });
});

describe("unit: mutation error handling", () => {
  it("resets after error", async () => {
    const mutation = useMutation<TestData>("/api/error-endpoint");

    await mutation.mutate({ id: 1, name: "Error Test" });

    mutation.reset();

    expect(mutation.error).toBeNull();
    expect(mutation.data).toBeNull();
    console.log("Error state reset");
  });
});
