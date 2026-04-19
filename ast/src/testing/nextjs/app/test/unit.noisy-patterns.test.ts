// @ts-nocheck
import { createApiHandlers, deepConfig, apiService } from "../../lib/noisy-patterns";

describe("unit: noisy patterns - category 3 (returned from function)", () => {
  function helperInsideDescribe(data: any) {
    return createApiHandlers().onSuccess(data);
  }

  it("calls onSuccess handler", () => {
    const handlers = createApiHandlers();
    const result = handlers.onSuccess({ amount: 42 });
    expect(result).toBeDefined();
  });

  it("calls onError handler", () => {
    const handlers = createApiHandlers();
    handlers.onError(new Error("fail"));
  });
});

describe("unit: noisy patterns - category 4 (deep config)", () => {
  it("accesses deeply nested handler", async () => {
    const result = await deepConfig.level1.level2.handler();
    expect(result.ready).toBe(true);
  });

  it("accesses deeply nested transform", () => {
    const result = deepConfig.level1.level2.transform("hello");
    expect(result).toBe("HELLO");
  });
});

describe("unit: noisy patterns - category 5 (class method return)", () => {
  it("uses parse from getHandlers", () => {
    const handlers = apiService.getHandlers();
    const data = handlers.parse('{"a":1}');
    expect(data.a).toBe(1);
  });

  it("uses serialize from getHandlers", () => {
    const handlers = apiService.getHandlers();
    const json = handlers.serialize({ b: 2 });
    expect(json).toBe('{"b":2}');
  });
});
