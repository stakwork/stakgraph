// @ast node: IntegrationTest "integration: items and person apis"
// @ast edge: Contains <- File "integration.api.spec.ts" "app/test/integration.api.spec.ts"
describe("integration: items and person apis", () => {
  beforeAll(async () => {});
  it("fetches items", async () => {
    await fetch("http://localhost:3000/api/items");
  });
  it("creates person", async () => {
    await fetch("http://localhost:3000/api/person", { method: "POST" });
  });
});
