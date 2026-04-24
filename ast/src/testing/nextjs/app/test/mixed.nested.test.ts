// @ast node: IntegrationTest "integration: api and page mix"
// @ast edge: Contains <- File "mixed.nested.test.ts" "app/test/mixed.nested.test.ts"
describe("integration: api and page mix", () => {
  describe("nested unit block", () => {
    it("does simple assertion", () => {});
  });
  it("hits api", async () => {
    await fetch("http://localhost:3000/api/items");
  });
});
// @ast node: UnitTest "nested unit block"
