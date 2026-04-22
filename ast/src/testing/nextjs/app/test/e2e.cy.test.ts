// @ast node: E2eTest "e2e: cypress style flow"
// @ast edge: Contains <- File "e2e.cy.test.ts" "src/testing/nextjs/app/test/e2e.cy.test.ts"
describe("e2e: cypress style flow", () => {
  it("visits items", () => {
    cy.visit("http://localhost:3000/items");
    cy.get('input[placeholder="Title"]').type("X");
  });
});
