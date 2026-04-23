// @ast node: E2eTest "navigate to page test"
// @ast edge: Contains <- File "e2e.playwright.each.test.ts" "src/testing/nextjs/app/test/e2e.playwright.each.test.ts"
describe("navigate to page test", () => {
  it("navigates ", async () => {
    await page.goto("http://localhost:3000/items");
  });
});
