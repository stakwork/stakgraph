// @ast node: E2eTest "e2e: puppeteer basic"
// @ast edge: Contains <- File "puppeteer.flow.test.ts" "app/test/puppeteer.flow.test.ts"
describe("e2e: puppeteer basic", () => {
  it("opens page", async () => {
    await page.goto("http://localhost:3000/items");
    const el = await page.$("body");
  });
});
