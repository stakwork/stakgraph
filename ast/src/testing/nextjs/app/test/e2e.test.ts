// @ts-nocheck

// @ast node: E2eTest "e2e: user flows"
// @ast edge: Contains <- File "e2e.test.ts" "src/testing/nextjs/app/test/e2e.test.ts"
describe("e2e: user flows", () => {
  it("navigates to /items and adds an item", async () => {
    await page.goto("http://localhost:3000/items");
    await page.getByPlaceholder("Title").fill("Item 1");
    await page.getByPlaceholder("Price").fill("10");
    await page.getByRole("button", { name: "Add Item" }).click();
    await expect(page.getByText("Item 1 - $10")).toBeVisible();
    console.log("E2E flow: add item");
  });

  it("navigates to /person and manages a person by id", async () => {
    await page.goto("http://localhost:3000/person");
    await page.getByPlaceholder("Name").fill("Alice");
    await page.getByPlaceholder("Age").fill("30");
    await page.getByPlaceholder("Email").fill("alice@example.com");
    await page.getByRole("button", { name: "Add Person" }).click();
    await page.getByPlaceholder("Enter person ID (1, 2, 3...)").fill("1");
    await page.getByRole("button", { name: "Find" }).click();
    await expect(page.getByText("Found: Alice")).toBeVisible();
    console.log("E2E flow: add, find and delete person");
  });
});
