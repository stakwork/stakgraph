describe("navigate to page test", () => {
  it("navigates ", async () => {
    await page.goto("http://localhost:3000/items");
  });
});
