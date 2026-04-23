// @ts-nocheck

// @ast node: UnitTest "unit: Calculator add"
describe("unit: Calculator add", () => {
  it("adds two positive numbers", () => {
    expect(2 + 2).toBe(4);
  });
});

// @ast node: UnitTest "unit: Calculator multiply"
describe("unit: Calculator multiply", () => {
  it("multiplies correctly", () => {
    expect(3 * 4).toBe(12);
  });
});
