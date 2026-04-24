// @ts-nocheck

function formatDate(date: Date): string {
  return date.toISOString();
}

// @ast node: UnitTest "unit: formatDate utility"
describe("unit: formatDate utility", () => {
  it("formats date correctly", () => {
    const date = new Date("2024-01-01");
    expect(formatDate(date)).toContain("2024");
  });
});

describe.skip("unit: skipped tests", () => {
  it("should be skipped", () => {
    expect(true).toBe(false);
  });
});

test.todo("unit: future implementation");
// @ast node: Function "formatDate"
// @ast node: UnitTest "unit: skipped tests"
// @ast node: UnitTest "unit: future implementation"
