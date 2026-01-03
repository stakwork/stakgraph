// @ts-nocheck

function formatDate(date: Date): string {
  return date.toISOString();
}

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
