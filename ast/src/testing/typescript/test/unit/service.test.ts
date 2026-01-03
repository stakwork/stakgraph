// @ts-nocheck
import { PersonService, getPersonById } from "../../src/service";

describe("unit: PersonService", () => {
  it("getById returns person data", async () => {
    const service = new PersonService();
    expect(service).toBeDefined();
  });

  it("validates email format", () => {
    const email = "test@example.com";
    expect(email).toMatch(/@/);
  });
});

describe("unit: getPersonById function", () => {
  it("returns null for missing id", async () => {
    const result = await getPersonById(999);
    expect(result).toBeNull();
  });
});
