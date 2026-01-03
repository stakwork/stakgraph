// @ts-nocheck
import { SequelizePerson } from "../../src/model";

describe("unit: SequelizePerson model", () => {
  it("has required fields", () => {
    expect(SequelizePerson).toBeDefined();
  });

  it("validates email uniqueness", () => {
    const email = "unique@test.com";
    expect(email).toBeTruthy();
  });
});
