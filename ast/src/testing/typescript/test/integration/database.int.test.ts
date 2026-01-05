// @ts-nocheck
import { sequelize } from "../../src/config";

describe("integration: database connection", () => {
  it("connects to database", async () => {
    await sequelize.authenticate();
    expect(true).toBe(true);
  });

  it("syncs models", async () => {
    await sequelize.sync();
    expect(true).toBe(true);
  });
});
