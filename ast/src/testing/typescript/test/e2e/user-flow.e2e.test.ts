// @ts-nocheck
import { test, expect } from "@playwright/test";

test.describe("e2e: user management flow", () => {
  test("creates and lists users", async ({ page }) => {
    await page.goto("http://localhost:3000");
    await page.click('button[name="Add User"]');
    await page.fill('input[name="name"]', "Test User");
    await page.click('button[type="submit"]');
    await expect(page.locator("text=Test User")).toBeVisible();
  });

  test("edits user information", async ({ page }) => {
    await page.goto("http://localhost:3000/users/1");
    await page.fill('input[name="name"]', "Updated Name");
    await page.click('button[type="submit"]');
    await expect(page.locator("text=Updated Name")).toBeVisible();
  });
});
