import { test, expect } from "@playwright/test";

test.describe("Dashboard E2E Tests", () => {
  test("should navigate to dashboard", async ({ page }) => {
    await page.goto("http://localhost:3000/dashboard");
    await expect(page).toHaveTitle(/Dashboard/);
  });

  test("should display stats cards", async ({ page }) => {
    await page.goto("http://localhost:3000/dashboard");

    await page.click('[data-testid="stats-toggle"]');
    await page.evaluate(() => (document.body.scrollTop = 0));

    const usersCard = page.locator(".stat-card").first();
    await expect(usersCard).toBeVisible();
  });

  test("should handle user interactions", async ({ page }) => {
    await page.goto("http://localhost:3000/users");

    await page.fill('[data-testid="search-input"]', "John");
    await page.click('[data-testid="search-button"]');

    await expect(page.locator(".user-list")).toContainText("John");
  });
});

test.describe("Authentication Flow", () => {
  test("should redirect unauthenticated users", async ({ page }) => {
    await page.goto("http://localhost:3000/admin");
    await expect(page).toHaveURL(/.*login/);
  });

  test("successful login redirects to dashboard", async ({ page }) => {
    await page.goto("http://localhost:3000/login");

    await page.fill('[name="email"]', "admin@test.com");
    await page.fill('[name="password"]', "password123");
    await page.click('button[type="submit"]');

    await expect(page).toHaveURL(/.*dashboard/);
  });
});
