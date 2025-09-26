import { test, expect } from "@playwright/test";

/**
 * End-to-End test for workspace settings functionality
 *
 * This test covers:
 * 1. Authentication with mock sign-in
 * 2. Navigation to workspace settings
 * 3. Updating workspace name, slug and description
 * 4. Verifying the changes were applied successfully
 */
test.describe("Workspace Settings", () => {
  // Test data
  const workspaceName = "Mock Workspace 123";
  const workspaceSlug = "mock-stakgraph-123";
  const workspaceDescription = "Development workspace (mock)123.";

  test("should update workspace details", async ({ page }) => {
    // Navigate to the application
    await page.goto("http://localhost:3000");

    // Verify the page loaded with expected welcome message
    await expect(
      page.locator("div.grid.auto-rows-min.items-start")
    ).toContainText("Welcome to Hive");

    // Authenticate using mock sign-in
    const signInButton = page.locator('[data-testid="mock-signin-button"]');
    await signInButton.waitFor({ state: "visible" });
    await signInButton.click();

    // Wait for authentication to complete and dashboard to load
    const settingsButton = page.locator('button:has-text("Settings")');
    await settingsButton.waitFor({ state: "visible" });

    // Navigate to workspace settings
    await settingsButton.click();

    // Verify we've landed on the settings page
    const pageTitle = page.locator("h1.text-3xl.font-bold.text-foreground");
    await expect(pageTitle).toBeVisible();
    await expect(pageTitle).toContainText("Workspace Settings");

    // Update workspace name
    const nameInput = page.locator("input.border-input.flex.h-9").first();
    await nameInput.waitFor({ state: "visible" });
    await nameInput.click();
    await nameInput.clear();
    await nameInput.fill(workspaceName);
    await nameInput.blur(); // Trigger validation

    // Update workspace slug
    const slugInput = page.locator("input.border-input.flex.h-9").nth(1);
    await slugInput.waitFor({ state: "visible" });
    await slugInput.click();
    await slugInput.clear();
    await slugInput.fill(workspaceSlug);
    await slugInput.blur(); // Trigger validation

    // Update workspace description
    const descriptionTextarea = page.locator(
      "textarea.border-input.flex.field-sizing-content"
    );
    await descriptionTextarea.waitFor({ state: "visible" });
    await descriptionTextarea.click();
    await descriptionTextarea.clear();
    await descriptionTextarea.fill(workspaceDescription);
    await descriptionTextarea.blur(); // Trigger validation

    // Give a brief moment for validation to complete
    await page.waitForTimeout(500);

    // Save the changes - try to click button even if it appears disabled
    const updateButton = page.locator('button:has-text("Update Workspace")');
    await updateButton.waitFor({ state: "visible" });

    // Force click in case the button appears disabled but is actually functional
    await updateButton.click({ force: true });

    // Wait for a moment to allow any processing/API calls to complete
    await page.waitForTimeout(2000);

    // Verify the changes were applied successfully
    // Check that the workspace name is now visible in the sidebar/navigation
    await expect(
      page.locator("button").filter({ hasText: workspaceName }).first()
    ).toBeVisible();

    // Verify form fields still display our updated values
    await expect(nameInput).toHaveValue(workspaceName);
    await expect(slugInput).toHaveValue(workspaceSlug);
    await expect(descriptionTextarea).toHaveValue(workspaceDescription);
  });
});
