import { test, expect } from "@playwright/test";

test("User interaction replay with form elements", async ({ page }) => {
  await page.goto("http://localhost:3000/preact/frame.html");

  await page.waitForLoadState("networkidle");

  await page.setViewportSize({ width: 1280, height: 720 });

  const showFormButton = page.locator('button:text("Show Form Elements")');
  await showFormButton.waitFor({ state: "visible" });
  await showFormButton.click();

  await page.waitForTimeout(500);

  const checkbox = page.locator('[data-testid="staktrak-checkbox"]');
  await checkbox.waitFor({ state: "visible" });

  await checkbox.check();

  await expect(checkbox).toBeChecked();

  await page.waitForTimeout(500);

  await checkbox.uncheck();

  await expect(checkbox).not.toBeChecked();

  await page.waitForTimeout(500);

  const radioOption2 = page.locator('[data-testid="staktrak-radio-2"]');
  await radioOption2.waitFor({ state: "visible" });
  await radioOption2.check();

  await expect(radioOption2).toBeChecked();

  const radioOption3 = page.locator('[data-testid="staktrak-radio-3"]');
  await radioOption3.check();

  await expect(radioOption3).toBeChecked();
  await expect(radioOption2).not.toBeChecked();

  await page.waitForTimeout(500);

  const selectElement = page.locator('[data-testid="staktrak-select"]');
  await selectElement.waitFor({ state: "visible" });

  await selectElement.selectOption({ value: "banana" });

  await expect(selectElement).toHaveValue("banana");

  await page.waitForTimeout(500);

  await selectElement.selectOption({ label: "Cherry" });

  await expect(selectElement).toHaveValue("cherry");

  await page.waitForTimeout(1000);

  await checkbox.check();
  await radioOption2.check();
  await selectElement.selectOption({ label: "Durian" });

  await expect(checkbox).toBeChecked();
  await expect(radioOption2).toBeChecked();
  await expect(radioOption3).not.toBeChecked();
  await expect(selectElement).toHaveValue("durian");

  await page.waitForTimeout(2500);
});
