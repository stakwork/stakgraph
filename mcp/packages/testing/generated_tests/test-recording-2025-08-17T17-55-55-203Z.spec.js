import { test, expect } from '@playwright/test';
    
  test('User interaction replay', async ({ page }) => {
    // Navigate to the page
    await page.goto('http://localhost:3001');
    
    // Wait for page to load
    await page.waitForLoadState('networkidle');
    
    // Set viewport size to match recorded session
    await page.setViewportSize({ 
      width: 856, 
      height: 785 
    });
  
    // Click on input "Enter username (defaults to 'dev-user')"
  await page.click('#mock-username');

  await page.waitForTimeout(3029);

  // Fill input: #mock-username
  await page.fill('#mock-username', 'dev-user');

  // Click on button "Signing in..."
  await page.click('[data-testid="mock-signin-button"]');

  await page.waitForTimeout(1543);

  // Click on button "Stakgraph"
  await page.click('button:has-text("Stakgraph")');

  await page.waitForTimeout(869);

  // Click on button "Settings"
  await page.click('button:has-text("Settings")');

  await page.waitForTimeout(1033);

  // Click on button "Tasks"
  await page.click('button:has-text("Tasks")');

  await page.waitForTimeout(732);

  // Click on button "Dashboard"
  await page.click('button:has-text("Dashboard")');

  await page.waitForTimeout(1610);

  // Click on html
  await page.click('html.dark');

  await page.waitForTimeout(883);

  // Click on html
  await page.click('html.dark');

  await page.waitForTimeout(1021);

  // Click on button "Settings"
  await page.click('button:has-text("Settings")');

  await page.waitForTimeout(1184);

  // Click on button "Dashboard"
  await page.click('button:has-text("Dashboard")');


  
    await page.waitForTimeout(432);
  });