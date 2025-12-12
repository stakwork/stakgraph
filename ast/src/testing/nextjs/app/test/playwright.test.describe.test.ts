import { test, expect } from '@playwright/test'

test.describe('Composer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('can send a test email', async ({ page }) => {
    await page.getByTestId('send-test-email-button').click()
    await page.getByTestId('skip-gmail-auth-button').click()
    await page.getByTestId('toast').waitFor({ state: 'visible' })
    await expect(page.getByTestId('toast')).toHaveText('Successfully sent the test mail')
  })

  test('can send a message', async ({ page }) => {
    await page.getByTestId('subject').fill('Test Subject')
    await page.getByTestId('send-button').click()
    await expect(page.getByText('Test Subject')).toBeVisible()
  })
})

test.describe.only('Dashboard', () => {
  test('shows user stats', async ({ page }) => {
    await page.goto('/dashboard')
    await expect(page.getByTestId('user-stats')).toBeVisible()
  })
})
