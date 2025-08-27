describe('e2e: puppeteer basic', () => {
  it('opens page', async () => {
    await page.goto('http://localhost:3000/items');
    const el = await page.$('body');
  });
});
