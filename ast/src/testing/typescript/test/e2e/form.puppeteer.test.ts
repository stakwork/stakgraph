// @ts-nocheck
import puppeteer from "puppeteer";

describe("e2e: form submission", () => {
  it("submits contact form", async () => {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    await page.goto("http://localhost:3000/contact");
    await page.type('input[name="email"]', "test@test.com");
    await page.type('textarea[name="message"]', "Hello");
    await page.click('button[type="submit"]');
    await browser.close();
  });
});
