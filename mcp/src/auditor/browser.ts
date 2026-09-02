import { Stagehand, AISdkClient } from "@browserbasehq/stagehand";
import type { LanguageModel } from "ai";

export class AuditBrowser {
  private stagehand?: Stagehand;
  private readonly model: LanguageModel;

  constructor(model: LanguageModel) {
    this.model = model;
  }

  private async ensure(): Promise<Stagehand> {
    if (this.stagehand) return this.stagehand;
    const sh = new Stagehand({
      env: "LOCAL",
      domSettleTimeout: 60000,
      localBrowserLaunchOptions: {
        headless: true,
        viewport: { width: 1024, height: 768 },
      },
      llmClient: new AISdkClient({ model: this.model as any }),
    });
    await sh.init();
    this.stagehand = sh;
    return sh;
  }

  private async page(sh: Stagehand) {
    const page = sh.context.activePage();
    if (!page) throw new Error("no active page available");
    return page;
  }

  async open(url: string): Promise<{ url: string; ok: boolean }> {
    const sh = await this.ensure();
    const page = await this.page(sh);
    await page.goto(url);
    return { url, ok: true };
  }

  async act(action: string): Promise<{ action: string; result: unknown }> {
    const sh = await this.ensure();
    const result = await sh.act(action);
    return { action, result };
  }

  async observe(
    instruction: string,
  ): Promise<{ instruction: string; observations: unknown }> {
    const sh = await this.ensure();
    const observations = await sh.observe(instruction);
    return { instruction, observations };
  }

  async extract(
    instruction: string,
  ): Promise<{ instruction: string; extraction: unknown }> {
    const sh = await this.ensure();
    const extraction = await sh.extract(instruction);
    return { instruction, extraction };
  }

  async currentUrl(): Promise<string> {
    const sh = await this.ensure();
    const page = await this.page(sh);
    return page.url();
  }

  async screenshot(): Promise<string> {
    const sh = await this.ensure();
    const page = await this.page(sh);
    const buffer = await page.screenshot({ fullPage: false });
    return buffer.toString("base64");
  }

  async close(): Promise<void> {
    if (this.stagehand) {
      const sh = this.stagehand;
      this.stagehand = undefined;
      await sh.close();
    }
  }
}
