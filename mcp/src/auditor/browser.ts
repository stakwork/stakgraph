import { Stagehand } from "@browserbasehq/stagehand";
import { JobModel } from "./types.js";

interface StagehandModel {
  modelName: string;
  apiKey: string;
  baseURL?: string;
}

function deriveModel(job: JobModel): StagehandModel {
  const raw = job.model ?? "";
  if (job.provider === "openrouter") {
    return {
      modelName: raw.replace(/^openrouter\//, ""),
      apiKey: job.apiKey,
      baseURL: job.host || "https://openrouter.ai/api/v1",
    };
  }
  return { modelName: raw, apiKey: job.apiKey };
}

export class AuditBrowser {
  private stagehand?: Stagehand;
  private readonly model: JobModel;

  constructor(model: JobModel) {
    this.model = model;
  }

  private async ensure(): Promise<Stagehand> {
    if (this.stagehand) return this.stagehand;
    const cfg = deriveModel(this.model);
    const sh = new Stagehand({
      env: "LOCAL",
      domSettleTimeout: 60000,
      localBrowserLaunchOptions: {
        headless: true,
        viewport: { width: 1024, height: 768 },
      },
      model: {
        modelName: cfg.modelName,
        apiKey: cfg.apiKey,
        ...(cfg.baseURL ? { baseURL: cfg.baseURL } : {}),
      },
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
