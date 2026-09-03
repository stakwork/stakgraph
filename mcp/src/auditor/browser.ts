import { Stagehand, AISdkClient } from "@browserbasehq/stagehand";
import type { LanguageModel } from "ai";

export interface NetEntry {
  method: string;
  url: string;
  status: number;
  type: string;
  mimeType?: string;
  error?: string;
}

export interface ConsoleEntry {
  level: string;
  text: string;
  url?: string;
}

const NET_TYPES = new Set(["XHR", "Fetch", "Document"]);
const NET_CAP = 200;
const CONSOLE_CAP = 200;
const ACTION_CAP = 50;
const CONSOLE_TEXT_CAP = 500;

export class AuditBrowser {
  private stagehand?: Stagehand;
  private readonly model: LanguageModel;

  private readonly attachedSessions = new Set<string>();
  private readonly consoleAttached = new Set<string>();
  private readonly pending = new Map<string, { method: string; url: string; type: string }>();

  private networkBuf: NetEntry[] = [];
  private networkSinceAction: NetEntry[] = [];
  private consoleBuf: ConsoleEntry[] = [];
  private consoleSinceAction: ConsoleEntry[] = [];

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

  private async page(sh: Stagehand): Promise<any> {
    const page = sh.context.activePage() as any;
    if (!page) throw new Error("no active page available");
    this.attach(page);
    return page;
  }

  private pushNet(entry: NetEntry): void {
    this.networkBuf.push(entry);
    if (this.networkBuf.length > NET_CAP) this.networkBuf.shift();
    this.networkSinceAction.push(entry);
    if (this.networkSinceAction.length > ACTION_CAP) this.networkSinceAction.shift();
  }

  private pushConsole(entry: ConsoleEntry): void {
    if (entry.text && entry.text.length > CONSOLE_TEXT_CAP) {
      entry.text = entry.text.slice(0, CONSOLE_TEXT_CAP) + "…";
    }
    this.consoleBuf.push(entry);
    if (this.consoleBuf.length > CONSOLE_CAP) this.consoleBuf.shift();
    this.consoleSinceAction.push(entry);
    if (this.consoleSinceAction.length > ACTION_CAP) this.consoleSinceAction.shift();
  }

  private attach(page: any): void {
    try {
      const targetId = page?.targetId?.() ?? "default";
      if (!this.consoleAttached.has(targetId)) {
        this.consoleAttached.add(targetId);
        page.on("console", (msg: any) => {
          try {
            this.pushConsole({
              level: msg.type?.() ?? "log",
              text: msg.text?.() ?? "",
              url: msg.location?.()?.url,
            });
          } catch {
            /* ignore console decode errors */
          }
        });
      }

      const session = page?.mainFrame?.()?.session;
      if (!session) return;
      const sid = session.id ?? "default";
      if (this.attachedSessions.has(sid)) return;
      this.attachedSessions.add(sid);

      this.subscribe(session);
    } catch {
      /* a CDP quirk must never fail the browser action */
    }
  }

  private subscribe(session: any): void {
    session.send("Runtime.enable").catch(() => {});
    session.on("Runtime.exceptionThrown", (p: any) => {
      const d = p?.exceptionDetails;
      this.pushConsole({
        level: "error",
        text:
          d?.exception?.description ??
          d?.text ??
          "uncaught exception",
        url: d?.url,
      });
    });

    session.send("Log.enable").catch(() => {});
    session.on("Log.entryAdded", (p: any) => {
      const e = p?.entry;
      if (!e) return;
      if (e.level === "error" || e.level === "warning") {
        this.pushConsole({ level: e.level, text: e.text ?? "", url: e.url });
      }
    });

    session.send("Network.enable").catch(() => {});
    session.on("Network.requestWillBeSent", (p: any) => {
      const type = p?.type ?? "";
      if (p?.redirectResponse && NET_TYPES.has(type)) {
        this.pushNet({
          method: p?.request?.method ?? "GET",
          url: p?.redirectResponse?.url ?? p?.request?.url ?? "",
          status: p?.redirectResponse?.status ?? 0,
          type,
        });
      }
      this.pending.set(p.requestId, {
        method: p?.request?.method ?? "GET",
        url: p?.request?.url ?? "",
        type,
      });
      if (this.pending.size > 500) {
        const first = this.pending.keys().next().value;
        if (first !== undefined) this.pending.delete(first);
      }
    });
    session.on("Network.responseReceived", (p: any) => {
      const type = p?.type ?? this.pending.get(p?.requestId)?.type ?? "";
      if (!NET_TYPES.has(type)) {
        this.pending.delete(p?.requestId);
        return;
      }
      const req = this.pending.get(p?.requestId);
      this.pushNet({
        method: req?.method ?? "GET",
        url: p?.response?.url ?? req?.url ?? "",
        status: p?.response?.status ?? 0,
        type,
        mimeType: p?.response?.mimeType,
      });
      this.pending.delete(p?.requestId);
    });
    session.on("Network.loadingFailed", (p: any) => {
      const req = this.pending.get(p?.requestId);
      const type = req?.type ?? p?.type ?? "";
      if (req && !NET_TYPES.has(type)) {
        this.pending.delete(p?.requestId);
        return;
      }
      this.pushNet({
        method: req?.method ?? "",
        url: req?.url ?? "(unknown)",
        status: 0,
        type,
        error: p?.errorText ?? "loading failed",
      });
      this.pending.delete(p?.requestId);
    });
  }

  drainNetworkDelta(): NetEntry[] {
    const d = this.networkSinceAction;
    this.networkSinceAction = [];
    return d;
  }

  drainConsoleDelta(): ConsoleEntry[] {
    const d = this.consoleSinceAction;
    this.consoleSinceAction = [];
    return d;
  }

  snapshotNetwork(): NetEntry[] {
    return [...this.networkBuf];
  }

  snapshotConsole(): ConsoleEntry[] {
    return [...this.consoleBuf];
  }

  async open(url: string): Promise<{ url: string; ok: boolean }> {
    const sh = await this.ensure();
    const page = await this.page(sh);
    await page.goto(url);
    return { url, ok: true };
  }

  async act(action: string): Promise<{ action: string; result: unknown }> {
    const sh = await this.ensure();
    await this.page(sh);
    const result = await sh.act(action);
    return { action, result };
  }

  async observe(
    instruction: string,
  ): Promise<{ instruction: string; observations: unknown }> {
    const sh = await this.ensure();
    await this.page(sh);
    const observations = await sh.observe(instruction);
    return { instruction, observations };
  }

  async extract(
    instruction: string,
  ): Promise<{ instruction: string; extraction: unknown }> {
    const sh = await this.ensure();
    await this.page(sh);
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
