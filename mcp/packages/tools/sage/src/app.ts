import express from "express";
import bodyParser from "body-parser";
import { StakworkService } from "./services/stakwork.js";
import { GitHubIssueAdapter } from "./adapters/github.js";
import { ChatAdapter, Adapter, EmptyAdapters } from "./adapters/adapter.js";
import { MessagesController } from "./controllers/messages.js";
import { WebhookController } from "./controllers/webhook.js";
import { Message } from "./types/index.js";
import { loadConfig, Config } from "./utils/config.js";

export class App {
  public app: express.Application;
  private stakworkService!: StakworkService;
  private adapters: Record<Adapter, ChatAdapter>;
  private messagesController!: MessagesController;
  private webhookController!: WebhookController;
  private config: Config;

  constructor(parentApp?: any) {
    this.app = parentApp || express();
    const configPath = process.env.SAGE_CONFIG_PATH || "sage_config.json";
    this.config = loadConfig(configPath);
    if (!parentApp) {
      this.configureMiddleware();
    }
    this.initializeServices();
    this.adapters = this.initializeAdapters();
    this.setupControllers();
    this.setupRoutes();
  }

  private configureMiddleware(): void {
    this.app.use(bodyParser.json());
  }

  private initializeServices(): void {
    const workflow_id = parseInt(this.config.workflow_id || "38842", 10);
    const repo = `${this.config.github.owner}/${this.config.github.repo}`;
    this.stakworkService = new StakworkService(
      this.config.stakwork_api_key,
      workflow_id,
      this.config.codeSpaceURL,
      this.config["2b_base_url"],
      this.config.secret,
      this.config.dry_run,
      repo
    );
  }

  private initializeAdapters(): Record<Adapter, ChatAdapter> {
    const adapters = EmptyAdapters();

    const { owner, repo, token } = this.config.github;

    if (token && owner && repo) {
      const DATA_DIR = this.config.data_dir || "./data";
      const githubAdapter = new GitHubIssueAdapter(
        token,
        owner,
        repo,
        DATA_DIR
      );
      adapters["github"] = githubAdapter;

      // Initialize adapter and set up callback
      githubAdapter.initialize().catch((error: any) => {
        console.error("Error initializing GitHub adapter:", error);
      });

      githubAdapter.onMessageReceived(
        async (chatId: string, message: Message) => {
          try {
            await this.messagesController.processMessages(chatId, [message]);
          } catch (error) {
            console.error(
              "Error processing message from GitHub adapter:",
              error
            );
          }
        }
      );
    } else {
      console.warn(
        "GitHub adapter not initialized due to missing configuration"
      );
    }

    return adapters;
  }

  private setupControllers(): void {
    const WEBHOOK_BASE_URL =
      this.config.webhook_url || "http://localhost:3000/webhook";
    this.messagesController = new MessagesController(
      this.stakworkService,
      WEBHOOK_BASE_URL,
      this.adapters
    );
    this.webhookController = new WebhookController(this.adapters);
  }

  private setupRoutes(): void {
    this.app.post("/msg", (req, res) =>
      this.messagesController.handleMessage(req, res)
    );
    this.app.post("/webhook", (req, res) =>
      this.webhookController.handleWebhook(req, res)
    );
  }
}
