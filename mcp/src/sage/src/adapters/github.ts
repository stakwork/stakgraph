import { Octokit } from "@octokit/rest";
import { BaseAdapter } from "./adapter.js";
import { Message } from "../types/index.js";
import * as fs from "fs";
import * as path from "path";

export class GitHubIssueAdapter extends BaseAdapter {
  private octokit: Octokit;
  private owner: string;
  private repo: string;
  private processedIssues: Set<number> = new Set();
  private dataDir: string;
  private persistFilePath: string;

  constructor(
    githubToken: string,
    owner: string,
    repo: string,
    dataDir: string = "./data"
  ) {
    super();
    this.octokit = new Octokit({ auth: githubToken });
    this.owner = owner;
    this.repo = repo;
    this.dataDir = dataDir;
    this.persistFilePath = path.join(
      this.dataDir,
      `github-${owner}-${repo}-processed-issues.json`
    );
  }

  async initialize(): Promise<void> {
    console.log(`Initializing GitHub adapter for ${this.owner}/${this.repo}`);

    // Ensure data directory exists
    await this.ensureDataDirExists();

    // Load previously processed issues
    await this.loadProcessedIssues();

    // Start polling for new issues
    setInterval(async () => {
      try {
        await this.checkForNewIssues();
      } catch (error) {
        console.error("Error checking for new GitHub issues:", error);
      }
    }, 60000); // Check every minute
  }

  async sendResponse(chatId: string, message: Message): Promise<void> {
    // Extract issue number from chatId
    const issueNumber = parseInt(chatId.replace("github-issue-", ""), 10);

    if (isNaN(issueNumber)) {
      throw new Error(`Invalid GitHub issue chat ID: ${chatId}`);
    }

    console.log(`Sending message to GitHub issue #${issueNumber}`);
    await this.octokit.issues.createComment({
      owner: this.owner,
      repo: this.repo,
      issue_number: issueNumber,
      body: message.content,
    });
  }

  private async ensureDataDirExists(): Promise<void> {
    if (!fs.existsSync(this.dataDir)) {
      await fs.promises.mkdir(this.dataDir, { recursive: true });
      console.log(`Created data directory: ${this.dataDir}`);
    }
  }

  private async loadProcessedIssues(): Promise<void> {
    try {
      if (fs.existsSync(this.persistFilePath)) {
        const data = await fs.promises.readFile(this.persistFilePath, "utf-8");
        const issues = JSON.parse(data);
        this.processedIssues = new Set(issues);
        console.log(
          `Loaded ${this.processedIssues.size} previously processed issues`
        );
      } else {
        console.log("No previously processed issues found");
      }
    } catch (error) {
      console.error("Error loading processed issues:", error);
      // Continue with empty set if loading fails
    }
  }

  private async saveProcessedIssues(): Promise<void> {
    try {
      const issues = Array.from(this.processedIssues);
      await fs.promises.writeFile(
        this.persistFilePath,
        JSON.stringify(issues),
        "utf-8"
      );
      console.log(`Saved ${issues.length} processed issues`);
    } catch (error) {
      console.error("Error saving processed issues:", error);
    }
  }

  async checkForNewIssues(): Promise<void> {
    console.log("Checking for new GitHub issues...");
    const issues = await this.octokit.issues.listForRepo({
      owner: this.owner,
      repo: this.repo,
      state: "open",
      sort: "created",
      direction: "desc",
      per_page: 10,
    });

    let newIssuesProcessed = false;

    for (const issue of issues.data) {
      // Skip if already processed
      if (this.processedIssues.has(issue.number)) {
        continue;
      }

      // Check if @hive is mentioned
      if (issue.body && issue.body.includes("@hive")) {
        const chatId = `github-issue-${issue.number}`;
        const message: Message = {
          role: "user",
          content: issue.body,
        };

        console.log(
          `Processing GitHub issue #${issue.number} with @hive mention`
        );
        this.processedIssues.add(issue.number);
        newIssuesProcessed = true;

        try {
          await this.messageCallback(chatId, message);
        } catch (error) {
          console.error(
            `Error processing GitHub issue #${issue.number}:`,
            error
          );
        }
      }
    }

    // Save processed issues if any new ones were added
    if (newIssuesProcessed) {
      await this.saveProcessedIssues();
    }
  }
}
