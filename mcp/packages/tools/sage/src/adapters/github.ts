import { Octokit } from "@octokit/rest";
import { BaseAdapter, ChatInfo } from "./adapter.js";
import { Message } from "../types/index.js";
import * as fs from "fs";
import * as path from "path";
import { extractCodespaceUrl } from "../utils/markdown.js";

export class GitHubIssueAdapter extends BaseAdapter {
  private octokit: Octokit;
  private owner: string;
  private repo: string;
  private processedChats: Map<string, ChatInfo> = new Map();
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
      `chat-info-${owner}-${repo}.json`
    );
  }

  async initialize(): Promise<void> {
    console.log(`Initializing GitHub adapter for ${this.owner}/${this.repo}`);

    // Ensure data directory exists
    await this.ensureDataDirExists();

    // Load previously processed chats
    await this.loadProcessedChats();

    // Start polling for new messages
    setInterval(async () => {
      try {
        await this.checkForNewMessages();
      } catch (error) {
        console.error("Error checking for new GitHub messages:", error);
      }
    }, 60000); // Check every 60 seconds
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

  async getMessageCount(chatId: string): Promise<number> {
    const issueNumber = parseInt(chatId.replace("github-issue-", ""), 10);
    if (isNaN(issueNumber)) {
      return 0;
    }

    try {
      const comments = await this.octokit.issues.listComments({
        owner: this.owner,
        repo: this.repo,
        issue_number: issueNumber,
      });

      // +1 for the original issue body
      return comments.data.length + 1;
    } catch (error) {
      console.error(
        `Error getting message count for issue #${issueNumber}:`,
        error
      );
      return 0;
    }
  }

  isMessageFromBot(author: string): boolean {
    return author === "stakgraph";
  }

  async updateChatInfo(chatId: string, chatInfo: ChatInfo): Promise<void> {
    this.processedChats.set(chatId, chatInfo);
    await this.saveProcessedChats();
  }

  private async ensureDataDirExists(): Promise<void> {
    if (!fs.existsSync(this.dataDir)) {
      await fs.promises.mkdir(this.dataDir, { recursive: true });
      console.log(`Created data directory: ${this.dataDir}`);
    }
  }

  private async loadProcessedChats(): Promise<void> {
    try {
      if (fs.existsSync(this.persistFilePath)) {
        const data = await fs.promises.readFile(this.persistFilePath, "utf-8");
        const chatsArray = JSON.parse(data);
        this.processedChats = new Map(chatsArray);
        console.log(
          `Loaded ${this.processedChats.size} previously processed chats`
        );
      } else {
        console.log("No previously processed chats found");
      }
    } catch (error) {
      console.error("Error loading processed chats:", error);
      // Continue with empty map if loading fails
    }
  }

  private async saveProcessedChats(): Promise<void> {
    try {
      const chatsArray = Array.from(this.processedChats.entries());
      await fs.promises.writeFile(
        this.persistFilePath,
        JSON.stringify(chatsArray),
        "utf-8"
      );
      console.log(`Saved ${this.processedChats.size} processed chats`);
    } catch (error) {
      console.error("Error saving processed chats:", error);
    }
  }

  async checkForNewMessages(): Promise<void> {
    const issues = await this.octokit.issues.listForRepo({
      owner: this.owner,
      repo: this.repo,
      state: "open",
      sort: "updated",
      direction: "desc",
      per_page: 20,
    });

    for (const issue of issues.data) {
      if (!issue.body) {
        continue;
      }

      const chatId = `github-issue-${issue.number}`;
      const currentMessageCount = await this.getMessageCount(chatId);
      const existingChatInfo = this.processedChats.get(chatId);

      // Check if this is a new issue with @stakwork mention
      if (
        !existingChatInfo &&
        (issue.body.includes("@stakwork") || issue.body.includes("@stakgraph"))
      ) {
        console.log(
          `Processing new GitHub issue #${issue.number} with @stakwork mention`
        );

        // Extract codespace URL from the issue body
        const extractedCodespaceUrl = extractCodespaceUrl(issue.body);

        const message: Message = {
          role: "user",
          content: issue.body,
          ...(extractedCodespaceUrl && { codespaceUrl: extractedCodespaceUrl }),
        };

        if (extractedCodespaceUrl) {
          console.log(`Extracted codespace URL: ${extractedCodespaceUrl}`);
        }

        // Store new chat info
        const newChatInfo: ChatInfo = {
          messageCount: currentMessageCount,
        };
        this.processedChats.set(chatId, newChatInfo);
        await this.saveProcessedChats();

        try {
          await this.messageCallback(chatId, message);
        } catch (error) {
          console.error(
            `Error processing GitHub issue #${issue.number}:`,
            error
          );
        }
      }
      // Check if there are new messages in existing chat
      else if (
        existingChatInfo &&
        currentMessageCount > existingChatInfo.messageCount
      ) {
        console.log(
          `Checking for new messages in GitHub issue #${issue.number}`
        );

        // Get the new comments
        const comments = await this.octokit.issues.listComments({
          owner: this.owner,
          repo: this.repo,
          issue_number: issue.number,
        });

        // Calculate how many comments we've already processed
        // existingChatInfo.messageCount includes the issue body (+1)
        // So comments already processed = existingChatInfo.messageCount - 1
        const commentsAlreadyProcessed = existingChatInfo.messageCount - 1;

        // Find new comments since last check
        const newComments = comments.data.slice(commentsAlreadyProcessed);

        for (const comment of newComments) {
          // Skip if comment is from bot
          if (this.isMessageFromBot(comment.user?.login || "")) {
            console.log(
              `Skipping comment from bot user: ${comment.user?.login}`
            );
            continue;
          }

          console.log(
            `Processing new comment from user: ${comment.user?.login}`
          );

          const message: Message = {
            role: "user",
            content: comment.body || "",
          };

          try {
            // Use stored webhook if available
            if (existingChatInfo.webhookToStore) {
              await this.sendToStoredWebhook(
                existingChatInfo.webhookToStore,
                chatId,
                message
              );
            } else {
              await this.messageCallback(chatId, message);
            }
          } catch (error) {
            console.error(
              `Error processing new comment in issue #${issue.number}:`,
              error
            );
          }
        }

        // Update message count
        existingChatInfo.messageCount = currentMessageCount;
        await this.saveProcessedChats();
      }
    }
  }

  private async sendToStoredWebhook(
    webhookUrl: string,
    chatId: string,
    message: Message
  ): Promise<void> {
    try {
      const response = await fetch(webhookUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          chat_id: chatId,
          messages: [message],
        }),
      });

      if (!response.ok) {
        throw new Error(`Webhook request failed: ${response.status}`);
      }

      console.log(`Sent message to stored webhook for ${chatId}`);
    } catch (error) {
      console.error(`Error sending to stored webhook:`, error);
      // Fall back to normal processing
      await this.messageCallback(chatId, message);
    }
  }
}
