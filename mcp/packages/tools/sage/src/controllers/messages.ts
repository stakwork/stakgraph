import { Request, Response } from "express";
import { StakworkService } from "../services/stakwork.js";
import { ChatRequest, Message } from "../types/index.js";
import { getAdapterFromChatId } from "../utils/chatId.js";
import { ChatAdapter, Adapter } from "../adapters/adapter.js";

export class MessagesController {
  private stakworkService: StakworkService;
  private webhookBaseUrl: string;
  private adapters: Record<string, ChatAdapter>; // Add adapters reference

  constructor(
    stakworkService: StakworkService,
    webhookBaseUrl: string,
    adapters: Record<Adapter, ChatAdapter> // Add adapters parameter
  ) {
    this.stakworkService = stakworkService;
    this.webhookBaseUrl = webhookBaseUrl;
    this.adapters = adapters;
  }

  async handleMessage(req: Request, res: Response): Promise<void> {
    try {
      const { chat_id, messages } = req.body as ChatRequest;
      if (!chat_id) {
        res.status(400).json({
          success: false,
          message: `Error processing message: Chat ID is required`,
        });
        return;
      }
      const provider = getAdapterFromChatId(chat_id);
      if (provider === "none") {
        res.status(400).json({
          success: false,
          message: `Error processing message: Invalid chat ID`,
        });
        return;
      }

      if (!messages || !Array.isArray(messages) || messages.length === 0) {
        res.status(400).json({
          success: false,
          message: "Invalid or missing messages array",
          chat_id: chat_id,
        });
        return;
      }

      const finalChatId = await this.processMessages(chat_id, messages);

      res.status(200).json({
        success: true,
        message: "Messages sent to Stakwork",
        chat_id: finalChatId,
      });
    } catch (error) {
      console.error("Error processing message:", error);
      res.status(500).json({
        success: false,
        message: `Error processing message: ${error}`,
        chat_id: req.body.chat_id || "",
      });
    }
  }

  async processMessages(chatId: string, messages: Message[]): Promise<string> {
    const payload = this.stakworkService.buildStakworkPayload(
      chatId,
      messages,
      `${this.webhookBaseUrl}?chat_id=${chatId}`
    );

    // Send to Stakwork and get project_id
    const projectId = await this.stakworkService.sendToStakwork(payload);

    // Send immediate response if we have a project_id and appropriate adapter
    if (projectId) {
      await this.sendImmediateResponse(chatId, projectId);
    }

    return chatId;
  }

  private async sendImmediateResponse(
    chatId: string,
    projectId: number
  ): Promise<void> {
    try {
      const adapterType = getAdapterFromChatId(chatId);
      const adapter = this.adapters[adapterType];

      if (adapter) {
        const stakworkUrl = `https://jobs.stakwork.com/admin/projects/${projectId}`;
        const immediateMessage: Message = {
          role: "assistant",
          content: `🤖 Processing your request... [Stakwork run](${stakworkUrl})\n\nI've received your message and started working on it. You'll receive a detailed response shortly.`,
        };

        await adapter.sendResponse(chatId, immediateMessage);
        console.log(
          `Sent immediate response for project ${projectId} to ${chatId}`
        );
      }
    } catch (error) {
      console.error("Error sending immediate response:", error);
      // Don't throw here - we don't want to fail the main flow if immediate response fails
    }
  }
}
