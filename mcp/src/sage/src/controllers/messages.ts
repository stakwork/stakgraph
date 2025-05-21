import { Request, Response } from "express";
import { StakworkService } from "../services/stakwork.js";
import { ChatRequest, Message } from "../types/index.js";
import { getAdapterFromChatId } from "../utils/chatId.js";

export class MessagesController {
  private stakworkService: StakworkService;
  private webhookBaseUrl: string;

  constructor(stakworkService: StakworkService, webhookBaseUrl: string) {
    this.stakworkService = stakworkService;
    this.webhookBaseUrl = webhookBaseUrl;
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
    await this.stakworkService.sendToStakwork(payload);
    return chatId;
  }
}
