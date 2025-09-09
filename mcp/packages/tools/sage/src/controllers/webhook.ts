import { Request, Response } from "express";
import { Adapter, ChatAdapter, ChatInfo } from "../adapters/adapter.js";
import { WebhookPayload } from "../types/index.js";
import { getAdapterFromChatId } from "../utils/chatId.js";

export class WebhookController {
  private adapters: Record<Adapter, ChatAdapter>;

  constructor(adapters: Record<Adapter, ChatAdapter>) {
    this.adapters = adapters;
  }

  async handleWebhook(req: Request, res: Response): Promise<void> {
    try {
      const payload = req.body as WebhookPayload;
      const chatId = payload.value?.chatId;

      console.log("=> handleWebhook payload", payload);

      if (!chatId) {
        res.status(400).json({
          success: false,
          message: "Missing chat_id parameter",
        });
        return;
      }

      // Determine which adapter to use based on chatId prefix
      const adapterKey = getAdapterFromChatId(chatId);
      const adapter = this.adapters[adapterKey];

      if (!adapter) {
        res.status(500).json({
          success: false,
          message: `No adapter found for chat ${chatId}`,
        });
        return;
      }

      // Send message to the appropriate platform
      await adapter.sendResponse(chatId, {
        role: "assistant",
        content: payload.value?.response,
      });

      // Extract and store webhook if present
      const webhookToStore = payload.value?.artifacts
        ?.find((artifact) => artifact.type === "action")
        ?.content.options.find(
          (option) => option.action_type === "chat"
        )?.webhook;

      if (webhookToStore) {
        console.log("=> webhookToStore", webhookToStore);

        // Get current message count and update chat info
        const currentMessageCount = await adapter.getMessageCount(chatId);
        const updatedChatInfo: ChatInfo = {
          webhookToStore,
          messageCount: currentMessageCount,
        };

        await adapter.updateChatInfo(chatId, updatedChatInfo);
        console.log(`Updated chat info for ${chatId} with webhook`);
      }

      res.status(200).json({
        success: true,
        message: "Webhook processed successfully",
        chat_id: chatId,
      });
    } catch (error) {
      console.error("Error processing webhook:", error);
      res.status(500).json({
        success: false,
        message: `Error processing webhook: ${error}`,
      });
    }
  }
}
