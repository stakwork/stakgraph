import { ModelMessage } from "ai";
import { v4 as uuidv4 } from "uuid";

export interface Conversation {
  id: string;
  summary: string;
  timestamp: number;
}

export interface ConversationData {
  id: string;
  summary: string;
  messages: ModelMessage[];
  lastUpdated: number;
}

export const MAX_CONVERSATIONS = 100;

export abstract class ConversationStorage {
  // Abstract methods to be implemented by platform-specific storage
  abstract currentConversationId(): Promise<string>;
  abstract selectConversation(conversationId: string): Promise<void>;
  abstract listConversations(): Promise<Conversation[]>;
  abstract getConversation(
    conversationId: string
  ): Promise<ConversationData | null>;
  abstract storeConversationData(
    conversationId: string,
    data: ConversationData
  ): Promise<void>;
  abstract deleteConversationData(conversationId: string): Promise<boolean>;

  // Shared functionality that works across platforms
  async createConversation(
    messages: ModelMessage[]
  ): Promise<ConversationData> {
    const id = uuidv4();
    const initialMessage = messages.find((msg) => msg.role === "user");
    const summary = initialMessage
      ? this.generateSummary(initialMessage.content as string)
      : `New conversation ${new Date().toLocaleString()}`;
    const conversation: ConversationData = {
      id,
      summary,
      messages,
      lastUpdated: Date.now(),
    };
    await this.storeConversationData(id, conversation);
    await this.selectConversation(id);
    await this.pruneOldConversations();
    return conversation;
  }

  async addMessageToConversation(
    conversationId: string,
    message: ModelMessage
  ): Promise<ConversationData> {
    const conversation = await this.getConversation(conversationId);
    if (!conversation) {
      throw new Error(`Conversation with ID ${conversationId} not found`);
    }
    conversation.messages.push({ ...message });
    conversation.lastUpdated = Date.now();
    await this.storeConversationData(conversationId, conversation);
    return conversation;
  }

  async getCurrentConversation(): Promise<ConversationData | null> {
    const currentId = await this.currentConversationId();
    if (!currentId) return null;
    return await this.getConversation(currentId);
  }

  async getLatestConversation(): Promise<ConversationData | null> {
    const convos = await this.listConversations();
    if (!convos.length) return null;
    // Sort by most recent first
    const sortedConvos = [...convos].sort((a, b) => b.timestamp - a.timestamp);
    return await this.getConversation(sortedConvos[0].id);
  }

  async updateConversationSummary(
    conversationId: string,
    summary: string
  ): Promise<ConversationData> {
    const conversation = await this.getConversation(conversationId);
    if (!conversation) {
      throw new Error(`Conversation with ID ${conversationId} not found`);
    }
    conversation.summary = summary;
    conversation.lastUpdated = Date.now();
    await this.storeConversationData(conversationId, conversation);
    return conversation;
  }

  private async pruneOldConversations(): Promise<void> {
    const conversations = await this.listConversations();
    if (conversations.length <= MAX_CONVERSATIONS) return;
    // Sort by timestamp (oldest first)
    const sortedConversations = [...conversations].sort(
      (a, b) => a.timestamp - b.timestamp
    );
    // Calculate how many need to be deleted
    const deleteCount = sortedConversations.length - MAX_CONVERSATIONS;
    // Get the conversations to delete (the oldest ones)
    const conversationsToDelete = sortedConversations.slice(0, deleteCount);
    // Delete each conversation
    for (const convo of conversationsToDelete) {
      await this.deleteConversationData(convo.id);
    }
  }

  protected generateSummary(content: string): string {
    let summary = content.split("\n")[0].trim();
    if (summary.length > 50) {
      summary = summary.substring(0, 47) + "...";
    } else if (summary.length === 0 && content.length > 0) {
      summary = content.substring(0, Math.min(50, content.length));
      if (summary.length === 50) summary = summary + "...";
    } else if (summary.length === 0) {
      summary = `Conversation created on ${new Date().toLocaleString()}`;
    }
    return summary;
  }
}
