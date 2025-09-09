import { Message } from "../types/index.js";

export type Adapter = "github" | "none";

export interface ChatInfo {
  webhookToStore?: string;
  messageCount: number;
}

export interface ChatAdapter {
  initialize(): Promise<void>;
  sendResponse(chatId: string, message: Message): Promise<void>;
  onMessageReceived(
    callback: (chatId: string, message: Message) => Promise<void>
  ): void;
  getMessageCount(chatId: string): Promise<number>;
  isMessageFromBot(author: string): boolean;
  updateChatInfo(chatId: string, chatInfo: ChatInfo): Promise<void>;
}

export abstract class BaseAdapter implements ChatAdapter {
  // Initialize with a no-op async function
  protected messageCallback: (
    chatId: string,
    message: Message
  ) => Promise<void> = async () => {
    /* default no-op implementation */
  };

  abstract initialize(): Promise<void>;
  abstract sendResponse(chatId: string, message: Message): Promise<void>;
  abstract getMessageCount(chatId: string): Promise<number>;
  abstract isMessageFromBot(author: string): boolean;
  abstract updateChatInfo(chatId: string, chatInfo: ChatInfo): Promise<void>;

  onMessageReceived(
    callback: (chatId: string, message: Message) => Promise<void>
  ): void {
    this.messageCallback = callback;
  }
}

export class NoAdapter extends BaseAdapter {
  async initialize(): Promise<void> {}
  async sendResponse(_: string, __: Message): Promise<void> {}
  async getMessageCount(_: string): Promise<number> {
    return 0;
  }
  isMessageFromBot(_: string): boolean {
    return false;
  }
  async updateChatInfo(_: string, __: ChatInfo): Promise<void> {}
}

export function EmptyAdapters(): Record<Adapter, ChatAdapter> {
  return {
    github: new NoAdapter(),
    none: new NoAdapter(),
  };
}
