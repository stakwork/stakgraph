export interface Message {
  role: string;
  content: string;
}

export interface ChatRequest {
  chat_id?: string;
  messages: Message[];
}

export interface ChatResponse {
  success: boolean;
  message: string;
  chat_id: string;
}

export interface WebhookPayload {
  chat_id: string;
  message: Message;
  workflow_id: number;
  project_id: number;
}

export interface StakworkChatPayload {
  name: string;
  workflow_id: number;
  workflow_params: {
    set_var: {
      attributes: {
        vars: {
          chatId: string;
          messages: Message[];
          [key: string]: any;
        };
      };
    };
  };
  webhook_url: string;
}

export interface StakworkResponse {
  success: boolean;
  data: {
    project_id: number;
  };
}
