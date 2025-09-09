export interface Message {
  role: string;
  content: string;
  codespaceUrl?: string; // Add optional codespace URL
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
  value: {
    chatId: string;
    messageId: string;
    response: string;
    sourceWebsocketId: string;
    artifacts?: ChatMessageArtifact[];
  };
}

export interface ChatMessageArtifact {
  id: string;
  type: ArtifactType;
  content: ActionArtifactContent;
}

// render "WebhookPayload.response" and "ActionArtifactContent.actionText" and send to "ActionArtifactContentOption.webhook"
export type ArtifactType = "action" | string;

export interface ActionArtifactContent {
  actionText: string;
  options: ActionArtifactContentOption[];
}

export interface ActionArtifactContentOption {
  action_type: "chat" | string;
  option_label: string;
  option_response: "textbox" | string;
  webhook: string;
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
