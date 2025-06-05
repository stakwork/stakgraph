import axios from "axios";
import {
  Message,
  StakworkChatPayload,
  StakworkResponse,
} from "../types/index.js";
import { getAdapterFromChatId } from "../utils/chatId.js";

export class StakworkService {
  private apiKey: string;
  private baseUrl: string = "https://api.stakwork.com/api/v1/projects";
  private workflowId: number;
  private codeSpaceURL: string;
  private twoBBaseUrl: string;
  private secret: string;
  private dryRun: boolean;
  private repo: string;

  constructor(
    apiKey: string,
    workflowId: number = 38842,
    codeSpaceURL: string = "",
    twoBBaseUrl: string = "",
    secret: string = "",
    dryRun: boolean = false,
    repo: string = ""
  ) {
    this.apiKey = apiKey;
    this.workflowId = workflowId;
    this.codeSpaceURL = codeSpaceURL;
    this.twoBBaseUrl = twoBBaseUrl;
    this.secret = secret;
    this.dryRun = dryRun;
    this.repo = repo;
  }

  // src/sage/src/services/stakwork.ts
  async sendToStakwork(payload: StakworkChatPayload): Promise<number | null> {
    if (this.dryRun) {
      console.log("Dry run, not sending to Stakwork");
      return null;
    }
    console.log(JSON.stringify(payload, null, 2));
    try {
      console.log("Sending payload to Stakwork");

      const response = await axios.post<StakworkResponse>(
        this.baseUrl,
        payload,
        {
          headers: {
            Authorization: `Token token=${this.apiKey}`,
            "Content-Type": "application/json",
            "X-Secret": this.secret,
          },
        }
      );

      if (response.status !== 200) {
        throw new Error(`Stakwork API error: ${JSON.stringify(response.data)}`);
      }

      console.log("Stakwork response received", response.data);
      return response.data.data.project_id; // Return the project_id
    } catch (error) {
      console.error("Error sending request to Stakwork:", error);
      throw new Error(`Error sending request to Stakwork: ${error}`);
    }
  }

  buildStakworkPayload(
    chatId: string,
    messages: Message[],
    webhookUrl: string
  ): StakworkChatPayload {
    const source = getAdapterFromChatId(chatId);

    // Check if any message has a codespace URL
    const messageWithCodespace = messages.find((msg) => msg.codespaceUrl);
    const codeSpaceURL =
      messageWithCodespace?.codespaceUrl || this.codeSpaceURL;

    console.log(
      `Using codespace URL: ${codeSpaceURL} ${
        messageWithCodespace?.codespaceUrl ? "(from message)" : "(from config)"
      }`
    );

    const history: { role: string; content: string }[] = [];

    return {
      name: "Hive Chat Processor",
      workflow_id: this.workflowId,
      workflow_params: {
        set_var: {
          attributes: {
            vars: {
              chatId,
              messages,
              query: messages[messages.length - 1].content,
              codeSpaceURL, // Use the extracted or default codespace URL
              "2b_base_url": this.twoBBaseUrl,
              secret: this.secret,
              source,
              history,
              webhook_url: webhookUrl,
              repo: this.repo,
            },
          },
        },
      },
      webhook_url: webhookUrl,
    };
  }
}
