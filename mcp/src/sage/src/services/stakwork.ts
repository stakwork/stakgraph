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

  constructor(
    apiKey: string,
    workflowId: number = 38842,
    codeSpaceURL: string = "",
    twoBBaseUrl: string = "",
    secret: string = "",
    dryRun: boolean = false
  ) {
    this.apiKey = apiKey;
    this.workflowId = workflowId;
    this.codeSpaceURL = codeSpaceURL;
    this.twoBBaseUrl = twoBBaseUrl;
    this.secret = secret;
    this.dryRun = dryRun;
  }

  async sendToStakwork(payload: StakworkChatPayload): Promise<number> {
    if (this.dryRun) {
      console.log("Dry run, not sending to Stakwork");
      console.log(JSON.stringify(payload, null, 2));
      return 0;
    }

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

      console.log("Stakwork response received");
      return response.data.data.project_id;
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
    /*
    history: [
      {
        role: "user",
        content: "message",
      },
    ]
    */
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
              codeSpaceURL: this.codeSpaceURL,
              "2b_base_url": this.twoBBaseUrl,
              secret: this.secret,
              source,
              history,
            },
          },
        },
      },
      webhook_url: webhookUrl,
    };
  }
}
