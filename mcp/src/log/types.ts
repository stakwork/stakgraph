export interface AgentLogSummary {
  agent: string;
  url: string;
}

export interface StakworkRunSummary {
  projectId: number;
  type: string;
  status: string;
  feature?: string | null;
  createdAt: string;
  agentLogs?: AgentLogSummary[];
}
