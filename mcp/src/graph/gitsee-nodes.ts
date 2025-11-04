import { v4 as uuidv4 } from "uuid";
import { NodeType, NodeData } from "./types.js";

interface GitSeeRepository {
  id: number;
  name: string;
  full_name: string;
  owner: {
    login: string;
    id: number;
    avatar_url: string;
  };
  description?: string;
  stargazers_count: number;
  forks_count: number;
  language?: string;
  created_at: string;
  updated_at: string;
  clone_url: string;
  html_url: string;
}

interface GitSeeContributor {
  id: number;
  login: string;
  avatar_url: string;
  contributions: number;
  url?: string;
  html_url?: string;
  type?: string;
}

interface GitSeeRepoStats {
  stars: number;
  totalIssues: number;
  totalCommits: number;
  ageInYears: number;
}

export function prepareGitHubRepoNode(repo: GitSeeRepository): {
  node_type: NodeType;
  node_data: NodeData;
} {
  const now = Date.now();
  return {
    node_type: "GitHubRepo",
    node_data: {
      name: repo.full_name,
      file: repo.html_url,
      body: repo.description || "",
      start: 0,
      end: 0,
      ref_id: uuidv4(),
      date_added_to_graph: now.toString(),
      github_id: repo.id,
      owner: repo.owner.login,
      stars: repo.stargazers_count,
      forks: repo.forks_count,
      language: repo.language || "Unknown",
      created_at: repo.created_at,
      updated_at: repo.updated_at,
      clone_url: repo.clone_url,
    },
  };
}

export function prepareContributorNode(contributor: GitSeeContributor): {
  node_type: NodeType;
  node_data: NodeData;
} {
  const now = Date.now();
  return {
    node_type: "Contributor",
    node_data: {
      name: contributor.login,
      file: contributor.html_url || "",
      body: `${contributor.contributions} contributions`,
      start: 0,
      end: 0,
      ref_id: uuidv4(),
      date_added_to_graph: now.toString(),
      github_id: contributor.id,
      avatar_url: contributor.avatar_url,
      contributions: contributor.contributions,
      user_type: contributor.type || "User",
    },
  };
}

export function prepareRepoStatsNode(
  stats: GitSeeRepoStats,
  repoFullName: string
): {
  node_type: NodeType;
  node_data: NodeData;
} {
  const now = Date.now();
  return {
    node_type: "RepoStats",
    node_data: {
      name: `${repoFullName}-stats`,
      file: `stats:${repoFullName}`,
      body: `Stars: ${stats.stars}, Issues: ${stats.totalIssues}, Commits: ${stats.totalCommits}`,
      start: 0,
      end: 0,
      ref_id: uuidv4(),
      date_added_to_graph: now.toString(),
      stars: stats.stars,
      total_issues: stats.totalIssues,
      total_commits: stats.totalCommits,
      age_in_years: stats.ageInYears,
      snapshot_date: now,
    },
  };
}

export function prepareHasContributorEdge(
  repoRefId: string,
  contributorRefId: string
) {
  return {
    edge_type: "HAS_CONTRIBUTOR",
    source: repoRefId,
    target: contributorRefId,
  };
}

export function prepareHasStatsEdge(repoRefId: string, statsRefId: string) {
  return {
    edge_type: "HAS_STATS",
    source: repoRefId,
    target: statsRefId,
  };
}
