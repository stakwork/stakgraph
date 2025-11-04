import { v4 as uuidv4 } from "uuid";
import { NodeType, NodeData } from "./types.js";

interface GitSeeRepository {
  id: number;
  name?: string;
  full_name?: string;
  owner?: {
    login: string;
    id: number;
    avatar_url: string;
  };
  description?: string;
  stargazers_count?: number;
  forks_count?: number;
  language?: string;
  created_at?: string;
  updated_at?: string;
  clone_url?: string;
  html_url?: string;
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
  const fullName = repo.full_name || "unknown/unknown";
  const ownerName = repo.owner?.login || fullName.split("/")[0] || "unknown";

  return {
    node_type: "GitHubRepo",
    node_data: {
      name: fullName,
      file: repo.html_url || "",
      body: repo.description || "",
      start: 0,
      end: 0,
      ref_id: uuidv4(),
      date_added_to_graph: now.toString(),
      Data_Bank: fullName,
      last_synced: new Date().toISOString(),
      sync_source: "gitsee",
      github_id: repo.id,
      owner: ownerName,
      stars: repo.stargazers_count || 0,
      forks: repo.forks_count || 0,
      language: repo.language || "Unknown",
      created_at: repo.created_at || "",
      updated_at: repo.updated_at || "",
      clone_url: repo.clone_url || "",
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
      Data_Bank: contributor.login,
      last_synced: new Date().toISOString(),
      sync_source: "gitsee",
      github_id: contributor.id,
      avatar_url: contributor.avatar_url,
      contributions: contributor.contributions,
      user_type: contributor.type || "User",
    },
  };
}

export function prepareStarsNode(
  stars: number,
  repoFullName: string
): {
  node_type: NodeType;
  node_data: NodeData;
} {
  const now = Date.now();
  return {
    node_type: "Stars",
    node_data: {
      name: `${repoFullName}-stars`,
      file: `stars:${repoFullName}`,
      body: `Star count: ${stars}`,
      start: 0,
      end: 0,
      ref_id: uuidv4(),
      date_added_to_graph: now.toString(),
      Data_Bank: `${repoFullName}-stars`,
      last_synced: new Date().toISOString(),
      sync_source: "gitsee",
      stars: stars,
    },
  };
}

export function prepareCommitsNode(
  totalCommits: number,
  repoFullName: string
): {
  node_type: NodeType;
  node_data: NodeData;
} {
  const now = Date.now();
  return {
    node_type: "Commits",
    node_data: {
      name: `${repoFullName}-commits`,
      file: `commits:${repoFullName}`,
      body: `Total commits: ${totalCommits}`,
      start: 0,
      end: 0,
      ref_id: uuidv4(),
      date_added_to_graph: now.toString(),
      Data_Bank: `${repoFullName}-commits`,
      last_synced: new Date().toISOString(),
      sync_source: "gitsee",
      total_commits: totalCommits,
    },
  };
}

export function prepareAgeNode(
  ageInYears: number,
  repoFullName: string
): {
  node_type: NodeType;
  node_data: NodeData;
} {
  const now = Date.now();
  return {
    node_type: "Age",
    node_data: {
      name: `${repoFullName}-age`,
      file: `age:${repoFullName}`,
      body: `Repository age: ${ageInYears} years`,
      start: 0,
      end: 0,
      ref_id: uuidv4(),
      date_added_to_graph: now.toString(),
      Data_Bank: `${repoFullName}-age`,
      last_synced: new Date().toISOString(),
      sync_source: "gitsee",
      age_in_years: ageInYears,
    },
  };
}

export function prepareIssuesNode(
  totalIssues: number,
  repoFullName: string
): {
  node_type: NodeType;
  node_data: NodeData;
} {
  const now = Date.now();
  return {
    node_type: "Issues",
    node_data: {
      name: `${repoFullName}-issues`,
      file: `issues:${repoFullName}`,
      body: `Total issues: ${totalIssues}`,
      start: 0,
      end: 0,
      ref_id: uuidv4(),
      date_added_to_graph: now.toString(),
      Data_Bank: `${repoFullName}-issues`,
      last_synced: new Date().toISOString(),
      sync_source: "gitsee",
      total_issues: totalIssues,
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

export function prepareHasStarsEdge(repoRefId: string, starsRefId: string) {
  return {
    edge_type: "HAS_STARS",
    source: repoRefId,
    target: starsRefId,
  };
}

export function prepareHasCommitsEdge(repoRefId: string, commitsRefId: string) {
  return {
    edge_type: "HAS_COMMITS",
    source: repoRefId,
    target: commitsRefId,
  };
}

export function prepareHasAgeEdge(repoRefId: string, ageRefId: string) {
  return {
    edge_type: "HAS_AGE",
    source: repoRefId,
    target: ageRefId,
  };
}

export function prepareHasIssuesEdge(repoRefId: string, issuesRefId: string) {
  return {
    edge_type: "HAS_ISSUES",
    source: repoRefId,
    target: issuesRefId,
  };
}
