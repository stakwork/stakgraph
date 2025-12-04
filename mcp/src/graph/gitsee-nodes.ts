import { v4 as uuidv4 } from "uuid";
import { NodeType, NodeData } from "./types.js";

function getTimestamp(): string {
  const seconds = Date.now() / 1000;
  const ts = seconds.toFixed(7);
  console.log(`Generated timestamp: ${ts}`);
  return ts;
}

interface GitSeeRepository {
  id: string;
  name?: string;
  html_url?: string;
  stargazers_count?: number;
  forks_count?: number;
  icon?: string;
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

export function prepareGitHubRepoNode(repo: GitSeeRepository): {
  node_type: NodeType;
  node_data: NodeData;
} {
  console.log("===> prepareGitHubRepoNode", JSON.stringify(repo, null, 2));
  const fullName = repo.name || "unknown/unknown";

  return {
    node_type: "GitHubRepo",
    node_data: {
      name: fullName,
      file: repo.name || "",
      body: repo.html_url || "",
      start: 0,
      end: 0,
      ref_id: uuidv4(),
      date_added_to_graph: getTimestamp(),
      Data_Bank: fullName,
      last_synced: new Date().toISOString(),
      sync_source: "gitsee",
      github_id: repo.id,
      stars: repo.stargazers_count || 0,
      forks: repo.forks_count || 0,
      icon: repo.icon,
    },
  };
}

export function prepareContributorNode(contributor: GitSeeContributor): {
  node_type: NodeType;
  node_data: NodeData;
} {
  return {
    node_type: "Contributor",
    node_data: {
      name: contributor.login,
      file: contributor.html_url || "",
      body: `${contributor.contributions} contributions`,
      start: 0,
      end: 0,
      ref_id: uuidv4(),
      date_added_to_graph: getTimestamp(),
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
  return {
    node_type: "Stars",
    node_data: {
      name: `${repoFullName}-stars`,
      file: `stars:${repoFullName}`,
      body: `Star count: ${stars}`,
      start: 0,
      end: 0,
      ref_id: uuidv4(),
      date_added_to_graph: getTimestamp(),
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
  return {
    node_type: "Commits",
    node_data: {
      name: `${repoFullName}-commits`,
      file: `commits:${repoFullName}`,
      body: `Total commits: ${totalCommits}`,
      start: 0,
      end: 0,
      ref_id: uuidv4(),
      date_added_to_graph: getTimestamp(),
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
  return {
    node_type: "Age",
    node_data: {
      name: `${repoFullName}-age`,
      file: `age:${repoFullName}`,
      body: `Repository age: ${ageInYears} years`,
      start: 0,
      end: 0,
      ref_id: uuidv4(),
      date_added_to_graph: getTimestamp(),
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
  return {
    node_type: "Issues",
    node_data: {
      name: `${repoFullName}-issues`,
      file: `issues:${repoFullName}`,
      body: `Total issues: ${totalIssues}`,
      start: 0,
      end: 0,
      ref_id: uuidv4(),
      date_added_to_graph: getTimestamp(),
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
