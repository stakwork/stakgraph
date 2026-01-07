import { useState, useEffect } from "react";

interface Bounty {
  id: string;
  title: string;
  description: string;
  amount: number;
  status: string;
  workspaceId: string;
  creatorPubkey: string;
}

interface BountyFilters {
  status?: string;
}

interface PaginationParams {
  page: number;
  limit: number;
}

interface BountySortParams {
  field: string;
  order: "asc" | "desc";
}

export const bountyKeys = {
  all: ["bounties"] as const,
  lists: () => [...bountyKeys.all, "list"] as const,
  list: (
    filters?: BountyFilters,
    pagination?: PaginationParams,
    sort?: BountySortParams
  ) => [...bountyKeys.lists(), { filters, pagination, sort }] as const,
  details: () => [...bountyKeys.all, "detail"] as const,
  detail: (id: string) => [...bountyKeys.details(), id] as const,
  workspace: (workspaceId: string) =>
    [...bountyKeys.all, "workspace", workspaceId] as const,
  assignee: (assigneePubkey: string) =>
    [...bountyKeys.all, "assignee", assigneePubkey] as const,
  creator: (creatorPubkey: string) =>
    [...bountyKeys.all, "creator", creatorPubkey] as const,
};

interface QueryResult<T> {
  data: T | null;
  isLoading: boolean;
  isSuccess: boolean;
  isError: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
}

export function useGetBounty(id: string, enabled = true): QueryResult<Bounty> {
  const [data, setData] = useState<Bounty | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const [isError, setIsError] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const queryKey = bountyKeys.detail(id);

  const fetchBounty = async () => {
    if (!enabled || !id) return;

    setIsLoading(true);
    setIsError(false);
    setError(null);

    try {
      const response = await fetch(`/api/bounties/${id}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch bounty: ${response.status}`);
      }
      const bountyData = await response.json();
      setData(bountyData);
      setIsSuccess(true);
      console.log("fetchBounty:", bountyData, "queryKey:", queryKey);
    } catch (err) {
      const errorObj = err instanceof Error ? err : new Error("Unknown error");
      setError(errorObj);
      setIsError(true);
      console.error("fetchBounty error:", errorObj.message);
    } finally {
      setIsLoading(false);
    }
  };

  const refetch = async () => {
    console.log("refetch: re-fetching bounty data");
    await fetchBounty();
  };

  useEffect(() => {
    if (enabled && id) {
      fetchBounty();
    }
  }, [id, enabled]);

  return {
    data,
    isLoading,
    isSuccess,
    isError,
    error,
    refetch,
  };
}

export function useGetBounties(
  filters?: BountyFilters,
  pagination?: PaginationParams,
  sort?: BountySortParams
): QueryResult<Bounty[]> {
  const [data, setData] = useState<Bounty[] | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const [isError, setIsError] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const queryKey = bountyKeys.list(filters, pagination, sort);

  const fetchBounties = async () => {
    setIsLoading(true);
    setIsError(false);
    setError(null);

    try {
      const params = new URLSearchParams();
      if (filters?.status) params.append("status", filters.status);
      if (pagination) {
        params.append("page", String(pagination.page));
        params.append("limit", String(pagination.limit));
      }
      if (sort) {
        params.append("sortField", sort.field);
        params.append("sortOrder", sort.order);
      }

      const response = await fetch(`/api/bounties?${params.toString()}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch bounties: ${response.status}`);
      }
      const bountiesData = await response.json();
      setData(bountiesData);
      setIsSuccess(true);
      console.log("fetchBounties:", bountiesData, "queryKey:", queryKey);
    } catch (err) {
      const errorObj = err instanceof Error ? err : new Error("Unknown error");
      setError(errorObj);
      setIsError(true);
      console.error("fetchBounties error:", errorObj.message);
    } finally {
      setIsLoading(false);
    }
  };

  const refetch = async () => {
    console.log("refetch: re-fetching bounties data");
    await fetchBounties();
  };

  useEffect(() => {
    fetchBounties();
  }, [
    JSON.stringify(filters),
    JSON.stringify(pagination),
    JSON.stringify(sort),
  ]);

  return {
    data,
    isLoading,
    isSuccess,
    isError,
    error,
    refetch,
  };
}
