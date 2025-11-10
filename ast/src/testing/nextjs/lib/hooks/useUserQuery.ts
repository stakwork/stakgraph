import { useState, useEffect } from "react";

interface User {
  id: string;
  name: string;
  email: string;
}

interface QueryResult {
  data: User | null;
  isLoading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
  invalidate: () => void;
  reset: () => void;
}

export function useUserQuery(userId: string): QueryResult {
  const [data, setData] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchUser = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`/api/users/${userId}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch user: ${response.status}`);
      }
      const userData = await response.json();
      setData(userData);
      console.log("fetchUser:", userData);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Unknown error";
      setError(errorMessage);
      console.error("fetchUser error:", errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const refetch = async () => {
    console.log("refetch: re-fetching user data");
    await fetchUser();
  };

  const invalidate = () => {
    setData(null);
    console.log("invalidate: cache invalidated");
  };

  const reset = () => {
    setData(null);
    setError(null);
    setIsLoading(false);
    console.log("reset: query state reset");
  };

  useEffect(() => {
    if (userId) {
      fetchUser();
    }
  }, [userId]);

  return {
    data,
    isLoading,
    error,
    refetch,
    invalidate,
    reset,
  };
}
