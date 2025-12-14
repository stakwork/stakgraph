import { useState } from "react";

interface MutationResult<T> {
  mutate: (data: T) => Promise<void>;
  reset: () => void;
  isLoading: boolean;
  error: string | null;
  data: any | null;
}

export function useMutation<T>(endpoint: string): MutationResult<T> {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<any | null>(null);

  const mutate = async (payload: T) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setData(result);
      console.log("Mutation successful:", result);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Unknown error";
      setError(errorMessage);
      console.error("Mutation failed:", errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const reset = () => {
    setIsLoading(false);
    setError(null);
    setData(null);
    console.log("Mutation state reset");
  };

  return {
    mutate,
    reset,
    isLoading,
    error,
    data,
  };
}
