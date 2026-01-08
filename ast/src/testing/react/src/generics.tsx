import React, { useState, useCallback } from "react";

// Generic React Hook
export function useGenericState<T>(initialValue: T): [T, (value: T) => void] {
  const [state, setState] = useState<T>(initialValue);
  return [state, setState];
}

// Generic Component Props
interface ListProps<T> {
  items: T[];
  renderItem: (item: T, index: number) => React.ReactNode;
  keyExtractor: (item: T) => string;
}

// Generic Component
export function GenericList<T>({
  items,
  renderItem,
  keyExtractor,
}: ListProps<T>) {
  return (
    <ul>
      {items.map((item, index) => (
        <li key={keyExtractor(item)}>{renderItem(item, index)}</li>
      ))}
    </ul>
  );
}

// Generic Higher-Order Component
export function withLoading<T extends object>(
  WrappedComponent: React.ComponentType<T>
): React.FC<T & { isLoading: boolean }> {
  return function WithLoadingComponent({ isLoading, ...props }) {
    if (isLoading) return <div>Loading...</div>;
    return <WrappedComponent {...(props as T)} />;
  };
}

// Generic data fetching hook
export function useFetch<T>(url: string): {
  data: T | null;
  loading: boolean;
  error: Error | null;
} {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useCallback(async () => {
    try {
      const response = await fetch(url);
      const result = await response.json();
      setData(result);
    } catch (e) {
      setError(e as Error);
    } finally {
      setLoading(false);
    }
  }, [url]);

  return { data, loading, error };
}

// Generic utility types for React
export type PropsWithChildren<P = unknown> = P & { children?: React.ReactNode };
export type Dispatch<A> = (action: A) => void;
