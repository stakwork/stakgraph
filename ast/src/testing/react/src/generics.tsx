import React, { useState, useCallback } from "react";

// Generic React Hook
// @ast node: Function "useGenericState"
export function useGenericState<T>(initialValue: T): [T, (value: T) => void] {
  const [state, setState] = useState<T>(initialValue);
  return [state, setState];
}

// Generic Component Props
// @ast node: DataModel "ListProps"
interface ListProps<T> {
  items: T[];
  renderItem: (item: T, index: number) => React.ReactNode;
  keyExtractor: (item: T) => string;
}

// Generic Component
// @ast node: Function "GenericList"
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
// @ast node: Function "withLoading"
export function withLoading<T extends object>(
  WrappedComponent: React.ComponentType<T>
): React.FC<T & { isLoading: boolean }> {
  return function WithLoadingComponent({ isLoading, ...props }) {
    if (isLoading) return <div>Loading...</div>;
    return <WrappedComponent {...(props as T)} />;
  };
}

// Generic data fetching hook
// @ast node: Function "useFetch"
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
// @ast node: DataModel "PropsWithChildren"
export type PropsWithChildren<P = unknown> = P & { children?: React.ReactNode };
// @ast node: DataModel "Dispatch"
export type Dispatch<A> = (action: A) => void;
