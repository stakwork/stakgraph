import { useState, useCallback } from "react";

export type StoreState = {
  people: Person[];
  loading: boolean;
};

export interface Person {
  id: number;
  name: string;
  email: string;
}

const initialState: StoreState = {
  people: [],
  loading: false,
};

export function useStore() {
  const [state, setState] = useState<StoreState>(initialState);

  const setPeople = useCallback((people: Person[]) => {
    setState((prev) => ({ ...prev, people }));
  }, []);

  const setLoading = useCallback((loading: boolean) => {
    setState((prev) => ({ ...prev, loading }));
  }, []);

  const addPerson = useCallback((person: Person) => {
    setState((prev) => ({
      ...prev,
      people: [...prev.people, person],
    }));
  }, []);

  return {
    state,
    setPeople,
    setLoading,
    addPerson,
  };
}
