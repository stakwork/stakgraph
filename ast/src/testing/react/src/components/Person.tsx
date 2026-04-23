import { useState, useCallback } from "react";

// @ast node: DataModel "StoreState"
export type StoreState = {
  people: Person[];
  loading: boolean;
};

// @ast node: DataModel "Person"
export interface Person {
  id: number;
  name: string;
  email: string;
}

// @ast node: Var "initialState"
const initialState: StoreState = {
  people: [],
  loading: false,
};

// @ast absent: Function "setPeople" "Person.tsx"
// @ast absent: Function "setLoading" "Person.tsx"
// @ast absent: Function "addPerson" "Person.tsx"
// @ast node: Function "useStore"
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
