import { useState } from "react";

interface Action {
  id: string;
  type: string;
  timestamp: number;
}

interface ActionsResult {
  actions: Action[];
  addForce: (type: string) => void;
  removeForce: (id: string) => void;
  clearAll: () => void;
  getCount: () => number;
}

export function useActions(): ActionsResult {
  const [actions, setActions] = useState<Action[]>([]);

  const addForce = (type: string) => {
    const newAction: Action = {
      id: `action-${Date.now()}`,
      type,
      timestamp: Date.now(),
    };
    setActions((prev) => [...prev, newAction]);
    console.log("addForce:", newAction);
  };

  const removeForce = (id: string) => {
    setActions((prev) => prev.filter((a) => a.id !== id));
    console.log("removeForce:", id);
  };

  const clearAll = () => {
    setActions([]);
    console.log("clearAll: all actions cleared");
  };

  const getCount = () => {
    const count = actions.length;
    console.log("getCount:", count);
    return count;
  };

  return {
    actions,
    addForce,
    removeForce,
    clearAll,
    getCount,
  };
}
