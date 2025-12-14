// @ts-nocheck
import { useActions } from "../../lib/hooks/useActions";

describe("unit: actions hook - object pattern", () => {
  it("adds force via object access", () => {
    const actions = useActions();

    actions.addForce("cluster");

    expect(actions.actions.length).toBeGreaterThan(0);
    const lastAction = actions.actions[actions.actions.length - 1];
    expect(lastAction.type).toBe("cluster");
    console.log("actions.addForce called via object");
  });

  it("removes force via object access", () => {
    const actions = useActions();

    actions.addForce("gravity");
    const forceId = actions.actions[0].id;

    actions.removeForce(forceId);

    const found = actions.actions.find((a) => a.id === forceId);
    expect(found).toBeUndefined();
    console.log("actions.removeForce called via object");
  });

  it("clears all via object access", () => {
    const actions = useActions();

    actions.addForce("cluster");
    actions.addForce("gravity");
    expect(actions.actions.length).toBe(2);

    actions.clearAll();

    expect(actions.actions.length).toBe(0);
    console.log("actions.clearAll called via object");
  });

  it("gets count via object access", () => {
    const actions = useActions();

    actions.addForce("cluster");
    actions.addForce("gravity");

    const count = actions.getCount();

    expect(count).toBe(2);
    console.log("actions.getCount called via object:", count);
  });
});

describe("unit: actions hook - destructured pattern", () => {
  it("adds force via destructured function", () => {
    const { addForce, actions } = useActions();

    addForce("cluster");

    expect(actions.length).toBeGreaterThan(0);
    console.log("addForce called via destructuring");
  });

  it("removes force via destructured function", () => {
    const { addForce, removeForce, actions } = useActions();

    addForce("gravity");
    const forceId = actions[0].id;

    removeForce(forceId);

    const found = actions.find((a) => a.id === forceId);
    expect(found).toBeUndefined();
    console.log("removeForce called via destructuring");
  });

  it("clears all via destructured function", () => {
    const { addForce, clearAll, actions } = useActions();

    addForce("cluster");
    addForce("gravity");

    clearAll();

    expect(actions.length).toBe(0);
    console.log("clearAll called via destructuring");
  });

  it("gets count via destructured function", () => {
    const { addForce, getCount } = useActions();

    addForce("cluster");
    addForce("gravity");

    const count = getCount();

    expect(count).toBe(2);
    console.log("getCount called via destructuring:", count);
  });
});

describe("unit: actions hook - pattern comparison", () => {
  it("compares object vs destructured patterns", () => {
    const actionsObj = useActions();
    const { addForce, removeForce, clearAll } = useActions();

    actionsObj.addForce("cluster");
    expect(actionsObj.actions.length).toBe(1);

    addForce("gravity");

    actionsObj.clearAll();

    clearAll();

    console.log("Both patterns tested for call resolution");
  });

  it("chains multiple calls via object pattern", () => {
    const actions = useActions();

    actions.addForce("cluster");
    actions.addForce("gravity");
    actions.addForce("collision");

    expect(actions.getCount()).toBe(3);

    const firstId = actions.actions[0].id;
    actions.removeForce(firstId);

    expect(actions.getCount()).toBe(2);

    actions.clearAll();

    expect(actions.getCount()).toBe(0);

    console.log("Chained object pattern calls successful");
  });

  it("chains multiple calls via destructured pattern", () => {
    const { addForce, removeForce, clearAll, getCount, actions } =
      useActions();

    addForce("cluster");
    addForce("gravity");
    addForce("collision");

    expect(getCount()).toBe(3);

    const firstId = actions[0].id;
    removeForce(firstId);

    expect(getCount()).toBe(2);

    clearAll();

    expect(getCount()).toBe(0);

    console.log("Chained destructured pattern calls successful");
  });
});
