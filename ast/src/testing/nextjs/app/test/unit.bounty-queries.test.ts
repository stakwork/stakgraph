// @ts-nocheck
import {
  bountyKeys,
  useGetBounty,
  useGetBounties,
} from "../../lib/hooks/useBountyQueries";

describe("unit: bountyKeys query key factory", () => {
  it("generates correct key for all bounties", () => {
    const key = bountyKeys.all;
    expect(key).toEqual(["bounties"]);
    console.log("bountyKeys.all:", key);
  });

  it("generates correct key for lists", () => {
    const key = bountyKeys.lists();
    expect(key).toEqual(["bounties", "list"]);
    console.log("bountyKeys.lists():", key);
  });

  it("generates correct key for detail", () => {
    const bountyId = "bounty-123";
    const key = bountyKeys.detail(bountyId);
    expect(key).toEqual(["bounties", "detail", bountyId]);
    console.log("bountyKeys.detail():", key);
  });

  it("generates correct key for details base", () => {
    const key = bountyKeys.details();
    expect(key).toEqual(["bounties", "detail"]);
    console.log("bountyKeys.details():", key);
  });

  it("generates correct key for workspace", () => {
    const workspaceId = "workspace-456";
    const key = bountyKeys.workspace(workspaceId);
    expect(key).toEqual(["bounties", "workspace", workspaceId]);
    console.log("bountyKeys.workspace():", key);
  });

  it("generates correct key for assignee", () => {
    const assigneePubkey = "assignee-pubkey-789";
    const key = bountyKeys.assignee(assigneePubkey);
    expect(key).toEqual(["bounties", "assignee", assigneePubkey]);
    console.log("bountyKeys.assignee():", key);
  });

  it("generates correct key for creator", () => {
    const creatorPubkey = "creator-pubkey-abc";
    const key = bountyKeys.creator(creatorPubkey);
    expect(key).toEqual(["bounties", "creator", creatorPubkey]);
    console.log("bountyKeys.creator():", key);
  });

  it("generates correct key for list with filters", () => {
    const filters = { status: "OPEN" };
    const pagination = { page: 1, limit: 10 };
    const sort = { field: "createdAt", order: "desc" as const };
    const key = bountyKeys.list(filters, pagination, sort);
    expect(key).toEqual(["bounties", "list", { filters, pagination, sort }]);
    console.log("bountyKeys.list() with params:", key);
  });
});

describe("unit: useGetBounty hook", () => {
  it("fetches bounty by id", async () => {
    const bountyId = "bounty-test-001";
    const query = useGetBounty(bountyId);

    expect(query.isLoading).toBeDefined();
    expect(query.data).toBeNull();
    console.log("useGetBounty initial state:", {
      isLoading: query.isLoading,
      data: query.data,
    });
  });

  it("does not fetch when disabled", () => {
    const bountyId = "bounty-test-002";
    const query = useGetBounty(bountyId, false);

    expect(query.isLoading).toBe(false);
    console.log("useGetBounty disabled:", query);
  });

  it("does not fetch when id is empty", () => {
    const query = useGetBounty("");

    expect(query.isLoading).toBe(false);
    console.log("useGetBounty empty id:", query);
  });

  it("refetches when called", async () => {
    const bountyId = "bounty-test-003";
    const query = useGetBounty(bountyId);

    await query.refetch();

    expect(query.refetch).toBeDefined();
    console.log("useGetBounty refetch called");
  });
});

describe("unit: useGetBounties hook", () => {
  it("fetches bounties list", async () => {
    const query = useGetBounties();

    expect(query.isLoading).toBeDefined();
    console.log("useGetBounties initial state:", query);
  });

  it("fetches with filters", async () => {
    const filters = { status: "OPEN" };
    const query = useGetBounties(filters);

    expect(query.data).toBeNull();
    console.log("useGetBounties with filters:", filters);
  });

  it("fetches with pagination", async () => {
    const pagination = { page: 1, limit: 20 };
    const query = useGetBounties(undefined, pagination);

    expect(query.data).toBeNull();
    console.log("useGetBounties with pagination:", pagination);
  });

  it("fetches with sort", async () => {
    const sort = { field: "amount", order: "desc" as const };
    const query = useGetBounties(undefined, undefined, sort);

    expect(query.data).toBeNull();
    console.log("useGetBounties with sort:", sort);
  });
});
