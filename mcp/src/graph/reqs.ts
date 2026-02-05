import * as uuid from "uuid";

interface Request {
  status: Status;
  result?: any;
  error?: any;
  progress?: any;
}

const REQS: Record<string, Request> = {};
const REQ_ORDER: string[] = []; // Track insertion order
const MAX_REQS = 100;

type Status = "pending" | "completed" | "failed";

export function startReq(): string {
  const key = uuid.v4();

  // If we're at the limit, remove the oldest request
  if (REQ_ORDER.length >= MAX_REQS) {
    const oldestKey = REQ_ORDER.shift();
    if (oldestKey) {
      delete REQS[oldestKey];
    }
  }

  REQS[key] = {
    status: "pending",
    result: undefined,
  };
  REQ_ORDER.push(key);

  return key;
}

export function finishReq(id: string, result: any) {
  if (REQS[id]) {
    REQS[id].status = "completed";
    REQS[id].result = result;
  }
}

export function failReq(id: string, error: any) {
  if (REQS[id]) {
    REQS[id].status = "failed";
    REQS[id].error = error;
  }
}

export function updateReq(id: string, progress: any) {
  if (REQS[id]) {
    REQS[id].progress = progress;
  }
}

export function checkReq(id: string): Request {
  return REQS[id];
}
