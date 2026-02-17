import { Request, Response, NextFunction } from "express";

interface TrackedOperation {
  id: string;
  routeName: string;
  startTime: number;
  timeout: NodeJS.Timeout;
}

const TIMEOUT_MS = 30 * 60 * 1000;
const activeOperations = new Map<string, TrackedOperation>();
let operationCounter = 0;

function generateOperationId(routeName: string): string {
  const sanitized = routeName.replace(/\W+/g, "_");
  return `${sanitized}_${Date.now()}_${++operationCounter}`;
}

function cleanup(operationId: string): void {
  const op = activeOperations.get(operationId);
  if (op) {
    clearTimeout(op.timeout);
    activeOperations.delete(operationId);
    console.log(
      `[busy] Cleaned up operation ${operationId}. Active: ${activeOperations.size}`
    );
  }
}

export function startTracking(routeName: string): string {
  const operationId = generateOperationId(routeName);
  const timeout = setTimeout(() => {
    console.warn(
      `[busy] TIMEOUT: Operation ${operationId} (${routeName}) exceeded 30 minutes, forcing cleanup`
    );
    cleanup(operationId);
  }, TIMEOUT_MS);

  activeOperations.set(operationId, {
    id: operationId,
    routeName,
    startTime: Date.now(),
    timeout,
  });

  console.log(
    `[busy] Started tracking ${operationId} (${routeName}). Active: ${activeOperations.size}`
  );
  return operationId;
}

export function endTracking(operationId: string): void {
  cleanup(operationId);
}

export function getBusy(): boolean {
  const isBusy = activeOperations.size > 0;
  console.log(`[getBusy] Active operations: ${activeOperations.size}, returning busy=${isBusy}`);
  return isBusy;
}

export function busyMiddleware(
  req: Request,
  res: Response,
  next: NextFunction
): void {
  const operationId = startTracking(`${req.method} ${req.path}`);

  const cleanup = () => {
    endTracking(operationId);
  };

  res.on("finish", cleanup);
  res.on("close", cleanup);

  next();
}
