import { Request, Response, NextFunction } from "express";

let busy = false;

export function getBusy(): boolean {
  return busy;
}

export function setBusy(value: boolean): void {
  busy = value;
}

export function busyMiddleware(
  req: Request,
  res: Response,
  next: NextFunction
): void {
  setBusy(true);

  const cleanup = () => setBusy(false);

  res.on("finish", cleanup);
  res.on("close", cleanup);

  next();
}
