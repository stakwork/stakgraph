import { Request, Response, NextFunction } from "express";

let busy = false;

export function getBusy(): boolean {
  console.log(`[getBusy] Returning busy=${busy}`);
  return busy;
}

export function setBusy(value: boolean): void {
  console.log(`[setBusy] Setting busy=${value}`);
  busy = value;
}

export function busyMiddleware(
  req: Request,
  res: Response,
  next: NextFunction
): void {
  setBusy(true);
  console.log(`[busyMiddleware] Set busy=true for ${req.method} ${req.path}`);

  const cleanup = () => {
    console.log(
      `[busyMiddleware] Response finished for ${req.method} ${req.path}, setting busy=false`
    );
    setBusy(false);
  };

  res.on("finish", cleanup);
  res.on("close", cleanup);

  next();
}
