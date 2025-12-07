export interface Logger {
  info(message: string): void;
  error(message: string): void;
  warn?(message: string): void;
  debug?(message: string): void;
  trace?(message: string): void;
  fatal?(message: string): void;
}

export function consoleLogger(name: string): Logger {
  return {
    info(message: string): void {
      console.log(`[${name}] ${message}`);
    },
    error(message: string): void {
      console.error(`[${name}] ${message}`);
    },
    warn(message: string): void {
      console.warn(`[${name}] ${message}`);
    },
    debug(message: string): void {
      console.debug(`[${name}] ${message}`);
    },
    trace(message: string): void {
      console.trace(`[${name}] ${message}`);
    },
    fatal(message: string): void {
      console.error(`==FATAL== [${name}] ${message}`);
    },
  };
}
