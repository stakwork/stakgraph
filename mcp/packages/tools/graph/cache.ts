import NodeCache from "node-cache";
import { Request, Response, NextFunction } from "express";

const cache = new NodeCache({
  stdTTL: 300, // 5 minutes default TTL
  checkperiod: 60, // check for expired keys every 60 seconds
  useClones: false, // better performance, but be careful with object mutations
});

export function cacheMiddleware() {
  const ttlSeconds = 3600 * 24; // 1 day
  return (req: Request, res: Response, next: NextFunction): void => {
    // Skip caching for non-GET requests
    if (req.method !== "GET") {
      return next();
    }

    const cacheKey = `${req.method}:${req.path}:${JSON.stringify(req.query)}`;

    const cachedData = cache.get(cacheKey);
    if (cachedData !== undefined) {
      console.log(`Cache hit for ${cacheKey}`);
      res.send(cachedData);
      return; // Important: return void, not the response
    }

    // Store original methods
    const originalSend = res.send;
    const originalJson = res.json;

    // Override send to cache the response
    res.send = function (data: any) {
      cache.set(cacheKey, data, ttlSeconds);
      console.log(`Cached response for ${cacheKey}`);
      return originalSend.call(this, data);
    };

    // Override json to cache the response
    res.json = function (data: any) {
      cache.set(cacheKey, data, ttlSeconds);
      console.log(`Cached JSON response for ${cacheKey}`);
      return originalJson.call(this, data);
    };

    next();
  };
}

// Utility functions for cache management
export function clearCache(): void {
  cache.flushAll();
  console.log("Cache cleared");
}

export function getCacheStats() {
  return cache.getStats();
}

export function getCacheKeys(): string[] {
  return cache.keys();
}

// Cache info endpoint
export function cacheInfo(req: Request, res: Response): void {
  res.json({
    stats: cache.getStats(),
    keys: cache.keys().length,
    keysList: cache.keys(),
  });
}
