// Server Action Generic Types
export type ActionResult<T> =
  | { success: true; data: T }
  | { success: false; error: string };

export async function handleAction<T>(
  action: () => Promise<T>
): Promise<ActionResult<T>> {
  try {
    const data = await action();
    return { success: true, data };
  } catch (error) {
    return { success: false, error: String(error) };
  }
}

// API Response Types
export interface ApiResponse<T> {
  data: T;
  metadata: {
    total: number;
    page: number;
    pageSize: number;
  };
}

export interface PaginatedResult<T> {
  items: T[];
  nextCursor: string | null;
  hasMore: boolean;
}

// Generic Service Class
export abstract class BaseService<T extends { id: string }> {
  abstract findById(id: string): Promise<T | null>;
  abstract create(data: Omit<T, "id">): Promise<T>;
  abstract update(id: string, data: Partial<T>): Promise<T>;
  abstract delete(id: string): Promise<void>;

  async findOrCreate(id: string, defaults: Omit<T, "id">): Promise<T> {
    const existing = await this.findById(id);
    if (existing) return existing;
    return this.create(defaults);
  }
}

// Cache utilities
export class Cache<K extends string, V> {
  private cache = new Map<K, { value: V; expiry: number }>();

  set(key: K, value: V, ttlMs: number): void {
    this.cache.set(key, { value, expiry: Date.now() + ttlMs });
  }

  get(key: K): V | null {
    const entry = this.cache.get(key);
    if (!entry) return null;
    if (Date.now() > entry.expiry) {
      this.cache.delete(key);
      return null;
    }
    return entry.value;
  }
}
