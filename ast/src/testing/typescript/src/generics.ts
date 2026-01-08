// 1. Generic Functions
export function identity<T>(arg: T): T {
  return arg;
}

export const combine = <A, B>(a: A, b: B): [A, B] => [a, b];

export async function fetchData<T>(url: string): Promise<T> {
  const response = await fetch(url);
  return response.json();
}

// 2. Generic Interfaces
export interface Repository<T> {
  findById(id: string): Promise<T | null>;
  findAll(): Promise<T[]>;
  save(entity: T): Promise<T>;
  delete(id: string): Promise<boolean>;
}

export interface Mapper<TInput, TOutput> {
  map(input: TInput): TOutput;
}

// 3. Generic Classes
export class GenericRepository<T> implements Repository<T> {
  private items: Map<string, T> = new Map();

  async findById(id: string): Promise<T | null> {
    return this.items.get(id) ?? null;
  }

  async findAll(): Promise<T[]> {
    return Array.from(this.items.values());
  }

  async save(entity: T): Promise<T> {
    return entity;
  }

  async delete(id: string): Promise<boolean> {
    return this.items.delete(id);
  }
}

// 4. Generic Type Aliases
export type Nullable<T> = T | null;
export type AsyncResult<T> = Promise<{ data: T; error: Error | null }>;
export type Pair<A, B = A> = [A, B];

// 5. Generic with Constraints
export function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
  return obj[key];
}

export class KeyValueStore<K extends string | number, V> {
  private store: Map<K, V> = new Map();

  set(key: K, value: V): void {
    this.store.set(key, value);
  }

  get(key: K): V | undefined {
    return this.store.get(key);
  }
}

// 6. Utility Types Usage
export type UserDTO = {
  id: string;
  name: string;
  email: string;
  password: string;
};

export type SafeUser = Omit<UserDTO, "password">;
export type UserUpdate = Partial<UserDTO>;
export type UserKeys = keyof UserDTO;
export type ReadonlyUser = Readonly<UserDTO>;
