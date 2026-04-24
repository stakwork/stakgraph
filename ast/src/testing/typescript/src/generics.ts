// 1. Generic Functions
// @ast node: Function "identity"
export function identity<T>(arg: T): T {
  return arg;
}

// @ast node: Function "combine"
export const combine = <A, B>(a: A, b: B): [A, B] => [a, b];

// @ast node: Function "fetchData"
export async function fetchData<T>(url: string): Promise<T> {
  const response = await fetch(url);
  return response.json();
}

// 2. Generic Interfaces
// @ast node: Trait "Repository"
export interface Repository<T> {
  findById(id: string): Promise<T | null>;
  findAll(): Promise<T[]>;
  save(entity: T): Promise<T>;
  delete(id: string): Promise<boolean>;
}

// @ast node: Trait "Mapper"
export interface Mapper<TInput, TOutput> {
  map(input: TInput): TOutput;
}

// 3. Generic Classes
// @ast node: Class "GenericRepository"
export class GenericRepository<T> implements Repository<T> {
  private items: Map<string, T> = new Map();

  // @ast node: Function "findById"
  async findById(id: string): Promise<T | null> {
    return this.items.get(id) ?? null;
  }

  // @ast node: Function "findAll"
  async findAll(): Promise<T[]> {
    return Array.from(this.items.values());
  }

  // @ast node: Function "save"
  async save(entity: T): Promise<T> {
    return entity;
  }

  // @ast node: Function "delete"
  async delete(id: string): Promise<boolean> {
    return this.items.delete(id);
  }
}

// 4. Generic Type Aliases
// @ast node: DataModel "Nullable"
export type Nullable<T> = T | null;
// @ast node: DataModel "AsyncResult"
export type AsyncResult<T> = Promise<{ data: T; error: Error | null }>;
// @ast node: DataModel "Pair"
export type Pair<A, B = A> = [A, B];

// 5. Generic with Constraints
// @ast node: Function "getProperty"
export function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
  return obj[key];
}

// @ast node: Class "KeyValueStore"
export class KeyValueStore<K extends string | number, V> {
  private store: Map<K, V> = new Map();

  // @ast node: Function "set"
  set(key: K, value: V): void {
    this.store.set(key, value);
  }

  // @ast node: Function "get"
  get(key: K): V | undefined {
    return this.store.get(key);
  }
}

// 6. Utility Types Usage
// @ast node: DataModel "UserDTO"
export type UserDTO = {
  id: string;
  name: string;
  email: string;
  password: string;
};

// @ast node: DataModel "SafeUser"
export type SafeUser = Omit<UserDTO, "password">;
// @ast node: DataModel "UserUpdate"
export type UserUpdate = Partial<UserDTO>;
// @ast node: DataModel "UserKeys"
export type UserKeys = keyof UserDTO;
// @ast node: DataModel "ReadonlyUser"
export type ReadonlyUser = Readonly<UserDTO>;
// @ast node: DataModel "Repository"
// @ast node: DataModel "Mapper"
