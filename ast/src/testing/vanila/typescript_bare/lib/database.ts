import { IEntity } from "../types.ts";

export class Database<T extends IEntity> {
  private store: Map<string, T> = new Map();

  async findById(id: string): Promise<T | null> {
    return this.store.get(id) || null;
  }

  async save(item: T): Promise<T> {
    this.store.set(item.id, item);
    return item;
  }

  async delete(id: string): Promise<boolean> {
    return this.store.delete(id);
  }
}

export const db = {
  users: new Database<any>(),
  posts: new Database<any>(),
};
