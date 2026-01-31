import { IEntity } from "../types.ts";

/**
 * Represents a system user
 */
export class User implements IEntity {
  id: string;
  username: string;
  email: string;
  createdAt: Date;
  updatedAt: Date;

  constructor(data: Partial<User>) {
    this.id = data.id || "";
    this.username = data.username || "";
    this.email = data.email || "";
    this.createdAt = data.createdAt || new Date();
    this.updatedAt = data.updatedAt || new Date();
  }

  isValid(): boolean {
    return this.username.length > 0 && this.email.includes("@");
  }
}
