import { PersonService } from "./service";
import type { SequelizePerson } from "./model";
import * as models from "./model";
import { SequelizePerson as SP, TypeOrmPerson as TP } from "./service";
// Side-effect import
import "./config";

// 1. Type Aliases
export type ID = string | number;

/** Data Transfer Object for User */
export type UserDTO = {
  id: ID;
  username: string;
  email: string;
};

// Type alias with methods -> Should be Trait?
export type Logger = {
  log(msg: string): void;
  error(msg: string): void;
};

// 2. Enums
export enum UserRole {
  ADMIN = "ADMIN",
  USER = "USER",
  GUEST = "GUEST",
}

export const enum Status {
  Active = 1,
  Inactive = 0,
}

// 3. Interface as DataModel (no methods)
export interface Config {
  apiKey: string;
  timeout: number;
}

// 4. Interface as Trait (methods)
export interface IGreeter {
  greet(name: string): string;
}
