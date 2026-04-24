import { PersonService } from "./service";
import type { SequelizePerson } from "./model";
import * as models from "./model";
import { SequelizePerson as SP, TypeOrmPerson as TP } from "./service";
// Side-effect import
import "./config";

export * from "./model";
export { PersonService } from "./service";

// 1. Type Aliases
// @ast node: DataModel "ID"
export type ID = string | number;

/** Data Transfer Object for User */
// @ast node: DataModel "UserDTO"
export type UserDTO = {
  id: ID;
  username: string;
  email: string;
};

// Type alias with methods -> Should be Trait?
// @ast node: Trait "Logger"
export type Logger = {
  log(msg: string): void;
  error(msg: string): void;
};

// 2. Enums
// @ast node: DataModel "UserRole"
export enum UserRole {
  ADMIN = "ADMIN",
  USER = "USER",
  GUEST = "GUEST",
}

// @ast node: DataModel "Status"
export const enum Status {
  Active = 1,
  Inactive = 0,
}

// 3. Interface as DataModel (no methods)
// @ast node: DataModel "Config"
export interface Config {
  apiKey: string;
  timeout: number;
}

// 4. Interface as Trait (methods)
// @ast node: Trait "IGreeter"
export interface IGreeter {
  greet(name: string): string;
}
// @ast node: DataModel "Logger"
// @ast node: DataModel "IGreeter"
