import { Entity, Column, PrimaryGeneratedColumn } from "typeorm";

// Interface with optional fields
// @ast node: DataModel "UserDTO"
export interface UserDTO {
  id: number;
  name: string;
  email: string;
  avatar?: string;
  createdAt?: Date;
}

// Type alias with generics
// @ast node: DataModel "ApiResponse"
export type ApiResponse<T> = {
  data: T;
  status: number;
  message: string;
  pagination?: {
    page: number;
    total: number;
  };
};

// Type alias for union
// @ast node: DataModel "UserRole"
export type UserRole = "admin" | "user" | "guest";

// Enum with string values
// @ast node: DataModel "UserStatus"
export enum UserStatus {
  Active = "ACTIVE",
  Inactive = "INACTIVE",
  Pending = "PENDING",
  Banned = "BANNED",
}

// Entity decorator pattern (TypeORM)
@Entity()
// @ast node: Class "UserEntity"
export class UserEntity {
  @PrimaryGeneratedColumn()
  id!: number;

  @Column()
  name!: string;

  @Column({ unique: true })
  email!: string;

  @Column({ type: "enum", enum: UserStatus, default: UserStatus.Pending })
  status!: UserStatus;
}

// Class extending Model pattern
// @ast node: Class "Model"
export class Model {
  id!: number;
  createdAt!: Date;
  updatedAt!: Date;
}

// @ast node: Class "Product"
export class Product extends Model {
  name!: string;
  price!: number;
  description?: string;
}

// Complex type with nested structure
// @ast node: DataModel "OrderWithDetails"
export type OrderWithDetails = {
  id: number;
  user: UserDTO;
  items: Array<{
    productId: number;
    quantity: number;
    price: number;
  }>;
  total: number;
};

// Interface extending another
// @ast node: DataModel "AdminUser"
export interface AdminUser extends UserDTO {
  permissions: string[];
  department: string;
}
