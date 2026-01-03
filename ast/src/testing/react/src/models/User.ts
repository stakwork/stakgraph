import { Entity, Column, PrimaryGeneratedColumn } from "typeorm";

// Interface with optional fields
export interface UserDTO {
  id: number;
  name: string;
  email: string;
  avatar?: string;
  createdAt?: Date;
}

// Type alias with generics
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
export type UserRole = "admin" | "user" | "guest";

// Enum with string values
export enum UserStatus {
  Active = "ACTIVE",
  Inactive = "INACTIVE",
  Pending = "PENDING",
  Banned = "BANNED",
}

// Entity decorator pattern (TypeORM)
@Entity()
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
export class Model {
  id!: number;
  createdAt!: Date;
  updatedAt!: Date;
}

export class Product extends Model {
  name!: string;
  price!: number;
  description?: string;
}

// Complex type with nested structure
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
export interface AdminUser extends UserDTO {
  permissions: string[];
  department: string;
}
