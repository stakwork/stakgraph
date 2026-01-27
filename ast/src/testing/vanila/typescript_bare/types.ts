export interface IEntity {
  id: string;
  createdAt: Date;
  updatedAt: Date;
}

export type UUID = string;

export interface APIResponse<T> {
  success: boolean;
  data: T;
  error?: string;
}
