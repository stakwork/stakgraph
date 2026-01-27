import { IEntity } from "../types.ts";

export class Post implements IEntity {
  id: string;
  authorId: string;
  title: string;
  content: string;
  createdAt: Date;
  updatedAt: Date;

  constructor(data: Partial<Post>) {
    this.id = data.id || "";
    this.authorId = data.authorId || "";
    this.title = data.title || "";
    this.content = data.content || "";
    this.createdAt = data.createdAt || new Date();
    this.updatedAt = data.updatedAt || new Date();
  }
}
