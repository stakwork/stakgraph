import { User } from '../model';

// @ast node: Class "UserService"
export class UserService {
  // @ast node: Function "findAll"
  async findAll() {
    // In a real app, this would query a database
    return [
      { id: "1", name: "User 1" },
      { id: "2", name: "User 2" },
    ];
  }

  // @ast node: Function "findById"
  async findById(id: string) {
    return { id, name: `User ${id}` };
  }

  // @ast node: Function "create"
  async create(userData: any) {
    return { id: "3", ...userData };
  }

  // @ast node: Function "update"
  async update(id: string, userData: any) {
    return { id, ...userData };
  }

  // @ast node: Function "delete"
  async delete(id: string) {
    return true;
  }
}