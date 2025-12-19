import { User } from '../model';

export class UserService {
  async findAll() {
    // In a real app, this would query a database
    return [{ id: '1', name: 'User 1' }, { id: '2', name: 'User 2' }];
  }

  async findById(id: string) {
    return { id, name: `User ${id}` };
  }

  async create(userData: any) {
    return { id: '3', ...userData };
  }

  async update(id: string, userData: any) {
    return { id, ...userData };
  }

  async delete(id: string) {
    return true;
  }
}