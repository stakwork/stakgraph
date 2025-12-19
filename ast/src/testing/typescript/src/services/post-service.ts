export class PostService {
  async findByUserId(userId: string) {
    return [
      { id: '1', userId, title: 'First Post' },
      { id: '2', userId, title: 'Second Post' }
    ];
  }

  async findOne(userId: string, postId: string) {
    return { id: postId, userId, title: `Post ${postId}` };
  }

  async create(userId: string, postData: any) {
    return { id: '3', userId, ...postData };
  }

  async likePost(postId: string) {
    return true;
  }

  async unlikePost(postId: string) {
    return true;
  }
}