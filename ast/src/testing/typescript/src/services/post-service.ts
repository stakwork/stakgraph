// @ast node: Class "PostService"
export class PostService {
  // @ast node: Function "findByUserId"
  async findByUserId(userId: string) {
    return [
      { id: "1", userId, title: "First Post" },
      { id: "2", userId, title: "Second Post" },
    ];
  }

  // @ast node: Function "findOne"
  async findOne(userId: string, postId: string) {
    return { id: postId, userId, title: `Post ${postId}` };
  }

  // @ast node: Function "create"
  async create(userId: string, postData: any) {
    return { id: "3", userId, ...postData };
  }

  // @ast node: Function "likePost"
  async likePost(postId: string) {
    return true;
  }

  // @ast node: Function "unlikePost"
  async unlikePost(postId: string) {
    return true;
  }
}
