interface User {
  id: string;
  name: string;
  email: string;
}

interface Post {
  id: string;
  title: string;
  content: string;
  authorId: string;
}

interface Comment {
  id: string;
  postId: string;
  text: string;
}

// @ast node: Class "UsersAPI"
// @ast edge: Contains <- File "apiClient.ts" "src/testing/nextjs/lib/api/apiClient.ts"
class UsersAPI {
  async get(id: string): Promise<User> {
    const response = await fetch(`/api/users/${id}`);
    const data = await response.json();
    console.log("UsersAPI.get:", id, data);
    return data;
  }

  async list(): Promise<User[]> {
    const response = await fetch("/api/users");
    const data = await response.json();
    console.log("UsersAPI.list:", data);
    return data;
  }

  async create(user: Omit<User, "id">): Promise<User> {
    const response = await fetch("/api/users", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(user),
    });
    const data = await response.json();
    console.log("UsersAPI.create:", data);
    return data;
  }

  async delete(id: string): Promise<void> {
    await fetch(`/api/users/${id}`, { method: "DELETE" });
    console.log("UsersAPI.delete:", id);
  }

  async update(id: string, user: Partial<User>): Promise<User> {
    const response = await fetch(`/api/users/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(user),
    });
    const data = await response.json();
    console.log("UsersAPI.update:", id, data);
    return data;
  }
}

// @ast node: Class "PostsAPI"
// @ast edge: Contains <- File "apiClient.ts" "src/testing/nextjs/lib/api/apiClient.ts"
class PostsAPI {
  async get(id: string): Promise<Post> {
    const response = await fetch(`/api/posts/${id}`);
    const data = await response.json();
    console.log("PostsAPI.get:", id, data);
    return data;
  }

  async list(): Promise<Post[]> {
    const response = await fetch("/api/posts");
    const data = await response.json();
    console.log("PostsAPI.list:", data);
    return data;
  }

  async create(post: Omit<Post, "id">): Promise<Post> {
    const response = await fetch("/api/posts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(post),
    });
    const data = await response.json();
    console.log("PostsAPI.create:", data);
    return data;
  }

  async delete(id: string): Promise<void> {
    await fetch(`/api/posts/${id}`, { method: "DELETE" });
    console.log("PostsAPI.delete:", id);
  }
}

// @ast node: Class "CommentsAPI"
// @ast edge: Contains <- File "apiClient.ts" "src/testing/nextjs/lib/api/apiClient.ts"
class CommentsAPI {
  async list(postId: string): Promise<Comment[]> {
    const response = await fetch(`/api/posts/${postId}/comments`);
    const data = await response.json();
    console.log("CommentsAPI.list:", postId, data);
    return data;
  }

  async create(comment: Omit<Comment, "id">): Promise<Comment> {
    const response = await fetch("/api/comments", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(comment),
    });
    const data = await response.json();
    console.log("CommentsAPI.create:", data);
    return data;
  }

  async delete(id: string): Promise<void> {
    await fetch(`/api/comments/${id}`, { method: "DELETE" });
    console.log("CommentsAPI.delete:", id);
  }
}

// @ast node: Class "APIClient"
// @ast edge: Contains <- File "apiClient.ts" "src/testing/nextjs/lib/api/apiClient.ts"
class APIClient {
  users = new UsersAPI();
  posts = new PostsAPI();
  comments = new CommentsAPI();
}

// @ast node: Var "api"
// @ast edge: Contains <- File "apiClient.ts" "src/testing/nextjs/lib/api/apiClient.ts"
export const api = new APIClient();
// @ast node: Function "get"
// @ast node: Function "list"
// @ast node: Function "create"
// @ast node: Function "delete"
// @ast node: Function "update"
// @ast node: Function "get"
// @ast node: Function "list"
// @ast node: Function "create"
// @ast node: Function "delete"
// @ast node: Function "list"
// @ast node: Function "create"
// @ast node: Function "delete"
// @ast node: DataModel "User"
// @ast node: DataModel "Post"
// @ast node: DataModel "Comment"
// @ast node: Request "/api/users/${id}"
// @ast node: Request "/api/users"
// @ast node: Request "/api/users"
// @ast node: Request "/api/users/${id}"
// @ast node: Request "/api/users/${id}"
// @ast node: Request "/api/posts/${id}"
// @ast node: Request "/api/posts"
// @ast node: Request "/api/posts"
// @ast node: Request "/api/posts/${id}"
// @ast node: Request "/api/posts/${postId}/comments"
// @ast node: Request "/api/comments"
// @ast node: Request "/api/comments/${id}"
