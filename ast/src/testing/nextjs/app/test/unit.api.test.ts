// @ts-nocheck
import { api } from "../../lib/api/apiClient";

describe("unit: api client - users", () => {
  it("gets user by id", async () => {
    const userId = "user-123";

    const user = await api.users.get(userId);

    expect(user).toBeDefined();
    console.log("api.users.get:", user);
  });

  it("lists all users", async () => {
    const users = await api.users.list();

    expect(Array.isArray(users)).toBe(true);
    console.log("api.users.list:", users.length, "users");
  });

  it("creates a new user", async () => {
    const newUser = {
      name: "Test User",
      email: "test@example.com",
    };

    const created = await api.users.create(newUser);

    expect(created).toBeDefined();
    expect(created.name).toBe(newUser.name);
    console.log("api.users.create:", created);
  });

  it("updates existing user", async () => {
    const userId = "user-123";
    const updates = { name: "Updated Name" };

    const updated = await api.users.update(userId, updates);

    expect(updated).toBeDefined();
    console.log("api.users.update:", updated);
  });

  it("deletes user", async () => {
    const userId = "user-to-delete";

    await api.users.delete(userId);

    console.log("api.users.delete: user deleted");
  });
});

describe("unit: api client - posts", () => {
  it("gets post by id", async () => {
    const postId = "post-123";

    const post = await api.posts.get(postId);

    expect(post).toBeDefined();
    console.log("api.posts.get:", post);
  });

  it("lists all posts", async () => {
    const posts = await api.posts.list();

    expect(Array.isArray(posts)).toBe(true);
    console.log("api.posts.list:", posts.length, "posts");
  });

  it("creates a new post", async () => {
    const newPost = {
      title: "Test Post",
      content: "Post content here",
      authorId: "author-1",
    };

    const created = await api.posts.create(newPost);

    expect(created).toBeDefined();
    expect(created.title).toBe(newPost.title);
    console.log("api.posts.create:", created);
  });

  it("deletes post", async () => {
    const postId = "post-to-delete";

    await api.posts.delete(postId);

    console.log("api.posts.delete: post deleted");
  });
});

describe("unit: api client - comments", () => {
  it("lists comments for a post", async () => {
    const postId = "post-123";

    const comments = await api.comments.list(postId);

    expect(Array.isArray(comments)).toBe(true);
    console.log("api.comments.list:", comments.length, "comments");
  });

  it("creates a new comment", async () => {
    const newComment = {
      postId: "post-123",
      text: "This is a comment",
    };

    const created = await api.comments.create(newComment);

    expect(created).toBeDefined();
    expect(created.text).toBe(newComment.text);
    console.log("api.comments.create:", created);
  });

  it("deletes comment", async () => {
    const commentId = "comment-to-delete";

    await api.comments.delete(commentId);

    console.log("api.comments.delete: comment deleted");
  });
});

describe("unit: api client - chained operations", () => {
  it("performs user workflow", async () => {
    const users = await api.users.list();
    expect(Array.isArray(users)).toBe(true);

    const newUser = await api.users.create({
      name: "Workflow User",
      email: "workflow@example.com",
    });
    expect(newUser).toBeDefined();

    const fetched = await api.users.get(newUser.id);
    expect(fetched.id).toBe(newUser.id);

    const updated = await api.users.update(newUser.id, {
      name: "Updated Workflow User",
    });
    expect(updated.name).toBe("Updated Workflow User");

    await api.users.delete(newUser.id);

    console.log("User workflow completed");
  });

  it("performs post and comment workflow", async () => {
    const post = await api.posts.create({
      title: "Workflow Post",
      content: "Content here",
      authorId: "author-1",
    });
    expect(post).toBeDefined();

    const comment = await api.comments.create({
      postId: post.id,
      text: "First comment",
    });
    expect(comment.postId).toBe(post.id);

    const comments = await api.comments.list(post.id);
    expect(comments.length).toBeGreaterThan(0);

    await api.comments.delete(comment.id);
    await api.posts.delete(post.id);

    console.log("Post and comment workflow completed");
  });

  it("chains multiple api calls", async () => {
    const user = await api.users.create({
      name: "Chain User",
      email: "chain@example.com",
    });

    const post1 = await api.posts.create({
      title: "Post 1",
      content: "Content 1",
      authorId: user.id,
    });

    const post2 = await api.posts.create({
      title: "Post 2",
      content: "Content 2",
      authorId: user.id,
    });

    const comment1 = await api.comments.create({
      postId: post1.id,
      text: "Comment on post 1",
    });

    const comment2 = await api.comments.create({
      postId: post2.id,
      text: "Comment on post 2",
    });

    expect(user.id).toBeDefined();
    expect(post1.id).toBeDefined();
    expect(post2.id).toBeDefined();
    expect(comment1.id).toBeDefined();
    expect(comment2.id).toBeDefined();

    console.log("Chained multiple API calls successfully");
  });
});
