"use client";
import { useEffect, useState } from "react";
import { api } from "../../lib/api/apiClient";
import {
  Card,
  CardHeader,
  CardContent,
  CardTitle,
} from "../../components/ui/card";
import { Button } from "../../components/ui/button";
import { Input } from "../../components/ui/input";

function APIDemo() {
  const [users, setUsers] = useState<any[]>([]);
  const [posts, setPosts] = useState<any[]>([]);
  const [selectedUserId, setSelectedUserId] = useState("");
  const [selectedPostId, setSelectedPostId] = useState("");
  const [newUserName, setNewUserName] = useState("");
  const [newUserEmail, setNewUserEmail] = useState("");

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    const usersList = await api.users.list();
    setUsers(usersList);

    const postsList = await api.posts.list();
    setPosts(postsList);
  };

  const handleGetUser = async () => {
    if (selectedUserId) {
      const user = await api.users.get(selectedUserId);
      console.log("Fetched user:", user);
    }
  };

  const handleCreateUser = async () => {
    const newUser = await api.users.create({
      name: newUserName,
      email: newUserEmail,
    });
    setUsers([...users, newUser]);
    setNewUserName("");
    setNewUserEmail("");
  };

  const handleUpdateUser = async () => {
    if (selectedUserId) {
      await api.users.update(selectedUserId, { name: "Updated Name" });
      await loadData();
    }
  };

  const handleDeleteUser = async (id: string) => {
    await api.users.delete(id);
    setUsers(users.filter((u) => u.id !== id));
  };

  const handleGetPost = async () => {
    if (selectedPostId) {
      const post = await api.posts.get(selectedPostId);
      console.log("Fetched post:", post);
    }
  };

  const handleCreatePost = async () => {
    const newPost = await api.posts.create({
      title: "New Post",
      content: "Post content",
      authorId: "author-1",
    });
    setPosts([...posts, newPost]);
  };

  const handleDeletePost = async (id: string) => {
    await api.posts.delete(id);
    setPosts(posts.filter((p) => p.id !== id));
  };

  const handleGetComments = async () => {
    if (selectedPostId) {
      const comments = await api.comments.list(selectedPostId);
      console.log("Fetched comments:", comments);
    }
  };

  const handleCreateComment = async () => {
    if (selectedPostId) {
      await api.comments.create({
        postId: selectedPostId,
        text: "New comment",
      });
    }
  };

  const handleDeleteComment = async (id: string) => {
    await api.comments.delete(id);
  };

  return (
    <main className="max-w-6xl mx-auto py-8">
      <div className="grid grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Users API</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <h4 className="font-medium mb-2">Create User</h4>
                <div className="space-y-2">
                  <Input
                    placeholder="Name"
                    value={newUserName}
                    onChange={(e) => setNewUserName(e.target.value)}
                  />
                  <Input
                    placeholder="Email"
                    value={newUserEmail}
                    onChange={(e) => setNewUserEmail(e.target.value)}
                  />
                  <Button onClick={handleCreateUser}>Create</Button>
                </div>
              </div>

              <div>
                <h4 className="font-medium mb-2">User Operations</h4>
                <div className="space-y-2">
                  <Input
                    placeholder="User ID"
                    value={selectedUserId}
                    onChange={(e) => setSelectedUserId(e.target.value)}
                  />
                  <div className="flex gap-2">
                    <Button onClick={handleGetUser} variant="outline">
                      Get
                    </Button>
                    <Button onClick={handleUpdateUser} variant="outline">
                      Update
                    </Button>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="font-medium mb-2">Users List</h4>
                <ul className="space-y-1">
                  {users.map((user) => (
                    <li key={user.id} className="flex justify-between text-sm">
                      <span>{user.name}</span>
                      <Button
                        onClick={() => handleDeleteUser(user.id)}
                        variant="destructive"
                        size="sm"
                      >
                        Delete
                      </Button>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Posts & Comments API</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <h4 className="font-medium mb-2">Post Operations</h4>
                <div className="space-y-2">
                  <Input
                    placeholder="Post ID"
                    value={selectedPostId}
                    onChange={(e) => setSelectedPostId(e.target.value)}
                  />
                  <div className="flex gap-2">
                    <Button onClick={handleGetPost} variant="outline">
                      Get Post
                    </Button>
                    <Button onClick={handleCreatePost}>Create Post</Button>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="font-medium mb-2">Comment Operations</h4>
                <div className="flex gap-2">
                  <Button onClick={handleGetComments} variant="outline">
                    Get Comments
                  </Button>
                  <Button onClick={handleCreateComment}>Add Comment</Button>
                </div>
              </div>

              <div>
                <h4 className="font-medium mb-2">Posts List</h4>
                <ul className="space-y-1">
                  {posts.map((post) => (
                    <li key={post.id} className="flex justify-between text-sm">
                      <span>{post.title}</span>
                      <Button
                        onClick={() => handleDeletePost(post.id)}
                        variant="destructive"
                        size="sm"
                      >
                        Delete
                      </Button>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </main>
  );
}

export { APIDemo as default };
