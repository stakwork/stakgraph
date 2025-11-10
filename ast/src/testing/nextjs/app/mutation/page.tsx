"use client";
import { useState } from "react";
import { useMutation } from "../../lib/hooks/useMutation";
import {
  Card,
  CardHeader,
  CardContent,
  CardTitle,
} from "../../components/ui/card";
import { Button } from "../../components/ui/button";
import { Input } from "../../components/ui/input";

interface UserData {
  name: string;
  email: string;
  age: number;
}

interface ItemData {
  title: string;
  price: number;
}

function MutationDemo() {
  const userMutation = useMutation<UserData>("/api/users");
  const itemMutation = useMutation<ItemData>("/api/items");

  const [userName, setUserName] = useState("");
  const [userEmail, setUserEmail] = useState("");
  const [userAge, setUserAge] = useState("");

  const [itemTitle, setItemTitle] = useState("");
  const [itemPrice, setItemPrice] = useState("");

  const handleCreateUser = async () => {
    await userMutation.mutate({
      name: userName,
      email: userEmail,
      age: Number(userAge),
    });
    setUserName("");
    setUserEmail("");
    setUserAge("");
  };

  const handleCreateItem = async () => {
    await itemMutation.mutate({
      title: itemTitle,
      price: Number(itemPrice),
    });
    setItemTitle("");
    setItemPrice("");
  };

  const handleResetUser = () => {
    userMutation.reset();
  };

  const handleResetItem = () => {
    itemMutation.reset();
  };

  return (
    <main className="max-w-4xl mx-auto py-8">
      <div className="grid grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Create User</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <Input
                placeholder="Name"
                value={userName}
                onChange={(e) => setUserName(e.target.value)}
              />
              <Input
                placeholder="Email"
                type="email"
                value={userEmail}
                onChange={(e) => setUserEmail(e.target.value)}
              />
              <Input
                placeholder="Age"
                type="number"
                value={userAge}
                onChange={(e) => setUserAge(e.target.value)}
              />

              <div className="flex gap-2">
                <Button
                  onClick={handleCreateUser}
                  disabled={userMutation.isLoading}
                >
                  {userMutation.isLoading ? "Creating..." : "Create User"}
                </Button>
                <Button onClick={handleResetUser} variant="outline">
                  Reset
                </Button>
              </div>

              {userMutation.error && (
                <p className="text-red-600 text-sm">{userMutation.error}</p>
              )}

              {userMutation.data && (
                <div className="p-3 bg-green-50 rounded">
                  <p className="text-sm">User created successfully!</p>
                  <pre className="text-xs mt-2">
                    {JSON.stringify(userMutation.data, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Create Item</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <Input
                placeholder="Title"
                value={itemTitle}
                onChange={(e) => setItemTitle(e.target.value)}
              />
              <Input
                placeholder="Price"
                type="number"
                value={itemPrice}
                onChange={(e) => setItemPrice(e.target.value)}
              />

              <div className="flex gap-2">
                <Button
                  onClick={handleCreateItem}
                  disabled={itemMutation.isLoading}
                >
                  {itemMutation.isLoading ? "Creating..." : "Create Item"}
                </Button>
                <Button onClick={handleResetItem} variant="outline">
                  Reset
                </Button>
              </div>

              {itemMutation.error && (
                <p className="text-red-600 text-sm">{itemMutation.error}</p>
              )}

              {itemMutation.data && (
                <div className="p-3 bg-green-50 rounded">
                  <p className="text-sm">Item created successfully!</p>
                  <pre className="text-xs mt-2">
                    {JSON.stringify(itemMutation.data, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </main>
  );
}

export { MutationDemo as default };
