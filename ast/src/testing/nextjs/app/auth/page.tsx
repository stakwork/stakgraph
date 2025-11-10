"use client";
import { useState } from "react";
import { useAuth } from "../../lib/context/AuthContext";
import {
  Card,
  CardHeader,
  CardContent,
  CardTitle,
} from "../../components/ui/card";
import { Button } from "../../components/ui/button";
import { Input } from "../../components/ui/input";

function AuthPage() {
  const auth = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [newName, setNewName] = useState("");

  const handleLogin = async () => {
    await auth.login({ email, password });
    setEmail("");
    setPassword("");
  };

  const handleLogout = () => {
    auth.logout();
  };

  const handleRefresh = async () => {
    await auth.refreshToken();
  };

  const handleUpdateProfile = () => {
    auth.updateProfile(newName);
    setNewName("");
  };

  return (
    <main className="max-w-2xl mx-auto py-8">
      <Card>
        <CardHeader>
          <CardTitle>Authentication Demo</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {!auth.isAuthenticated ? (
              <div className="space-y-4">
                <h3 className="font-semibold">Login</h3>
                <Input
                  type="email"
                  placeholder="Email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
                <Input
                  type="password"
                  placeholder="Password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
                <Button onClick={handleLogin}>Login</Button>
              </div>
            ) : (
              <div className="space-y-4">
                <div>
                  <h3 className="font-semibold">Welcome, {auth.user?.name}!</h3>
                  <p className="text-sm text-gray-600">{auth.user?.email}</p>
                </div>

                <div className="space-y-2">
                  <h4 className="font-medium">Update Profile</h4>
                  <div className="flex gap-2">
                    <Input
                      placeholder="New name"
                      value={newName}
                      onChange={(e) => setNewName(e.target.value)}
                    />
                    <Button onClick={handleUpdateProfile}>Update</Button>
                  </div>
                </div>

                <div className="flex gap-2">
                  <Button onClick={handleRefresh} variant="outline">
                    Refresh Token
                  </Button>
                  <Button onClick={handleLogout} variant="destructive">
                    Logout
                  </Button>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </main>
  );
}

export { AuthPage as default };
