"use client";
import { useState } from "react";
import { useUserQuery } from "../../lib/hooks/useUserQuery";
import {
  Card,
  CardHeader,
  CardContent,
  CardTitle,
} from "../../components/ui/card";
import { Button } from "../../components/ui/button";
import { Input } from "../../components/ui/input";

function QueryDemo() {
  const [userId, setUserId] = useState("user-123");
  const [inputValue, setInputValue] = useState("user-123");
  const query = useUserQuery(userId);

  const handleRefetch = async () => {
    await query.refetch();
  };

  const handleInvalidate = () => {
    query.invalidate();
  };

  const handleReset = () => {
    query.reset();
  };

  const handleChangeUser = () => {
    setUserId(inputValue);
  };

  return (
    <main className="max-w-2xl mx-auto py-8">
      <Card>
        <CardHeader>
          <CardTitle>User Query Demo</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <div>
              <h4 className="font-medium mb-2">Change User</h4>
              <div className="flex gap-2">
                <Input
                  placeholder="User ID"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                />
                <Button onClick={handleChangeUser}>Load User</Button>
              </div>
            </div>

            <div>
              <h4 className="font-medium mb-2">Query Actions</h4>
              <div className="flex gap-2">
                <Button onClick={handleRefetch} disabled={query.isLoading}>
                  {query.isLoading ? "Refetching..." : "Refetch"}
                </Button>
                <Button onClick={handleInvalidate} variant="outline">
                  Invalidate
                </Button>
                <Button onClick={handleReset} variant="outline">
                  Reset
                </Button>
              </div>
            </div>

            <div>
              <h4 className="font-medium mb-2">Query State</h4>
              <div className="space-y-2 text-sm">
                <p>
                  <strong>Status:</strong>{" "}
                  {query.isLoading ? (
                    <span className="text-blue-600">Loading...</span>
                  ) : query.error ? (
                    <span className="text-red-600">Error</span>
                  ) : query.data ? (
                    <span className="text-green-600">Success</span>
                  ) : (
                    <span className="text-gray-500">No Data</span>
                  )}
                </p>

                {query.error && (
                  <p className="text-red-600">
                    <strong>Error:</strong> {query.error}
                  </p>
                )}

                {query.data && (
                  <div className="p-4 bg-gray-50 rounded">
                    <p>
                      <strong>ID:</strong> {query.data.id}
                    </p>
                    <p>
                      <strong>Name:</strong> {query.data.name}
                    </p>
                    <p>
                      <strong>Email:</strong> {query.data.email}
                    </p>
                  </div>
                )}
              </div>
            </div>

            <div className="text-xs text-gray-500">
              <p>
                This demonstrates query patterns similar to React Query/TanStack
                Query where methods like <code>refetch()</code>,{" "}
                <code>invalidate()</code>, and <code>reset()</code> are called
                on the query object.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </main>
  );
}

export { QueryDemo as default };
