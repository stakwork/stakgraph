"use client";
import { useState, useEffect } from "react";
import {
  Card,
  CardHeader,
  CardContent,
  CardTitle,
} from "../../components/ui/card";
import { Input } from "../../components/ui/input";
import { Button } from "../../components/ui/button";

interface Person {
  id?: string;
  name: string;
  age: number;
  email: string;
}

export default function PersonPage() {
  const [people, setPeople] = useState<Person[]>([]);
  const [form, setForm] = useState<Person>({ name: "", age: 0, email: "" });
  const [loading, setLoading] = useState(false);
  const [searchId, setSearchId] = useState("");
  const [foundPerson, setFoundPerson] = useState<Person | null>(null);
  const [searchError, setSearchError] = useState("");

  // Fetch people
  useEffect(() => {
    fetch("/api/person")
      .then((res) => res.json())
      .then(setPeople);
  }, []);

  // Handle form submit
  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    await fetch("/api/person", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(form),
    });
    setForm({ name: "", age: 0, email: "" });
    // Refresh list
    fetch("/api/person")
      .then((res) => res.json())
      .then(setPeople)
      .finally(() => setLoading(false));
  }

  // Handle finding a single person by ID
  async function handleFindPerson(e: React.FormEvent) {
    e.preventDefault();
    if (!searchId.trim()) return;

    setFoundPerson(null);
    setSearchError("");

    try {
      const response = await fetch(`/api/person/${searchId}`);
      if (response.ok) {
        const person = await response.json();
        setFoundPerson(person);
      } else {
        setSearchError("Person not found");
      }
    } catch (error) {
      setSearchError("Error fetching person");
    }
  }

  // Handle deleting a person by ID
  async function handleDeletePerson(id: string) {
    try {
      const response = await fetch(`/api/person/${id}`, {
        method: "DELETE",
      });
      if (response.ok) {
        // Refresh the people list
        fetch("/api/person")
          .then((res) => res.json())
          .then(setPeople);
      }
    } catch (error) {
      console.error("Error deleting person:", error);
    }
  }

  return (
    <main className="max-w-xl mx-auto py-8">
      <Card>
        <CardHeader>
          <CardTitle>Add Person</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <Input
              placeholder="Name"
              value={form.name}
              onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
              required
            />
            <Input
              placeholder="Age"
              type="number"
              value={form.age || ""}
              onChange={(e) =>
                setForm((f) => ({ ...f, age: Number(e.target.value) }))
              }
              required
            />
            <Input
              placeholder="Email"
              type="email"
              value={form.email}
              onChange={(e) =>
                setForm((f) => ({ ...f, email: e.target.value }))
              }
              required
            />
            <Button type="submit" disabled={loading}>
              {loading ? "Adding..." : "Add Person"}
            </Button>
          </form>
        </CardContent>
      </Card>

      <Card className="mt-8">
        <CardHeader>
          <CardTitle>Find Person by ID</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleFindPerson} className="flex flex-col gap-4">
            <div className="flex gap-2">
              <Input
                placeholder="Enter person ID (1, 2, 3...)"
                value={searchId}
                onChange={(e) => setSearchId(e.target.value)}
              />
              <Button type="submit">Find</Button>
            </div>
            {searchError && (
              <div className="text-red-500 text-sm">{searchError}</div>
            )}
            {foundPerson && (
              <div className="mt-4 p-4 bg-secondary rounded-md">
                <p className="font-semibold">Found: {foundPerson.name}</p>
                <p>Age: {foundPerson.age}</p>
                <p>Email: {foundPerson.email}</p>
                <Button
                  onClick={() => handleDeletePerson(foundPerson.id!)}
                  className="mt-2"
                  variant="destructive"
                  size="sm"
                >
                  Delete Person
                </Button>
              </div>
            )}
          </form>
        </CardContent>
      </Card>

      <div className="mt-8">
        <Card>
          <CardHeader>
            <CardTitle>People</CardTitle>
          </CardHeader>
          <CardContent>
            <ul>
              {people.map((p, i) => (
                <li key={i} className="flex justify-between items-center py-2">
                  <span>
                    {p.name} ({p.age}) - {p.email}
                  </span>
                  {p.id && (
                    <Button
                      onClick={() => handleDeletePerson(p.id!)}
                      variant="destructive"
                      size="sm"
                    >
                      Delete
                    </Button>
                  )}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      </div>
    </main>
  );
}
