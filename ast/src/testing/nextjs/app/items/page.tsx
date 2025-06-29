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

interface Item {
  id: number;
  title: string;
  description?: string;
  price: number;
}

export default function ItemsPage() {
  const [items, setItems] = useState<Item[]>([]);
  const [form, setForm] = useState<Omit<Item, "id">>({
    title: "",
    description: "",
    price: 0,
  });
  const [loading, setLoading] = useState(false);

  // Fetch items
  useEffect(() => {
    fetch("/api/items")
      .then((res) => res.json())
      .then(setItems);
  }, []);

  // Handle form submit
  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    await fetch("/api/items", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(form),
    });
    setForm({ title: "", description: "", price: 0 });
    // Refresh list
    fetch("/api/items")
      .then((res) => res.json())
      .then(setItems)
      .finally(() => setLoading(false));
  }

  return (
    <main className="max-w-xl mx-auto py-8">
      <Card>
        <CardHeader>
          <CardTitle>Add Item</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <Input
              placeholder="Title"
              value={form.title}
              onChange={(e) =>
                setForm((f) => ({ ...f, title: e.target.value }))
              }
              required
            />
            <Input
              placeholder="Description"
              value={form.description}
              onChange={(e) =>
                setForm((f) => ({ ...f, description: e.target.value }))
              }
            />
            <Input
              placeholder="Price"
              type="number"
              value={form.price || ""}
              onChange={(e) =>
                setForm((f) => ({ ...f, price: Number(e.target.value) }))
              }
              required
            />
            <Button type="submit" disabled={loading}>
              {loading ? "Adding..." : "Add Item"}
            </Button>
          </form>
        </CardContent>
      </Card>
      <div className="mt-8">
        <Card>
          <CardHeader>
            <CardTitle>Items</CardTitle>
          </CardHeader>
          <CardContent>
            <ul>
              {items.map((item) => (
                <li key={item.id}>
                  {item.title} - ${item.price}{" "}
                  {item.description && `(${item.description})`}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      </div>
    </main>
  );
}
