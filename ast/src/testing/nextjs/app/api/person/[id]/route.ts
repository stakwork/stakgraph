import { NextResponse } from "next/server";

const people = [
  { id: "1", name: "Alice", age: 30, email: "alice@example.com" },
  { id: "2", name: "Bob", age: 25, email: "bob@example.com" },
  { id: "3", name: "Charlie", age: 35, email: "charlie@example.com" },
];

export async function GET(
  request: Request,
  { params }: { params: { id: string } }
) {
  const person = people.find((p) => p.id === params.id);

  if (!person) {
    return new Response("Person not found", { status: 404 });
  }

  return NextResponse.json(person);
}

export async function DELETE(
  request: Request,
  { params }: { params: { id: string } }
) {
  const index = people.findIndex((p) => p.id === params.id);

  if (index === -1) {
    return new Response("Person not found", { status: 404 });
  }

  people.splice(index, 1);
  return NextResponse.json({ success: true });
}
