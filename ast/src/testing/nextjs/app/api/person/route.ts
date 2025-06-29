import { NextResponse } from "next/server";

const people = [{ name: "Alice", age: 30, email: "alice@example.com" }];

export async function GET() {
  return NextResponse.json(people);
}

export async function POST(request: Request) {
  const body = await request.json();
  people.push(body);
  return NextResponse.json(body, { status: 201 });
}
