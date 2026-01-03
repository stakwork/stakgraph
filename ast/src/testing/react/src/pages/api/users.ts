import { NextRequest, NextResponse } from "next/server";

interface User {
  id: number;
  name: string;
  email: string;
}

const users: User[] = [];

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const id = searchParams.get("id");

  if (id) {
    const user = users.find((u) => u.id === parseInt(id));
    return NextResponse.json(user || { error: "Not found" });
  }

  return NextResponse.json(users);
}

export async function POST(request: NextRequest) {
  const body = await request.json();
  const newUser: User = {
    id: Date.now(),
    name: body.name,
    email: body.email,
  };
  users.push(newUser);
  return NextResponse.json(newUser, { status: 201 });
}

export async function PUT(request: NextRequest) {
  const body = await request.json();
  const index = users.findIndex((u) => u.id === body.id);
  if (index !== -1) {
    users[index] = body;
    return NextResponse.json(body);
  }
  return NextResponse.json({ error: "Not found" }, { status: 404 });
}

export async function DELETE(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const id = searchParams.get("id");

  if (!id) {
    return NextResponse.json({ error: "ID required" }, { status: 400 });
  }

  const index = users.findIndex((u) => u.id === parseInt(id));
  if (index !== -1) {
    users.splice(index, 1);
    return NextResponse.json({ message: "Deleted" });
  }
  return NextResponse.json({ error: "Not found" }, { status: 404 });
}

export async function PATCH(request: NextRequest) {
  const body = await request.json();
  const index = users.findIndex((u) => u.id === body.id);
  if (index !== -1) {
    users[index] = { ...users[index], ...body };
    return NextResponse.json(users[index]);
  }
  return NextResponse.json({ error: "Not found" }, { status: 404 });
}
