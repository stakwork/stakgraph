import { NextRequest, NextResponse } from "next/server";

interface User {
  id: number;
  name: string;
  email: string;
}

const users: User[] = [];

// @ast node: Function "GET"
export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const id = searchParams.get("id");

  if (id) {
    const user = users.find((u) => u.id === parseInt(id));
    return NextResponse.json(user || { error: "Not found" });
  }

  return NextResponse.json(users);
}

// @ast node: Function "POST"
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

// @ast node: Function "PUT"
export async function PUT(request: NextRequest) {
  const body = await request.json();
  const index = users.findIndex((u) => u.id === body.id);
  if (index !== -1) {
    users[index] = body;
    return NextResponse.json(body);
  }
  return NextResponse.json({ error: "Not found" }, { status: 404 });
}

// @ast node: Function "DELETE"
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

// @ast node: Function "PATCH"
export async function PATCH(request: NextRequest) {
  const body = await request.json();
  const index = users.findIndex((u) => u.id === body.id);
  if (index !== -1) {
    users[index] = { ...users[index], ...body };
    return NextResponse.json(users[index]);
  }
  return NextResponse.json({ error: "Not found" }, { status: 404 });
}
// @ast node: Endpoint "/api/users.ts" [verb=GET]
// @ast node: Endpoint "/api/users.ts" [verb=POST]
// @ast node: Endpoint "/api/users.ts" [verb=PUT]
// @ast node: Endpoint "/api/users.ts" [verb=DELETE]
// @ast node: Endpoint "/api/users.ts" [verb=PATCH]
// @ast node: Request "id"
// @ast node: Request "id"
// @ast node: Var "users"
// @ast node: DataModel "User"
