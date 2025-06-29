import { NextResponse } from "next/server";

const items = [
  { id: 1, title: "Sample Item", description: "A demo item", price: 10 },
];

export async function GET() {
  return NextResponse.json(items);
}

export async function POST(request: Request) {
  const body = await request.json();
  const newItem = {
    id: items.length + 1,
    ...body,
  };
  items.push(newItem);
  return NextResponse.json(newItem, { status: 201 });
}
