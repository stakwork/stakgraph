import { NextResponse } from "next/server";

const items = [
  { id: 1, title: "Sample Item", description: "A demo item", price: 10 },
];

// @ast node: Function "GET"
// @ast edge: Contains <- File "route.ts" "app/api/items/route.ts"
// @ast edge: Handler <- Endpoint "/api/items" "app/api/items/route.ts"
export async function GET() {
  return NextResponse.json(items);
}

// @ast node: Function "POST"
// @ast edge: Contains <- File "route.ts" "app/api/items/route.ts"
// @ast edge: Handler <- Endpoint "/api/items" "app/api/items/route.ts"
export async function POST(request: Request) {
  const body = await request.json();
  const newItem = {
    id: items.length + 1,
    ...body,
  };
  items.push(newItem);
  return NextResponse.json(newItem, { status: 201 });
}
// @ast node: Endpoint "/api/items" [verb=GET]
// @ast node: Endpoint "/api/items" [verb=POST]
// @ast node: Var "items"
