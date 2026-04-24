import { NextResponse } from "next/server";

const people = [{ name: "Alice", age: 30, email: "alice@example.com" }];

// @ast node: Function "GET"
// @ast edge: Contains <- File "route.ts" "src/testing/nextjs/app/api/person/route.ts"
// @ast edge: Handler <- Endpoint "/api/person" "src/testing/nextjs/app/api/person/route.ts" [verb=GET]
export async function GET() {
  return NextResponse.json(people);
}

// @ast node: Function "POST"
// @ast edge: Contains <- File "route.ts" "src/testing/nextjs/app/api/person/route.ts"
// @ast edge: Handler <- Endpoint "/api/person" "src/testing/nextjs/app/api/person/route.ts" [verb=POST]
export async function POST(request: Request) {
  const body = await request.json();
  people.push(body);
  return NextResponse.json(body, { status: 201 });
}
// @ast node: Endpoint "/api/person" [verb=GET]
// @ast node: Endpoint "/api/person" [verb=POST]
// @ast node: Var "people"
