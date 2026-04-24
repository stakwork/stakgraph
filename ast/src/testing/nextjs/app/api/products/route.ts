// @ast node: Function "GET"
// @ast edge: Contains <- File "route.ts" "src/testing/nextjs/app/api/products/route.ts"
// @ast edge: Handler <- Endpoint "/api/products" "src/testing/nextjs/app/api/products/route.ts" [verb=GET]
export async function GET() {
  return Response.json([{ id: 1, name: "Widget" }]);
}

// @ast node: Function "POST"
// @ast edge: Contains <- File "route.ts" "src/testing/nextjs/app/api/products/route.ts"
// @ast edge: Handler <- Endpoint "/api/products" "src/testing/nextjs/app/api/products/route.ts" [verb=POST]
export async function POST(request: Request) {
  const body = await request.json();
  return Response.json({ id: 1, ...body }, { status: 201 });
}
// @ast node: Endpoint "/api/products" [verb=GET]
// @ast node: Endpoint "/api/products" [verb=POST]
