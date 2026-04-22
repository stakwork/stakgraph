// @ast node: Function "POST"
// @ast edge: Contains <- File "route.ts" "src/testing/nextjs/app/api/categories/route.ts"
// @ast edge: Handler <- Endpoint "/api/categories" "src/testing/nextjs/app/api/categories/route.ts"
export async function POST(request: Request) {
  const body = await request.json();
  return Response.json({ id: 1, ...body }, { status: 201 });
}
