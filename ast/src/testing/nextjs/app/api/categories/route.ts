// @ast node: Function "POST"
// @ast edge: Contains <- File "route.ts" "app/api/categories/route.ts"
// @ast edge: Handler <- Endpoint "/api/categories" "app/api/categories/route.ts"
export async function POST(request: Request) {
  const body = await request.json();
  return Response.json({ id: 1, ...body }, { status: 201 });
}
// @ast node: Endpoint "/api/categories" [verb=POST]
