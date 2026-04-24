// @ast node: Function "POST"
// @ast edge: Contains <- File "route.ts" "src/testing/nextjs/app/api/users/route.ts"
// @ast edge: Handler <- Endpoint "/api/users" "src/testing/nextjs/app/api/users/route.ts"
export async function POST(request: Request) {
  const body = await request.json();
  return Response.json({ id: 1, name: body.name });
}
// @ast node: Endpoint "/api/users" [verb=POST]
