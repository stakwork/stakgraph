// @ast node: Function "PUT"
// @ast edge: Contains <- File "route.ts" "src/testing/nextjs/app/api/reviews/[id]/route.ts"
// @ast edge: Handler <- Endpoint "/api/reviews/[id]" "src/testing/nextjs/app/api/reviews/[id]/route.ts"
export async function PUT(request: Request) {
  const body = await request.json();
  return Response.json({ id: 1, ...body });
}
