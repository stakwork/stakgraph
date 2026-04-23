// @ast node: Function "POST"
// @ast edge: Contains <- File "route.ts" "src/testing/nextjs/app/api/reviews/route.ts"
// @ast edge: Handler <- Endpoint "/api/reviews" "src/testing/nextjs/app/api/reviews/route.ts"
export async function POST(request: Request) {
  const body = await request.json();
  return Response.json({ id: 1, ...body }, { status: 201 });
}
