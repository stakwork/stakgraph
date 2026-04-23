// @ast node: Function "GET"
// @ast edge: Contains <- File "route.ts" "src/testing/nextjs/app/api/categories/[id]/route.ts"
// @ast edge: Handler <- Endpoint "/api/categories/[id]" "src/testing/nextjs/app/api/categories/[id]/route.ts"
export async function GET() {
  return Response.json({ id: 1, name: "Electronics" });
}
