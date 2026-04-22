// @ast node: Function "DELETE"
// @ast edge: Contains <- File "route.ts" "src/testing/nextjs/app/api/sessions/[id]/route.ts"
// @ast edge: Handler <- Endpoint "/api/sessions/[id]" "src/testing/nextjs/app/api/sessions/[id]/route.ts"
export async function DELETE() {
  return Response.json({ success: true });
}
