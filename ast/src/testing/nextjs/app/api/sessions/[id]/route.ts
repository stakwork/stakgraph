// @ast node: Function "DELETE"
// @ast edge: Contains <- File "route.ts" "app/api/sessions/[id]/route.ts"
// @ast edge: Handler <- Endpoint "/api/sessions/[id]" "app/api/sessions/[id]/route.ts"
export async function DELETE() {
  return Response.json({ success: true });
}
// @ast node: Endpoint "/api/sessions/[id]" [verb=DELETE]
