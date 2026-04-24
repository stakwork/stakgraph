// @ast node: Function "POST"
// @ast edge: Contains <- File "route.ts" "app/api/sessions/route.ts"
// @ast edge: Handler <- Endpoint "/api/sessions" "app/api/sessions/route.ts"
export async function POST() {
  return Response.json({ id: "sess_123" }, { status: 201 });
}
// @ast node: Endpoint "/api/sessions" [verb=POST]
