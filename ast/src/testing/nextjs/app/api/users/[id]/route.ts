// @ast node: Function "PUT"
// @ast edge: Contains <- File "route.ts" "app/api/users/[id]/route.ts"
// @ast edge: Handler <- Endpoint "/api/users/[id]" "app/api/users/[id]/route.ts" [verb=PUT]
export async function PUT(request: Request) {
  const body = await request.json();
  return Response.json({ id: 1, ...body });
}

// @ast node: Function "DELETE"
// @ast edge: Contains <- File "route.ts" "app/api/users/[id]/route.ts"
// @ast edge: Handler <- Endpoint "/api/users/[id]" "app/api/users/[id]/route.ts" [verb=DELETE]
export async function DELETE() {
  return Response.json({ success: true });
}
// @ast node: Endpoint "/api/users/[id]" [verb=PUT]
// @ast node: Endpoint "/api/users/[id]" [verb=DELETE]
