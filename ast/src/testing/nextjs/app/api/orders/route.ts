import { NextRequest } from 'next/server';

// @ast node: Function "PUT"
// @ast edge: Contains <- File "route.ts" "app/api/orders/route.ts"
// @ast edge: Handler <- Endpoint "/api/orders" "app/api/orders/route.ts"
export async function PUT(req: NextRequest) {
  const orderId = req.nextUrl.searchParams.get('orderId');
  const body = await req.json();
  return Response.json({ orderId, status: body.status });
}
// @ast node: Endpoint "/api/orders" [verb=PUT]
