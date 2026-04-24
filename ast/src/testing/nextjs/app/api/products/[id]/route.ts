import { NextRequest } from 'next/server';

// @ast node: Function "GET"
// @ast edge: Contains <- File "route.ts" "src/testing/nextjs/app/api/products/[id]/route.ts"
// @ast edge: Handler <- Endpoint "/api/products/[id]" "src/testing/nextjs/app/api/products/[id]/route.ts"
export async function GET(req: NextRequest) {
  const id = req.nextUrl.searchParams.get('id');
  return Response.json({ id, name: 'Product A' });
}
// @ast node: Endpoint "/api/products/[id]" [verb=GET]
