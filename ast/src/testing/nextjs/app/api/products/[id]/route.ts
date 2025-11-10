import { NextRequest } from 'next/server';

export async function GET(req: NextRequest) {
  const id = req.nextUrl.searchParams.get('id');
  return Response.json({ id, name: 'Product A' });
}
