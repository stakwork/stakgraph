import { NextRequest } from 'next/server';

export async function PUT(req: NextRequest) {
  const orderId = req.nextUrl.searchParams.get('orderId');
  const body = await req.json();
  return Response.json({ orderId, status: body.status });
}
