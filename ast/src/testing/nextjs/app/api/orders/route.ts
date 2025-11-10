export async function PUT(request: Request) {
  const body = await request.json();
  return Response.json({ orderId: 1, status: body.status });
}
