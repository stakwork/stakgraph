export async function PUT(request: Request) {
  const body = await request.json();
  return Response.json({ id: 1, ...body });
}
