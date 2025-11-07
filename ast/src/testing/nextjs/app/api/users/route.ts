export async function POST(request: Request) {
  const body = await request.json();
  return Response.json({ id: 1, name: body.name });
}
