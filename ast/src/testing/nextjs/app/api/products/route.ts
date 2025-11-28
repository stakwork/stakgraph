export async function GET() {
  return Response.json([{ id: 1, name: 'Widget' }]);
}

export async function POST(request: Request) {
  const body = await request.json();
  return Response.json({ id: 1, ...body }, { status: 201 });
}
