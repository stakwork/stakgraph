import { Request as ExpressRequest, Response as ExpressResponse } from 'express';

type McpHandler = (request: Request) => Promise<Response>;

export function createExpressAdapter(handler: McpHandler) {
  return async (req: ExpressRequest, res: ExpressResponse) => {
    try {
      const fetchRequest = expressToFetchRequest(req);
      const fetchResponse = await handler(fetchRequest);
      await sendFetchResponse(fetchResponse, res);
    } catch (error) {
      console.error('MCP handler error:', error);
      res.status(500).json({ error: 'Internal server error' });
    }
  };
}

function expressToFetchRequest(req: ExpressRequest): Request {
  const protocol = req.protocol;
  const host = req.get('host') || 'localhost';
  const url = `${protocol}://${host}${req.originalUrl}`;

  const headers = new Headers();
  for (const [key, value] of Object.entries(req.headers)) {
    if (value) {
      if (Array.isArray(value)) {
        value.forEach(v => headers.append(key, v));
      } else {
        headers.set(key, value);
      }
    }
  }

  return new Request(url, {
    method: req.method,
    headers,
    body: ['GET', 'HEAD'].includes(req.method) ? undefined : JSON.stringify(req.body),
  });
}

async function sendFetchResponse(fetchResponse: Response, res: ExpressResponse) {
  res.status(fetchResponse.status);
  fetchResponse.headers.forEach((value, key) => {
    res.setHeader(key, value);
  });

  const body = await fetchResponse.text();
  res.send(body);
}
