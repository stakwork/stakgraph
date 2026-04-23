import { Request, Response, NextFunction } from 'express';

// @ast node: Function "authenticate"
export function authenticate(req: Request, res: Response, next: NextFunction) {
  const token = req.headers.authorization?.split(' ')[1];
  if (!token) {
    return res.status(401).json({ error: 'Authentication required' });
  }
  // In a real app, verify the token
  next();
}

// @ast node: Function "validateRequest"
export function validateRequest(req: Request, res: Response, next: NextFunction) {
  if (!req.body || Object.keys(req.body).length === 0) {
    return res.status(400).json({ error: 'Request body is required' });
  }
  next();
}

// @ast node: Function "errorHandler"
export function errorHandler(err: Error, req: Request, res: Response, next: NextFunction) {
  console.error(err);
  res.status(500).json({ error: 'Internal server error' });
}