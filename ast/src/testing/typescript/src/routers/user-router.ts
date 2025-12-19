import express from 'express';
import { authenticate, validateRequest } from '../middleware/auth';
import { UserService } from '../services/user-service';

const router = express.Router();
const userService = new UserService();

// Middleware applied to all routes in this router
router.use(authenticate);

// Basic route with handler
router.get('/', async (req, res) => {
  const users = await userService.findAll();
  res.json(users);
});

// Route with path parameter
router.get('/:id', async (req, res) => {
  const user = await userService.findById(req.params.id);
  res.json(user);
});

// POST with request body validation middleware
router.post('/', validateRequest, async (req, res) => {
  const newUser = await userService.create(req.body);
  res.status(201).json(newUser);
});

// PUT with multiple middlewares
router.put('/:id', authenticate, validateRequest, async (req, res) => {
  const updated = await userService.update(req.params.id, req.body);
  res.json(updated);
});

// DELETE route
router.delete('/:id', async (req, res) => {
  await userService.delete(req.params.id);
  res.status(204).end();
});

export default router;