import { Router, Request, Response } from "express";

const router = Router();

// GET / - should become /api/settings/ after grouping
router.get("/", async (req: Request, res: Response) => {
  res.json({ settings: [] });
});

// GET /:id - should become /api/settings/:id after grouping
router.get("/:id", async (req: Request, res: Response) => {
  res.json({ setting: req.params.id });
});

// POST / - should become /api/settings after grouping
router.post("/", async (req: Request, res: Response) => {
  res.status(201).json({ created: true });
});

export default router;
