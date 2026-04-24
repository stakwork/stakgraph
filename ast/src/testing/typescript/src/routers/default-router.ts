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
// @ast node: Function "get_handler_L5"
// @ast node: Function "get_param_id_handler_L10"
// @ast node: Function "post_handler_L15"
// @ast node: Endpoint "/api/settings/" [verb=GET]
// @ast node: Endpoint "/api/settings/:id" [verb=GET]
// @ast node: Endpoint "/api/settings/" [verb=POST]
