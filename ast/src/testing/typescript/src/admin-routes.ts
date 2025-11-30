import express, { Request, Response } from "express";

export const adminRouter = express.Router();

adminRouter.get("/users", listUsers);
adminRouter.delete("/users/:id", deleteUser);

async function listUsers(req: Request, res: Response) {
  try {
    return res.json({ users: [] });
  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: "Internal server error" });
  }
}

async function deleteUser(req: Request, res: Response) {
  const { id } = req.params;
  try {
    return res.json({ deleted: id });
  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: "Internal server error" });
  }
}
