import express, { Request, Response } from "express";

export const adminRouter = express.Router();

adminRouter.get("/users", listUsers);
adminRouter.delete("/users/:id", deleteUser);

// @ast node: Endpoint "/api/admin/users" [verb=GET]
// @ast node: Function "listUsers"
// @ast edge: Handler <- Endpoint "/api/admin/users" "admin-routes.ts" [verb=GET]
async function listUsers(req: Request, res: Response) {
  try {
    return res.json({ users: [] });
  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: "Internal server error" });
  }
}

// @ast node: Endpoint "/api/admin/users/:id" [verb=DELETE]
// @ast node: Function "deleteUser"
// @ast edge: Handler <- Endpoint "/api/admin/users/:id" "admin-routes.ts" [verb=DELETE]
async function deleteUser(req: Request, res: Response) {
  const { id } = req.params;
  try {
    return res.json({ deleted: id });
  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: "Internal server error" });
  }
}
