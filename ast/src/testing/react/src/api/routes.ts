import express from "express";

const router = express.Router();

// @ast node: Endpoint "/users" [verb=GET]
router.get("/users", async (req, res) => {
  const users = [{ id: 1, name: "John" }];
  res.json(users);
});

// @ast node: Endpoint "/users" [verb=POST]
router.post("/users", async (req, res) => {
  const { name, email } = req.body;
  const newUser = { id: Date.now(), name, email };
  res.status(201).json(newUser);
});

// @ast node: Endpoint "/users/:id" [verb=PUT]
router.put("/users/:id", async (req, res) => {
  const { id } = req.params;
  const { name, email } = req.body;
  res.json({ id, name, email });
});

// @ast node: Endpoint "/users/:id" [verb=DELETE]
router.delete("/users/:id", async (req, res) => {
  const { id } = req.params;
  res.json({ message: `User ${id} deleted` });
});

// PATCH endpoint
router.patch("/users/:id", async (req, res) => {
  const { id } = req.params;
  res.json({ id, ...req.body });
});

// Middleware usage
router.use("/admin", (req, res, next) => {
  // Auth middleware
  next();
});

export default router;
