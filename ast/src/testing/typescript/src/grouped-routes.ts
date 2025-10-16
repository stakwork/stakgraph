import { Router, Request, Response } from "express";
import { getPersonById, newPerson, deletePerson, PersonData } from "./service.js";

const personRouter = Router();

personRouter.get("/:id", async (req: Request, res: Response) => {
  const { id } = req.params;
  try {
    const person = (await getPersonById(Number(id))) as PersonData;
    if (!person) {
      return res.status(404).json({ error: "Person not found" });
    }
    return res.json(person);
  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: "Internal server error" });
  }
});

personRouter.post("/", async (req: Request, res: Response) => {
  const { name, email } = req.body;
  try {
    const person: PersonData = await newPerson({ name, email });
    return res.status(201).json(person);
  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: "Internal server error" });
  }
});

personRouter.delete("/:id", async (req: Request, res: Response) => {
  const { id } = req.params;
  try {
    await deletePerson(Number(id));
    return res.status(204).send();
  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: "Internal server error" });
  }
});

export function registerGroupedRoutes(app) {
  app.use("/items", personRouter);
}
