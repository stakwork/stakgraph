import express from "express";
import { Request, Response } from "express";
import { getPersonById, newPerson, PersonData } from "../shared/service.js";

export function registerRoutes(app: any) {
  app.use(express.urlencoded({ extended: true }));
  app.get("/person/:id", getPerson);

  app.post("/person", createPerson);
}

async function getPerson(req: Request, res: Response) {
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
}

async function createPerson(req: Request, res: Response) {
  const { name, email } = req.body;
  try {
    const person: PersonData = await newPerson({ name, email });
    return res.status(201).json(person);
  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: "Internal server error" });
  }
}
