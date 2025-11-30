import express, { Request, Response } from "express";

export const peopleRouter = express.Router();

peopleRouter.get("/search", searchPeople);
peopleRouter.get("/list", listPeople);

async function searchPeople(req: Request, res: Response) {
  const { query } = req.query;
  try {
    return res.json({ results: [], query });
  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: "Internal server error" });
  }
}

async function listPeople(req: Request, res: Response) {
  try {
    return res.json({ people: [] });
  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: "Internal server error" });
  }
}
