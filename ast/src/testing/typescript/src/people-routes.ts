import express, { Request, Response } from "express";

export const peopleRouter = express.Router();

peopleRouter.get("/search", searchPeople);
peopleRouter.get("/list", listPeople);

// @ast node: Endpoint "/api/people/search" [verb=GET]
// @ast node: Function "searchPeople"
// @ast edge: Handler <- Endpoint "/api/people/search" "people-routes.ts" [verb=GET]
async function searchPeople(req: Request, res: Response) {
  const { query } = req.query;
  try {
    return res.json({ results: [], query });
  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: "Internal server error" });
  }
}

// @ast node: Endpoint "/api/people/list" [verb=GET]
// @ast node: Function "listPeople"
// @ast edge: Handler <- Endpoint "/api/people/list" "people-routes.ts" [verb=GET]
async function listPeople(req: Request, res: Response) {
  try {
    return res.json({ people: [] });
  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: "Internal server error" });
  }
}
