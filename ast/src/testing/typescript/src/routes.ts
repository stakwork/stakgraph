import express, { Request, Response } from "express";
import { getPersonById, newPerson, PersonData } from "./service.js";
import { peopleRouter as crossFilePeopleRouter } from "./people-routes.js";
import { adminRouter } from "./admin-routes.js";

type PersonRequest = Request<{}, {}, { name: string; email: string }>;
type PersonResponse = Response<PersonData | { error: string }>;

// @ast node: DataModel "ResponseStatus"
export enum ResponseStatus {
  SUCCESS = 200,
  CREATED = 201,
  NOT_FOUND = 404,
  INTERNAL_ERROR = 500,
}

const peopleRouter = express.Router();

peopleRouter.post("/new", createNewPerson);
peopleRouter.get("/recent", getRecentPeople);

// @ast node: Function "registerRoutes"
export function registerRoutes(app) {
  app.get("/person/:id", getPerson);

  app.post("/person", createPerson);

  app.use("/people", peopleRouter);

  app.use("/api/people", crossFilePeopleRouter);
  app.use("/api/admin", adminRouter);
}

// @ast node: Endpoint "/person/:id" [verb=GET]
// @ast node: Function "getPerson"
// @ast edge: Handler <- Endpoint "/person/:id" "routes.ts" [verb=GET]
async function getPerson(req: Request, res: Response) {
  const { id } = req.params;

  try {
    const person = (await getPersonById(Number(id))) as PersonData;
    if (!person) {
      return res
        .status(ResponseStatus.NOT_FOUND)
        .json({ error: "Person not found" });
    }
    return res.json(person);
  } catch (error) {
    console.error(error);
    return res
      .status(ResponseStatus.INTERNAL_ERROR)
      .json({ error: "Internal server error" });
  }
}

/**
 * Create a new person.
 */
// @ast node: Endpoint "/person" [verb=POST]
// @ast node: Function "createPerson"
// @ast edge: Handler <- Endpoint "/person" "routes.ts" [verb=POST]
async function createPerson(req: PersonRequest, res: PersonResponse) {
  const { name, email } = req.body;
  try {
    const person: PersonData = await newPerson({ name, email });
    return res.status(ResponseStatus.CREATED).json(person);
  } catch (error) {
    console.error(error);
    return res
      .status(ResponseStatus.INTERNAL_ERROR)
      .json({ error: "Internal server error" });
  }
}

// @ast node: Endpoint "/people/new" [verb=POST]
// @ast node: Function "createNewPerson"
// @ast edge: Handler <- Endpoint "/people/new" "routes.ts" [verb=POST]
async function createNewPerson(req: PersonRequest, res: PersonResponse) {
  const { name, email } = req.body;
  try {
    const person: PersonData = await newPerson({ name, email });
    return res.status(ResponseStatus.CREATED).json(person);
  } catch (error) {
    console.error(error);
    return res
      .status(ResponseStatus.INTERNAL_ERROR)
      .json({ error: "Internal server error" });
  }
}

// @ast node: Endpoint "/people/recent" [verb=GET]
// @ast node: Function "getRecentPeople"
// @ast edge: Handler <- Endpoint "/people/recent" "routes.ts" [verb=GET]
async function getRecentPeople(req: Request, res: Response) {
  try {
    return res.json([]);
  } catch (error) {
    console.error(error);
    return res
      .status(ResponseStatus.INTERNAL_ERROR)
      .json({ error: "Internal server error" });
  }
}
// @ast node: Request "/person/:id"
// @ast node: Request "/person"
// @ast node: DataModel "PersonRequest"
// @ast node: DataModel "PersonResponse"
