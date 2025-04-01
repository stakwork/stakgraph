import { Controller, Get, Post, Param, Body } from "@nestjs/common";
import { PersonService } from "./person.service.js";
import { PersonData } from "../../shared/service.js";

export class PersonController {
  constructor(private readonly personService: PersonService) {}

  @Get("person/:id")
  async getPerson(@Param("id") id: number): Promise<PersonData | null> {
    return this.personService.getById(id);
  }

  @Post("person")
  async createPerson(@Body() personData: PersonData): Promise<PersonData> {
    return this.personService.create(personData);
  }
}
