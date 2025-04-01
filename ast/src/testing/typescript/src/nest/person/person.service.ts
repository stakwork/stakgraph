import { Injectable } from "@nestjs/common";
import { SequelizePersonService } from "../../shared/service.js";
import { PersonData } from "../../shared/service.js";

@Injectable()
export class PersonService {
  private sequelizeService = new SequelizePersonService();

  async getById(id: number): Promise<PersonData | null> {
    return this.sequelizeService.getById(id);
  }

  async create(personData: PersonData): Promise<PersonData> {
    return this.sequelizeService.create(personData);
  }
}
