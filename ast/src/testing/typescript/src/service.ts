import { SequelizePerson, TypeORMPerson } from "./model.js";
import { AppDataSource, prisma } from "./config.js";
export interface PersonData {
  id?: number;
  name: string;
  email: string;
}

type IdType = number | string;

export interface PersonService {
  getById(id: IdType): Promise<PersonData | null>;
  create(personData: PersonData): Promise<PersonData>;
}

export async function getPersonById(id: IdType): Promise<PersonData | null> {
  const person = await SequelizePerson.findByPk(id);
  if (!person) {
    return null;
  }
  return person.toJSON() as PersonData;
}
export async function newPerson(personData: PersonData): Promise<PersonData> {
  const person = await SequelizePerson.create(personData);
  return person.toJSON() as PersonData;
}
export class SequelizePersonService implements PersonService {
  async getById(id: IdType): Promise<PersonData | null> {
    const person = await SequelizePerson.findByPk(id);
    if (!person) {
      return null;
    }
    return person.toJSON() as PersonData;
  }
  async create(personData: PersonData): Promise<PersonData> {
    const person = await SequelizePerson.create(personData);
    return person.toJSON() as PersonData;
  }
}

export class TypeOrmPersonService implements PersonService {
  private respository = AppDataSource.getRepository(TypeORMPerson);

  async getById(id: IdType): Promise<PersonData | null> {
    const person = await this.respository.findOneBy({ id });
    if (!person) {
      return null;
    }
    return person;
  }

  async create(personData: PersonData): Promise<PersonData> {
    const person = this.respository.create(personData);
    await this.respository.save(person);
    return person;
  }
}

export class PrismaPersonService implements PersonService {
  async getById(id: IdType): Promise<PersonData | null> {
    const person = await prisma.person.findUnique({
      where: { id },
    });
    if (!person) {
      return null;
    }
    return person;
  }

  async create(personData: PersonData): Promise<PersonData> {
    const person = await prisma.person.create({
      data: personData,
    });
    return person;
  }
}
