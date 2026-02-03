import { SequelizePerson, TypeORMPerson } from "./model.js";
import { AppDataSource, prisma } from "./config.js";

function deprecated(message: string) {
  return function (
    target: any,
    propertyKey: string,
    descriptor: PropertyDescriptor,
  ) {
    console.warn(`${propertyKey} is deprecated: ${message}`);
  };
}

function log(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
  const originalMethod = descriptor.value;
  descriptor.value = function (...args: any[]) {
    console.log(`Calling ${propertyKey} with`, args);
    return originalMethod.apply(this, args);
  };
}

export interface PersonData {
  id?: number;
  name: string;
  email: string;
}

type IdType = number | string;

/** Interface for Person Service */
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
/** Service for managing people using Sequelize */
export class SequelizePersonService implements PersonService {
  @log
  async getById(id: IdType): Promise<PersonData | null> {
    const person = await SequelizePerson.findByPk(id);
    if (!person) {
      return null;
    }
    return person.toJSON() as PersonData;
  }

  @log
  async create(personData: PersonData): Promise<PersonData> {
    const person = await SequelizePerson.create(personData);
    return person.toJSON() as PersonData;
  }

  @deprecated("Use getById instead")
  async findPerson(id: IdType): Promise<PersonData | null> {
    return this.getById(id);
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
