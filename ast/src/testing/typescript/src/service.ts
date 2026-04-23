import { SequelizePerson, TypeORMPerson } from "./model.js";
import { AppDataSource, prisma } from "./config.js";

// @ast node: Function "deprecated"
function deprecated(message: string) {
  return function (
    target: any,
    propertyKey: string,
    descriptor: PropertyDescriptor,
  ) {
    console.warn(`${propertyKey} is deprecated: ${message}`);
  };
}

// @ast node: Function "log"
function log(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
  const originalMethod = descriptor.value;
  descriptor.value = function (...args: any[]) {
    console.log(`Calling ${propertyKey} with`, args);
    return originalMethod.apply(this, args);
  };
}

// @ast node: DataModel "PersonData"
export interface PersonData {
  id?: number;
  name: string;
  email: string;
}

// @ast node: DataModel "IdType"
type IdType = number | string;

/** Interface for Person Service */
// @ast node: Trait "PersonService"
export interface PersonService {
  getById(id: IdType): Promise<PersonData | null>;
  create(personData: PersonData): Promise<PersonData>;
}

// @ast node: Function "getPersonById"
export async function getPersonById(id: IdType): Promise<PersonData | null> {
  const person = await SequelizePerson.findByPk(id);
  if (!person) {
    return null;
  }
  return person.toJSON() as PersonData;
}
// @ast node: Function "newPerson"
export async function newPerson(personData: PersonData): Promise<PersonData> {
  const person = await SequelizePerson.create(personData);
  return person.toJSON() as PersonData;
}
/** Service for managing people using Sequelize */
// @ast node: Class "SequelizePersonService"
// @ast edge: Implements -> Trait "PersonService" "service.ts"
export class SequelizePersonService implements PersonService {
  @log
  // @ast node: Function "getById"
  async getById(id: IdType): Promise<PersonData | null> {
    const person = await SequelizePerson.findByPk(id);
    if (!person) {
      return null;
    }
    return person.toJSON() as PersonData;
  }

  @log
  // @ast node: Function "create"
  async create(personData: PersonData): Promise<PersonData> {
    const person = await SequelizePerson.create(personData);
    return person.toJSON() as PersonData;
  }

  @deprecated("Use getById instead")
  // @ast node: Function "findPerson"
  async findPerson(id: IdType): Promise<PersonData | null> {
    return this.getById(id);
  }
}

// @ast node: Class "TypeOrmPersonService"
// @ast edge: Implements -> Trait "PersonService" "service.ts"
export class TypeOrmPersonService implements PersonService {
  private respository = AppDataSource.getRepository(TypeORMPerson);

  // @ast node: Function "getById"
  async getById(id: IdType): Promise<PersonData | null> {
    const person = await this.respository.findOneBy({ id });
    if (!person) {
      return null;
    }
    return person;
  }

  // @ast node: Function "create"
  async create(personData: PersonData): Promise<PersonData> {
    const person = this.respository.create(personData);
    await this.respository.save(person);
    return person;
  }
}

// @ast node: Class "PrismaPersonService"
// @ast edge: Implements -> Trait "PersonService" "service.ts"
export class PrismaPersonService implements PersonService {
  // @ast node: Function "getById"
  async getById(id: IdType): Promise<PersonData | null> {
    const person = await prisma.person.findUnique({
      where: { id },
    });
    if (!person) {
      return null;
    }
    return person;
  }

  // @ast node: Function "create"
  async create(personData: PersonData): Promise<PersonData> {
    const person = await prisma.person.create({
      data: personData,
    });
    return person;
  }
}
