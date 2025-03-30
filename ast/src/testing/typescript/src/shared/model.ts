import { Model } from "sequelize";
import { Entity, Column, PrimaryGeneratedColumn } from "typeorm";

interface PersonAttributes {
  id?: number;
  name: string;
  email: string;
}

export class SequelizePerson
  extends Model<PersonAttributes>
  implements PersonAttributes
{
  public id!: number;
  public name!: string;
  public email!: string;
}

@Entity("persons")
export class TypeORMPerson {
  @PrimaryGeneratedColumn()
  id!: number;

  @Column()
  name!: string;

  @Column({ unique: true })
  email!: string;
}
