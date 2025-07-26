import DataTypes, { Model } from "sequelize";
import { Entity, Column, PrimaryGeneratedColumn } from "typeorm";
import { sequelize } from "./config.js";

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

SequelizePerson.init(
  {
    id: {
      type: DataTypes.INTEGER,
      autoIncrement: true,
      primaryKey: true,
    },
    name: {
      type: DataTypes.STRING,
      allowNull: false,
    },
    email: {
      type: DataTypes.STRING,
      allowNull: false,
    },
  },
  {
    sequelize,
    tableName: "people",
  }
);

@Entity("persons")
export class TypeORMPerson {
  @PrimaryGeneratedColumn()
  id!: number;

  @Column()
  name!: string;

  @Column({ unique: true })
  email!: string;
}

// Trait but is not implemented on a class || Unfortunately get's picked up as a data model as well. Till we find a way to filter it out... ideally treesitter queries
type CarType = {
  model: string;
  year: number;
  startEngine(): string;
};

//should not be DM
// interface TestInterface {
//   operand1: number;
//   operand2: number;
//   add(): number;
//   subtract?(): number;
// }

// TO FIGURE OUT QUERY THAT DOES NOT PICK IT UP AS DM
