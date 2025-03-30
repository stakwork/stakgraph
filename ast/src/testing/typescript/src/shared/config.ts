import { Sequelize } from "sequelize";
import { DataSource } from "typeorm";
import { PrismaClient } from "@prisma/client";
import { TypeORMPerson, SequelizePerson } from "./model.js";
import { DataTypes } from "sequelize";

export const sequelize = new Sequelize({
  dialect: "sqlite",
  storage: "./database.sqlite",
  logging: false,
});

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

export const AppDataSource = new DataSource({
  type: "sqlite",
  database: "./database.sqlite",
  entities: [TypeORMPerson],
  synchronize: true,
  logging: false,
});

export const prisma = new PrismaClient();
