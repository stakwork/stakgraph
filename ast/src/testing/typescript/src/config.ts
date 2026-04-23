import { Sequelize } from "sequelize";
import { DataSource } from "typeorm";
import { PrismaClient } from "@prisma/client";
import { TypeORMPerson } from "./model.js";

// @ast node: Var "sequelize"
export const sequelize = new Sequelize({
  dialect: "sqlite",
  storage: "./database.sqlite",
  logging: false,
});

// @ast node: Var "AppDataSource"
export const AppDataSource = new DataSource({
  type: "sqlite",
  database: "./database.sqlite",
  entities: [TypeORMPerson],
  synchronize: true,
  logging: false,
});

// @ast node: Var "prisma"
export const prisma = new PrismaClient();
