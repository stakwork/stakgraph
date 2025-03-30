import express from "express";
import "reflect-metadata";
import { sequelize, AppDataSource } from "./shared/config.js";
import { registerRoutes } from "./express/routes.js";
import { NestFactory } from "@nestjs/core";
import { AppModule } from "./nest/app.module.js";

const app = express();
const port = 3000;
const nestPort = 3002;

app.use(express.json());

registerRoutes(app);

async function runningServers() {
  try {
    await sequelize.sync();
    await AppDataSource.initialize();

    app.listen(port, () => {
      console.log(`Express server is running on http://localhost:${port}`);
    });

    const nestApp = await NestFactory.create(AppModule);
    await nestApp.listen(nestPort);
    console.log(`NestJS server is running on http://localhost:${nestPort}`);
  } catch (error) {
    console.error(`Error initializing servers: ${error}`);
    process.exit(1);
  }
}

runningServers();
