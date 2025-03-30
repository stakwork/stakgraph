import { Module } from "@nestjs/common";
import { PersonController } from "./person.controller.js";
import { PersonService } from "./person.service.js";

@Module({
  controllers: [PersonController],
  providers: [PersonService],
})
export class PersonModule {}
