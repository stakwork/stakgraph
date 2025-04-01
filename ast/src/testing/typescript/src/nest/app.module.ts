import { Module } from "@nestjs/common";
import { PersonModule } from "./person/person.module.js";

@Module({
  imports: [PersonModule],
})
export class AppModule {}
