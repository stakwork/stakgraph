import { db } from "../lib/database.ts";
import { User } from "../models/User.ts";
import { generateId } from "../lib/utils.ts";

export async function createUser(data: any) {
  const user = new User({
    ...data,
    id: generateId(),
  });

  if (!user.isValid()) {
    throw new Error("Invalid user data");
  }

  await db.users.save(user);
  return user;
}

export async function getUser(id: string) {
  return await db.users.findById(id);
}
