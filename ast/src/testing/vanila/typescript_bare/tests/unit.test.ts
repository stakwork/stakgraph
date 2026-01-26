import { User } from "../models/User.ts";

const testUser = new User({
  username: "testuser",
  email: "test@example.com",
});

if (testUser.isValid()) {
  console.log("PASS: User validation");
} else {
  console.error("FAIL: User validation");
}

const invalidUser = new User({
  username: "",
  email: "bad",
});

if (!invalidUser.isValid()) {
  console.log("PASS: Invalid user check");
} else {
  console.error("FAIL: Invalid user check");
}
