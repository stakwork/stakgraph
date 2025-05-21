import { App } from "./app.js";
import dotenv from "dotenv";

// Load environment variables
dotenv.config();

const app = new App();
const PORT = process.env.PORT || 3000;

app.app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
