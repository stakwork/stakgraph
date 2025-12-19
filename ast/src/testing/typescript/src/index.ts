import express from "express";
import "reflect-metadata";
import { sequelize, AppDataSource } from "./config.js";
import { registerRoutes } from "./routes.js";
// Import the routers
import userRouter from './routers/user-router';
import postRouter from './routers/post-router';
import { errorHandler } from './middleware/auth';

const app = express();
const port = 3000;

app.use(express.json());

// Register existing routes
registerRoutes(app);

// Mount routers with different patterns
app.use('/api/users', userRouter);
userRouter.use('/:userId/posts', postRouter);

// Global error handler middleware
app.use(errorHandler);

// Another pattern - router factory
function createApiRouter(prefix: string) {
  const apiRouter = express.Router();
  apiRouter.get('/', (req, res) => res.json({ message: `API at ${prefix}` }));
  return apiRouter;
}

// Mount a generated router
app.use('/api/v2', createApiRouter('/api/v2'));

async function initDatabases() {
  try {
    await sequelize.sync();

    await AppDataSource.initialize();

    app.listen(port, () => {
      console.log(`Server is running on http://localhost:${port}`);
    });
  } catch (error) {
    console.error(`Error initializing databases: ${error}`);
    process.exit(1);
  }
}

initDatabases();
