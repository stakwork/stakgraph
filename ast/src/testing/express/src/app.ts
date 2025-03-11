import express, { Request, Response } from 'express';
import { Sequelize } from 'sequelize';
import { SequelizeModel } from './models/SequelizeModel';
import { TypeORMModel } from './models/TypeORMModel';
import { createConnection, getRepository } from 'typeorm';
import 'reflect-metadata';

const app = express();
app.use(express.json());


const sequelize = new Sequelize({
  dialect: 'sqlite',
  storage: './database.sqlite',
});

sequelize.sync().then(() => {
  console.log('Database synced with Sequelize');
}).catch(err => {
  console.error('Error syncing database with Sequelize:', err);
});


createConnection({
  type: 'sqlite',
  database: './typeorm.sqlite',
  entities: [TypeORMModel],
  synchronize: true,
}).then(() => {
  console.log('Connected to TypeORM SQLite database');
}).catch((error) => {
  console.error('Error establishing TypeORM connection:', error);
});


app.get('/sequelize', (req: Request, res: Response) => {
  SequelizeModel.findAll().then((users) => {
    res.json(users);
  }).catch((error) => {
    res.status(500).json({ message: error.message });
  });
});


app.post('/sequelize', (req: Request, res: Response) => {
  const { name, age } = req.body;
  SequelizeModel.create({ name, age })
    .then((user) => {
      res.status(201).json(user);
    })
    .catch((error) => {
      res.status(500).json({ message: error.message });
    });
});


app.get('/typeorm', (req: Request, res: Response) => {

  const userRepository = getRepository(TypeORMModel);

  userRepository.find()
    .then((users: TypeORMModel[]) => {
      res.json(users);
    })
    .catch((error: any) => {
      res.status(500).json({ message: error.message });
    });
});


app.post('/typeorm', (req: Request, res: Response) => {
  const { name, age } = req.body;


  const userRepository = getRepository(TypeORMModel);


  const user = userRepository.create({ name, age });
  userRepository.save(user)
    .then((newUser: TypeORMModel) => {
      res.status(201).json(newUser);
    })
    .catch((error: any) => {
      res.status(500).json({ message: error.message });
    });
});

app.listen(3000, () => {
  console.log('Server is running on port 3000');
});
