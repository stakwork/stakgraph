import { Entity, PrimaryGeneratedColumn, Column, DataSource } from 'typeorm';


@Entity()
export class User {
  @PrimaryGeneratedColumn()
  id: number;

  @Column()
  name: string;

  @Column()
  age: number;




  constructor(id?: number, name?: string, age?: number) {
    this.id = id || 0;
    this.name = name || '';
    this.age = age || 0;
  }
}


const AppDataSource = new DataSource({
  type: 'sqlite',
  database: './mydb.db',
  entities: [User],
  synchronize: true,
  logging: true,
});

AppDataSource.initialize()
  .then(() => {
    console.log('TypeORM DataSource initialized!');
  })
  .catch((err) => {
    console.error('Error during TypeORM DataSource initialization', err);
  });

export { User as TypeORMModel, AppDataSource };
