import { Sequelize, DataTypes, Model } from 'sequelize';


const sequelize = new Sequelize('sqlite::memory:');

class User extends Model {
  public id!: number;
  public name!: string;
  public age!: number;
}

User.init(
  {
    id: {
      type: DataTypes.INTEGER,
      primaryKey: true,
      autoIncrement: true,
    },
    name: {
      type: DataTypes.STRING,
      allowNull: false,
    },
    age: {
      type: DataTypes.INTEGER,
      allowNull: false,
    },
  },
  {
    sequelize,
    modelName: 'User',
    tableName: 'users',
  }
);
sequelize.sync().then(() => {
  return User.findOrCreate({
    where: { name: 'John Doe' },
    defaults: { age: 30 },
  });
});
sequelize.sync();

export { User as SequelizeModel };
