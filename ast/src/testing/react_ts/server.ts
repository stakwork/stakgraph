import express from 'express';
import cors from 'cors';

const app = express();
const port = 5000;

app.use(cors());
app.use(express.json());


const people = [];


app.get('/people', (req, res) => {
  res.json(people);
});


app.post('/person', (req, res) => {
  const { name, email } = req.body;
  const newPerson = {
    id: people.length + 1,
    name,
    email,
  };
  people.push(newPerson);
  res.json(newPerson);
});

app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});
