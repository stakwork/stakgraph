import { useState } from 'react';

function NewPerson() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');

  const handleSubmit = async (event: React.FormEvent) => {
  event.preventDefault();

  const newPerson = {
    name: name,
    email: email,
  };

  try {
    const response = await fetch('http://localhost:5000/person', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(newPerson),
    });

    if (!response.ok) {
      console.error('Failed to add new person:', response.statusText);
      return;
    }

    const data = await response.json();
    console.log('New person added:', data);
  } catch (error) {
    console.error('Error adding new person:', error);
  }
};


  return (
    <div>
      <h2>Add New Person</h2>
      <form onSubmit={handleSubmit}>
        <div>
          <label>Name:</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
          />
        </div>
        <div>
          <label>Email:</label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </div>
        <button type="submit">Add Person</button>
      </form>
    </div>
  );
}

export default NewPerson;
