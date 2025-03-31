import { useState } from 'react';
import styled from 'styled-components';
import * as api from '../api';

const FormContainer = styled.div`
  max-width: 400px;
  margin: 0 auto;
  padding: 20px;
  border: 1px solid #ccc;
  border-radius: 8px;
  background-color: #f9f9f9;
`;

const FormTitle = styled.h2`
  text-align: center;
  color: #333;
`;

const FormGroup = styled.div`
  margin-bottom: 15px;
`;

const Label = styled.label`
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
  color: #555;
`;

const Input = styled.input`
  width: 100%;
  padding: 8px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 16px;
`;

const SubmitButton = styled.button`
  width: 100%;
  padding: 10px;
  background-color: #28a745;
  color: white;
  border: none;
  border-radius: 4px;
  font-size: 16px;
  cursor: pointer;

  &:hover {
    background-color: #218838;
  }
`;

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
      const response = await fetch(`${api.host}/person`, {
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
    <FormContainer>
      <FormTitle>Add New Person</FormTitle>
      <form onSubmit={handleSubmit}>
        <FormGroup>
          <Label>Name:</Label>
          <Input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
          />
        </FormGroup>
        <FormGroup>
          <Label>Email:</Label>
          <Input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </FormGroup>
        <SubmitButton type="submit">Add Person</SubmitButton>
      </form>
    </FormContainer>
  );
}

export default NewPerson;
