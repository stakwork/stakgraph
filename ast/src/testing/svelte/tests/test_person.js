<script>
  import { describe, test, expect } from 'vitest';
  import { Person } from '../src/lib/Person.js';

  function testPersonCreation() {
    const person = new Person('Alice', 'alice@example.com');
    expect(person.name).toBe('Alice');
    expect(person.email).toBe('alice@example.com');
  }

  function testPersonValidateAccepts() {
    const person = { name: 'Bob Smith', email: 'bob@test.com' };
    expect(() => Person.validate(person)).not.toThrow();
  }

  function testPersonValidateRejectsEmpty() {
    const person = { name: '   ', email: 'test@test.com' };
    expect(() => Person.validate(person)).toThrow();
  }

  function testPersonValidateRejectsShort() {
    const person = { name: 'A', email: 'test@test.com' };
    expect(() => Person.validate(person)).toThrow();
  }
</script>

<svelte:head>
  <title>Person Tests</title>
</svelte:head>

<div></div>
