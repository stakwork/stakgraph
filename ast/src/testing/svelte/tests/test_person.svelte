<script>
// @ast node: Class ""
// @ast node: DataModel "<script>"
// @ast node: DataModel "<svelte:head>"
// @ast node: DataModel "<div>"
// @ast node: Function "  const test_person_"
// @ast node: Function "  const test_person_validate"
// @ast node: Function "  const test_person_validate_rejec"
// @ast node: Function "  const test_person_validate_rejec"
  import Person from '../src/lib/Person.js';

  // Unit tests for Person class
  const test_person_creation = () => {
    const person = new Person('Alice', 'alice@example.com');
    return person.name === 'Alice' && person.email === 'alice@example.com';
  };

  const test_person_validate_accepts = () => {
    const person = { name: 'Bob Smith', email: 'bob@test.com' };
    try { Person.validate(person); return true; } catch { return false; }
  };

  const test_person_validate_rejects_empty = () => {
    const person = { name: '   ', email: 'test@test.com' };
    try { Person.validate(person); return false; } catch { return true; }
  };

  const test_person_validate_rejects_short = () => {
    const person = { name: 'A', email: 'test@test.com' };
    try { Person.validate(person); return false; } catch { return true; }
  };
</script>

<svelte:head>
  <title>Person Tests</title>
</svelte:head>

<div></div>
