<script>
  // Integration tests for API endpoints
  const test_get_people_endpoint = async () => {
    const response = await fetch('http://localhost:5173/api/people');
    const people = await response.json();
    return response.status === 200 && Array.isArray(people);
  };

  const test_post_people_validates = async () => {
    const newPerson = { name: 'Charlie', email: 'charlie@test.com' };
    const response = await fetch('http://localhost:5173/api/people', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(newPerson)
    });
    const added = await response.json();
    return response.status === 200 && added.name === 'Charlie';
  };

  const test_post_people_rejects_invalid = async () => {
    const invalidPerson = { name: 'X', email: 'invalid@test.com' };
    const response = await fetch('http://localhost:5173/api/people', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(invalidPerson)
    });
    return response.status === 400;
  };
</script>

<svelte:head>
  <title>API Integration Tests</title>
</svelte:head>

<div></div>
