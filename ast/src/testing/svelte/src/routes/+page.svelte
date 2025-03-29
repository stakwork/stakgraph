<script>
  import { onMount } from 'svelte';


  let people = [];
  let name = '';
  let email = '';
  let isLoading = false;
  let error = null;
  let formError = null;
  let formSuccess = false;

  onMount(async () => {
    await fetchPeople();
  });

  async function fetchPeople() {
    isLoading = true;
    error = null;

    try {
      const response = await fetch('/api/people');
      if (!response.ok) throw new Error('Failed to fetch');
      people = await response.json();
    } catch (err) {
      error = err.message;
    } finally {
      isLoading = false;
    }
  }

  async function addPerson() {
    if (!name) {
      formError = 'Name is required';
      return;
    }

    isLoading = true;
    formError = null;
    formSuccess = false;

    try {
      const response = await fetch('/api/people', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, email })
      });

      if (!response.ok) throw new Error('Failed to add person');

      // Clear form
      name = '';
      email = '';
      formSuccess = true;
      await fetchPeople();
    } catch (err) {
      formError = err.message;
    } finally {
      isLoading = false;
    }
  }
</script>

<main>
  <h1>People Manager</h1>

  <div class="form-section">
    <h2>Add New Person</h2>
    <form on:submit|preventDefault={addPerson}>
      <div>
        <label for="name">Name*</label>
        <input id="name" type="text" bind:value={name} required>
      </div>

      <div>
        <label for="email">Email</label>
        <input id="email" type="email" bind:value={email}>
      </div>

      {#if formError}
        <p class="error">{formError}</p>
      {/if}

      {#if formSuccess}
        <p class="success">Person added successfully!</p>
      {/if}

      <button type="submit" disabled={isLoading}>
        {isLoading ? 'Adding...' : 'Add Person'}
      </button>
    </form>
  </div>

  <div class="list-section">
    <h2>People List</h2>

    {#if isLoading && !people.length}
      <p>Loading people...</p>
    {:else if error}
      <p class="error">{error}</p>
    {:else}
      <ul>
        {#each people as person (person.id)}
          <li>
            {person.name}
            {#if person.email}
              <span class="email">({person.email})</span>
            {/if}
          </li>
        {:else}
          <li>No people found</li>
        {/each}
      </ul>

      <button on:click={fetchPeople} disabled={isLoading}>
        Refresh List
      </button>
    {/if}
  </div>
</main>

<style>
  main {
    max-width: 800px;
    margin: 0 auto;
    padding: 2rem;
    font-family: sans-serif;
  }

  h1 {
    color: #333;
    text-align: center;
  }

  .form-section, .list-section {
    background: #f5f5f5;
    padding: 1.5rem;
    border-radius: 8px;
    margin-bottom: 2rem;
  }

  label {
    display: block;
    margin-bottom: 0.5rem;
    font-weight: bold;
  }

  input {
    width: 100%;
    padding: 0.5rem;
    margin-bottom: 1rem;
    border: 1px solid #ddd;
    border-radius: 4px;
  }

  button {
    background: #4CAF50;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    cursor: pointer;
    margin-top: 0.5rem;
  }

  button:hover {
    background: #45a049;
  }

  button:disabled {
    background: #cccccc;
    cursor: not-allowed;
  }

  ul {
    list-style: none;
    padding: 0;
  }

  li {
    padding: 0.5rem;
    border-bottom: 1px solid #eee;
  }

  .email {
    color: #666;
    font-size: 0.9em;
  }

  .error {
    color: #f44336;
  }

  .success {
    color: #4CAF50;
  }
</style>
