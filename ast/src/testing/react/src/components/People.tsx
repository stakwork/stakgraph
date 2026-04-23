import { useEffect } from "react";
import { Person, useStore } from "./Person";
import * as api from "../api";

// @ast node: Function "People"
function People() {
  const { state, setPeople, setLoading } = useStore();
  const { people, loading } = state;

  useEffect(() => {
    setLoading(true);
    // Fetch people data
    // @ast node: Request "${api.host}/people" [verb=GET]
    fetch(`${api.host}/people`)
      .then((response) => response.json())
      .then((data: Person[]) => {
        setPeople(data);
        setLoading(false);
      })
      .catch((error) => {
        console.error("Error fetching people:", error);
        setLoading(false);
      });
  }, [setPeople, setLoading]);

  if (loading) {
    return <div>Loading...</div>;
  }

  return (
    <div>
      <h2>People List</h2>
      <ul>
        {people.map((person) => (
          <li key={person.id}>
            <h3>{person.name}</h3>
            <p>{person.email}</p>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default People;
