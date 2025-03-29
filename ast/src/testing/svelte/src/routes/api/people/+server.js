import { Person } from '$lib/Person'

let peopleDB = []

export function GET() {
  return new Response(JSON.stringify(peopleDB))
}

export async function POST({ request }) {
  try {
    const { name, email } = await request.json()
    const newPerson = new Person(name, email)

    Person.validate(newPerson) // Validate
    peopleDB.push(newPerson)   // Save

    return new Response(JSON.stringify(newPerson), {
      status: 201
    })

  } catch (error) {
    return new Response(error.message, {
      status: 400
    })
  }
}
