RSpec.describe PersonService do
  describe ".get_person_by_id" do
    it "returns a person when found" do
      person = Person.create(name: "John Doe", email: "john@example.com")
      result = PersonService.get_person_by_id(person.id)
      expect(result).to eq(person)
      expect(result.name).to eq("John Doe")
    end

    it "returns nil when person not found" do
      result = PersonService.get_person_by_id(9999)
      expect(result).to be_nil
    end
  end

  describe ".new_person" do
    it "creates a new person with valid params" do
      person_params = { name: "Jane Doe", email: "jane@example.com" }
      person = PersonService.new_person(person_params)
      expect(person.persisted?).to be true
      expect(person.name).to eq("Jane Doe")
      expect(person.email).to eq("jane@example.com")
    end

    it "fails to create person with invalid params" do
      person_params = { name: "", email: "" }
      person = PersonService.new_person(person_params)
      expect(person.persisted?).to be false
      expect(person.errors).not_to be_empty
    end
  end

  describe ".delete" do
    it "deletes an existing person" do
      person = Person.create(name: "To Delete", email: "delete@example.com")
      deleted = PersonService.delete(person.id)
      expect(deleted.destroyed?).to be true
      expect(Person.find_by(id: person.id)).to be_nil
    end
  end
end
