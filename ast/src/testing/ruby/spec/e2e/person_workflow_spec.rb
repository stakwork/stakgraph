RSpec.describe "Person Workflow" do
  it "creates, retrieves, and deletes a person" do
    # Step 1: Create a person
    person_params = { name: "Alice Smith", email: "alice@example.com" }
    person = PersonService.new_person(person_params)
    expect(person.persisted?).to be true
    expect(person.name).to eq("Alice Smith")
    
    # Step 2: Retrieve the person
    retrieved_person = PersonService.get_person_by_id(person.id)
    expect(retrieved_person).to eq(person)
    expect(retrieved_person.email).to eq("alice@example.com")
    
    # Step 3: Delete the person
    deleted_person = PersonService.delete(person.id)
    expect(deleted_person.destroyed?).to be true
    
    # Step 4: Verify deletion
    expect(PersonService.get_person_by_id(person.id)).to be_nil
  end

  it "manages person through controller endpoints" do
    # Step 1: Create person via controller
    controller = PeopleController.new
    controller.params = { person: { name: "Bob Jones", email: "bob@example.com" } }
    controller.create_person
    
    person = Person.find_by(email: "bob@example.com")
    expect(person).not_to be_nil
    
    # Step 2: Get person via controller
    controller.params = { id: person.id }
    controller.get_person
    expect(Person.find(person.id)).to eq(person)
    
    # Step 3: Delete person via controller
    controller.destroy
    expect(Person.find_by(id: person.id)).to be_nil
  end

  it "displays person profile page" do
    # Create a person
    person = Person.create(name: "Profile User", email: "profile@example.com")
    
    # Show profile
    controller = PeopleController.new
    controller.params = { id: person.id }
    controller.show_person_profile
    
    # Verify instance variable is set (simulating view rendering)
    expect(controller.instance_variable_get(:@person)).to eq(person)
  end
end
