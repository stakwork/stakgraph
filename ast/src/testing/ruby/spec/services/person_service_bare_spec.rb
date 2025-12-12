require 'rails_helper'

describe PersonService do
  let(:service) { PersonService.new }
  let(:person) { create(:person) }

  context "when creating a person" do
    it "returns the created person" do
      attrs = { name: "Jane Doe", email: "jane@example.com" }
      result = service.create_person(attrs)
      expect(result.name).to eq("Jane Doe")
    end
  end

  context "when updating a person" do
    it "updates the person attributes" do
      service.update_person(person.id, { name: "Updated Name" })
      person.reload
      expect(person.name).to eq("Updated Name")
    end
  end
end
