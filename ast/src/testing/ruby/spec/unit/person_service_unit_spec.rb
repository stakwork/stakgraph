RSpec.describe PersonService, type: :service do
  describe ".get_person_by_id" do
    let(:person) { create(:person) }
    
    context "when person exists" do
      it "returns the person" do
        result = PersonService.get_person_by_id(person.id)
        expect(result).to eq(person)
      end
      
      it "returns person with correct attributes" do
        result = PersonService.get_person_by_id(person.id)
        expect(result.name).to eq(person.name)
        expect(result.email).to eq(person.email)
      end
    end
    
    context "when person not found" do
      it "returns nil" do
        result = PersonService.get_person_by_id(9999)
        expect(result).to be_nil
      end
    end
  end

  describe ".new_person" do
    let(:valid_params) { { name: "Jane Doe", email: "jane@example.com" } }
    let(:invalid_params) { { name: "", email: "" } }
    
    context "with valid params" do
      it "creates a new person" do
        expect {
          PersonService.new_person(valid_params)
        }.to change(Person, :count).by(1)
      end
      
      it "returns persisted person with correct attributes" do
        person = PersonService.new_person(valid_params)
        expect(person.persisted?).to be true
        expect(person.name).to eq("Jane Doe")
        expect(person.email).to eq("jane@example.com")
      end
    end
    
    context "with invalid params" do
      it "does not create person" do
        expect {
          PersonService.new_person(invalid_params)
        }.not_to change(Person, :count)
      end
      
      it "returns unpersisted person with errors" do
        person = PersonService.new_person(invalid_params)
        expect(person.persisted?).to be false
        expect(person.errors).to be_present
      end
    end
  end

  describe ".delete" do
    let!(:person) { create(:person) }
    
    it "deletes the person" do
      expect {
        PersonService.delete(person.id)
      }.to change(Person, :count).by(-1)
    end
    
    it "returns destroyed person" do
      deleted = PersonService.delete(person.id)
      expect(deleted.destroyed?).to be true
    end
    
    it "removes person from database" do
      PersonService.delete(person.id)
      expect(Person.find_by(id: person.id)).to be_nil
    end
  end
end
