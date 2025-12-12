RSpec.describe PersonService, type: :service do
  let(:service) { described_class.new }
  
  describe "#find_person" do
    context "when person exists" do
      let!(:person) { create(:person, name: "John Doe") }
      
      it "returns the person" do
        result = service.find_person(person.id)
        expect(result).to eq(person)
      end
      
      it "includes person attributes" do
        result = service.find_person(person.id)
        expect(result.name).to eq("John Doe")
        expect(result.email).to eq(person.email)
      end
    end
    
    context "when person does not exist" do
      it "returns nil" do
        result = service.find_person(99999)
        expect(result).to be_nil
      end
    end
    
    context "with articles association" do
      let!(:person) { create(:person, :with_articles) }
      
      it "includes associated articles" do
        result = service.find_person(person.id)
        expect(result.articles.count).to eq(2)
      end
    end
  end
  
  describe "#create_person" do
    let(:valid_attributes) { { name: "Jane Doe", email: "jane@example.com" } }
    
    context "with valid attributes" do
      it "creates a new person" do
        expect {
          service.create_person(valid_attributes)
        }.to change(Person, :count).by(1)
      end
      
      it "returns the created person" do
        result = service.create_person(valid_attributes)
        expect(result).to be_a(Person)
        expect(result.name).to eq("Jane Doe")
        expect(result.email).to eq("jane@example.com")
      end
      
      it "persists the person" do
        result = service.create_person(valid_attributes)
        expect(result).to be_persisted
        expect(result.id).to be_present
      end
    end
    
    context "with invalid attributes" do
      let(:invalid_attributes) { { name: "", email: "" } }
      
      it "does not create a person" do
        expect {
          service.create_person(invalid_attributes)
        }.not_to change(Person, :count)
      end
      
      it "returns validation errors" do
        result = service.create_person(invalid_attributes)
        expect(result.errors).to be_present
        expect(result.errors[:name]).to include("can't be blank")
      end
    end
    
    context "with duplicate email" do
      let!(:existing) { create(:person, email: "existing@example.com") }
      let(:duplicate_attrs) { { name: "Another", email: "existing@example.com" } }
      
      it "does not create a person" do
        expect {
          service.create_person(duplicate_attrs)
        }.not_to change(Person, :count)
      end
      
      it "returns uniqueness error" do
        result = service.create_person(duplicate_attrs)
        expect(result.errors[:email]).to include("has already been taken")
      end
    end
  end
  
  describe "#update_person" do
    let(:person) { create(:person, name: "Original Name") }
    
    context "with valid attributes" do
      let(:new_attributes) { { name: "Updated Name" } }
      
      it "updates the person" do
        service.update_person(person.id, new_attributes)
        expect(person.reload.name).to eq("Updated Name")
      end
      
      it "returns true" do
        result = service.update_person(person.id, new_attributes)
        expect(result).to be true
      end
    end
    
    context "with invalid attributes" do
      let(:invalid_attributes) { { name: "" } }
      
      it "does not update the person" do
        original_name = person.name
        service.update_person(person.id, invalid_attributes)
        expect(person.reload.name).to eq(original_name)
      end
      
      it "returns false" do
        result = service.update_person(person.id, invalid_attributes)
        expect(result).to be false
      end
    end
  end
  
  describe "#delete_person" do
    let!(:person) { create(:person) }
    
    context "when person exists" do
      it "deletes the person" do
        expect {
          service.delete_person(person.id)
        }.to change(Person, :count).by(-1)
      end
      
      it "returns true" do
        result = service.delete_person(person.id)
        expect(result).to be true
      end
    end
    
    context "when person has articles" do
      let!(:person_with_articles) { create(:person, :with_articles) }
      
      it "deletes associated articles" do
        expect {
          service.delete_person(person_with_articles.id)
        }.to change(Article, :count).by(-2)
      end
    end
    
    context "when person does not exist" do
      it "returns false" do
        result = service.delete_person(99999)
        expect(result).to be false
      end
    end
  end
  
  describe "#list_people" do
    context "with no people" do
      it "returns empty array" do
        result = service.list_people
        expect(result).to eq([])
      end
    end
    
    context "with multiple people" do
      let!(:person1) { create(:person, name: "Alice") }
      let!(:person2) { create(:person, name: "Bob") }
      let!(:person3) { create(:person, name: "Charlie") }
      
      it "returns all people" do
        result = service.list_people
        expect(result.size).to eq(3)
      end
      
      it "returns people in order" do
        result = service.list_people
        names = result.map(&:name)
        expect(names).to eq(["Alice", "Bob", "Charlie"])
      end
    end
    
    context "with pagination" do
      before { create_list(:person, 15) }
      
      it "returns paginated results" do
        result = service.list_people(page: 1, per_page: 10)
        expect(result.size).to eq(10)
      end
    end
  end
end
