RSpec.describe PersonSerializer, type: :serializer do
  let(:person) { create(:person, name: "John Doe", email: "john@example.com") }
  let(:serializer) { PersonSerializer.new(person) }
  let(:serialization) { ActiveModelSerializers::Adapter.create(serializer) }

  describe "attributes" do
    subject { serialization.as_json }

    it "includes id" do
      expect(subject[:id]).to eq(person.id)
    end

    it "includes name" do
      expect(subject[:name]).to eq("John Doe")
    end

    it "includes email" do
      expect(subject[:email]).to eq("john@example.com")
    end

    it "does not include sensitive data" do
      expect(subject).not_to have_key(:password_digest)
      expect(subject).not_to have_key(:reset_token)
    end
  end

  describe "associations" do
    let(:person) { create(:person, :with_articles) }
    
    it "includes articles when specified" do
      serializer = PersonSerializer.new(person, include: :articles)
      result = ActiveModelSerializers::Adapter.create(serializer).as_json
      
      expect(result[:articles]).to be_present
      expect(result[:articles].length).to eq(2)
    end

    it "serializes article titles" do
      serializer = PersonSerializer.new(person, include: :articles)
      result = ActiveModelSerializers::Adapter.create(serializer).as_json
      
      expect(result[:articles].first).to have_key(:title)
    end
  end

  describe "custom methods" do
    it "includes full_name" do
      expect(serialization.as_json[:full_name]).to eq(person.name)
    end

    it "includes article_count" do
      create(:article, person: person)
      create(:article, person: person)
      
      expect(serialization.as_json[:article_count]).to eq(2)
    end
  end

  describe "with context" do
    it "includes admin fields for admin users" do
      serializer = PersonSerializer.new(person, scope: { admin: true })
      result = ActiveModelSerializers::Adapter.create(serializer).as_json
      
      expect(result).to have_key(:created_at)
      expect(result).to have_key(:updated_at)
    end

    it "excludes admin fields for regular users" do
      serializer = PersonSerializer.new(person, scope: { admin: false })
      result = ActiveModelSerializers::Adapter.create(serializer).as_json
      
      expect(result).not_to have_key(:created_at)
    end
  end
end
