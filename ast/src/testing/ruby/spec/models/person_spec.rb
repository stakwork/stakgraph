RSpec.describe Person, type: :model do
  subject { build(:person) }
  
  describe "validations" do
    it { should validate_presence_of(:name) }
    it { should validate_presence_of(:email) }
    it { should validate_uniqueness_of(:email) }
  end
  
  describe "associations" do
    it { should have_many(:articles).dependent(:destroy) }
  end
  
  it_behaves_like "a timestamped model"
  
  describe ".active" do
    let(:active_person) { create(:person) }
    let(:inactive_person) { create(:person, :inactive) }
    
    it "includes active people" do
      expect(Person.all).to include(active_person)
    end
  end
end
