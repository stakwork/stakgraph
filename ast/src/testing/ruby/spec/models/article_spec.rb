RSpec.describe Article, type: :model do
  subject { build(:article) }
  
  describe "validations" do
    it { should validate_presence_of(:title) }
    it { should validate_presence_of(:body) }
  end
  
  describe "associations" do
    it { should belong_to(:person) }
  end
  
  it_behaves_like "a timestamped model"
  
  describe "#published?" do
    let(:published_article) { create(:article, :published) }
    let(:draft_article) { create(:article, :draft) }
    
    it "returns true for published articles" do
      expect(published_article.published).to be true
    end
    
    it "returns false for draft articles" do
      expect(draft_article.published).to be false
    end
  end
end
