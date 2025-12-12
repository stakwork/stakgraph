RSpec.describe Person, type: :model do
  describe "validations" do
    it "validates presence of name" do
      person = build(:person, name: nil)
      expect(person.valid?).to be false
      expect(person.errors[:name]).to include("can't be blank")
    end

    it "validates uniqueness of email" do
      create(:person, email: "unique@example.com")
      person = build(:person, email: "unique@example.com")
      expect(person.valid?).to be false
      expect(person.errors[:email]).to include("has already been taken")
    end

    it "is valid with all required attributes" do
      person = build(:person)
      expect(person.valid?).to be true
    end
  end

  describe "associations" do
    it "has many articles" do
      person = create(:person, :with_articles)
      expect(person.articles.count).to eq(2)
    end

    it "destroys associated articles when person is destroyed" do
      person = create(:person)
      article = create(:article, person: person)
      person.destroy
      expect(Article.find_by(id: article.id)).to be_nil
    end
  end
end

RSpec.describe Article, type: :model do
  describe "validations" do
    it "validates presence of title" do
      article = build(:article, title: nil)
      expect(article.valid?).to be false
      expect(article.errors[:title]).to include("can't be blank")
    end

    it "validates presence of body" do
      article = build(:article, body: nil)
      expect(article.valid?).to be false
      expect(article.errors[:body]).to include("can't be blank")
    end

    it "is valid with all required attributes" do
      article = build(:article)
      expect(article.valid?).to be true
    end
  end

  describe "associations" do
    let(:person) { create(:person) }
    let(:article) { create(:article, person: person) }
    
    it "belongs to person" do
      expect(article.person).to eq(person)
    end
  end
end
