RSpec.describe Person do
  describe "validations" do
    it "validates presence of name" do
      person = Person.new(email: "test@example.com")
      expect(person.valid?).to be false
      expect(person.errors[:name]).to include("can't be blank")
    end

    it "validates presence and uniqueness of email" do
      Person.create(name: "First", email: "unique@example.com")
      person = Person.new(name: "Second", email: "unique@example.com")
      expect(person.valid?).to be false
      expect(person.errors[:email]).to include("has already been taken")
    end

    it "is valid with all required attributes" do
      person = Person.new(name: "Valid Person", email: "valid@example.com")
      expect(person.valid?).to be true
    end
  end

  describe "associations" do
    it "has many articles" do
      person = Person.create(name: "Author", email: "author@example.com")
      article1 = person.articles.create(title: "Article 1", body: "Content 1")
      article2 = person.articles.create(title: "Article 2", body: "Content 2")
      expect(person.articles.count).to eq(2)
      expect(person.articles).to include(article1, article2)
    end

    it "destroys associated articles when person is destroyed" do
      person = Person.create(name: "Author", email: "author@example.com")
      article = person.articles.create(title: "Article", body: "Content")
      person.destroy
      expect(Article.find_by(id: article.id)).to be_nil
    end
  end
end

RSpec.describe Article do
  describe "validations" do
    it "validates presence of title" do
      person = Person.create(name: "Author", email: "author@example.com")
      article = Article.new(body: "Content", person: person)
      expect(article.valid?).to be false
      expect(article.errors[:title]).to include("can't be blank")
    end

    it "validates presence of body" do
      person = Person.create(name: "Author", email: "author@example.com")
      article = Article.new(title: "Title", person: person)
      expect(article.valid?).to be false
      expect(article.errors[:body]).to include("can't be blank")
    end

    it "is valid with all required attributes" do
      person = Person.create(name: "Author", email: "author@example.com")
      article = Article.new(title: "Valid Article", body: "Valid Content", person: person)
      expect(article.valid?).to be true
    end
  end

  describe "associations" do
    it "belongs to a person" do
      person = Person.create(name: "Author", email: "author@example.com")
      article = Article.create(title: "Article", body: "Content", person: person)
      expect(article.person).to eq(person)
    end
  end
end
