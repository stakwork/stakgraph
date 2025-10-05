RSpec.describe "Articles API" do
  describe "GET /people/articles" do
    it "returns empty array when no articles exist" do
      controller = PeopleController.new
      controller.articles
      # Expected: render json: articles, status: :ok
      expect(Article.count).to eq(0)
    end

    it "returns all articles across all people" do
      person1 = Person.create(name: "Author 1", email: "author1@example.com")
      person2 = Person.create(name: "Author 2", email: "author2@example.com")
      
      article1 = person1.articles.create(title: "Article 1", body: "Content 1")
      article2 = person2.articles.create(title: "Article 2", body: "Content 2")
      article3 = person1.articles.create(title: "Article 3", body: "Content 3")
      
      controller = PeopleController.new
      controller.articles
      # Expected: render json: articles, status: :ok
      expect(Article.count).to eq(3)
    end
  end

  describe "POST /people/:id/articles" do
    it "creates article with valid params" do
      person = Person.create(name: "Author", email: "author@example.com")
      article_params = { title: "Test Article", body: "Test Content" }
      
      controller = PeopleController.new
      controller.params = { id: person.id, article: article_params }
      controller.create_article
      # Expected: render json: article, status: :created
      
      article = Article.find_by(title: "Test Article")
      expect(article).not_to be_nil
      expect(article.person).to eq(person)
      expect(article.body).to eq("Test Content")
    end

    it "returns errors with invalid params" do
      person = Person.create(name: "Author", email: "author@example.com")
      article_params = { title: "", body: "" }
      
      controller = PeopleController.new
      controller.params = { id: person.id, article: article_params }
      controller.create_article
      # Expected: render json: article.errors, status: :unprocessable_entity
    end

    it "handles association correctly" do
      person = Person.create(name: "Author", email: "author@example.com")
      article_params = { title: "Associated Article", body: "Associated Content" }
      
      controller = PeopleController.new
      controller.params = { id: person.id, article: article_params }
      controller.create_article
      
      expect(person.reload.articles.count).to eq(1)
      expect(person.articles.first.title).to eq("Associated Article")
    end
  end
end
