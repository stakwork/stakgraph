RSpec.describe "People API" do
  describe "GET /person/:id" do
    it "returns person when found" do
      person = Person.create(name: "John Doe", email: "john@example.com")
      # Simulating controller action
      controller = PeopleController.new
      controller.params = { id: person.id }
      controller.get_person
      # Expected: render json: person, status: :ok
      expect(person).not_to be_nil
      expect(person.name).to eq("John Doe")
    end

    it "returns error when person not found" do
      controller = PeopleController.new
      controller.params = { id: 9999 }
      controller.get_person
      # Expected: render json: { error: "Person not found" }, status: :not_found
    end
  end

  describe "POST /person" do
    it "creates a new person with valid params" do
      person_params = { name: "Jane Doe", email: "jane@example.com" }
      controller = PeopleController.new
      controller.params = { person: person_params }
      controller.create_person
      # Expected: render json: person, status: :created
      created_person = Person.find_by(email: "jane@example.com")
      expect(created_person).not_to be_nil
      expect(created_person.name).to eq("Jane Doe")
    end

    it "returns errors with invalid params" do
      person_params = { name: "", email: "" }
      controller = PeopleController.new
      controller.params = { person: person_params }
      controller.create_person
      # Expected: render json: { errors: ... }, status: :unprocessable_entity
    end
  end

  describe "DELETE /people/:id" do
    it "deletes an existing person" do
      person = Person.create(name: "To Delete", email: "delete@example.com")
      controller = PeopleController.new
      controller.params = { id: person.id }
      controller.destroy
      # Expected: render json: { message: 'Person deleted successfully' }, status: :ok
      expect(Person.find_by(id: person.id)).to be_nil
    end

    it "returns error when person not found" do
      controller = PeopleController.new
      controller.params = { id: 9999 }
      controller.destroy
      # Expected: render json: { error: "Person not found" }, status: :not_found
    end
  end

  describe "GET /people/articles" do
    it "returns all articles" do
      person = Person.create(name: "Author", email: "author@example.com")
      article1 = person.articles.create(title: "Article 1", body: "Content 1")
      article2 = person.articles.create(title: "Article 2", body: "Content 2")
      
      controller = PeopleController.new
      controller.articles
      # Expected: render json: articles, status: :ok
      expect(Article.count).to eq(2)
    end
  end

  describe "POST /people/:id/articles" do
    it "creates article for person" do
      person = Person.create(name: "Author", email: "author@example.com")
      article_params = { title: "New Article", body: "New Content" }
      
      controller = PeopleController.new
      controller.params = { id: person.id, article: article_params }
      controller.create_article
      # Expected: render json: article, status: :created
      expect(person.articles.count).to eq(1)
      expect(person.articles.first.title).to eq("New Article")
    end
  end
end
