RSpec.describe "Article Workflow" do
  it "creates person and adds articles to them" do
    # Step 1: Create a person
    person = Person.create(name: "Article Author", email: "author@example.com")
    expect(person.persisted?).to be true
    
    # Step 2: Add first article
    article1 = person.articles.create(title: "First Article", body: "First Content")
    expect(article1.persisted?).to be true
    expect(person.articles.count).to eq(1)
    
    # Step 3: Add second article
    article2 = person.articles.create(title: "Second Article", body: "Second Content")
    expect(person.articles.count).to eq(2)
    
    # Step 4: Verify articles association
    expect(person.articles).to include(article1, article2)
    expect(article1.person).to eq(person)
    expect(article2.person).to eq(person)
  end

  it "manages articles through controller workflow" do
    # Step 1: Create person
    person = Person.create(name: "Controller Author", email: "controller@example.com")
    
    # Step 2: Create article via controller
    controller = PeopleController.new
    controller.params = { 
      id: person.id, 
      article: { title: "Controller Article", body: "Controller Content" }
    }
    controller.create_article
    
    # Step 3: Verify article creation
    article = Article.find_by(title: "Controller Article")
    expect(article).not_to be_nil
    expect(article.person).to eq(person)
    
    # Step 4: Get all articles
    controller.articles
    expect(Article.all).to include(article)
  end

  it "handles complete article lifecycle with validation" do
    # Step 1: Create person
    person = Person.create(name: "Lifecycle Author", email: "lifecycle@example.com")
    
    # Step 2: Try to create invalid article
    invalid_article = person.articles.build(title: "", body: "")
    expect(invalid_article.valid?).to be false
    
    # Step 3: Create valid article
    valid_article = person.articles.create(title: "Valid Article", body: "Valid Content")
    expect(valid_article.persisted?).to be true
    
    # Step 4: Delete person and verify cascade delete
    person.destroy
    expect(Article.find_by(id: valid_article.id)).to be_nil
  end
end
