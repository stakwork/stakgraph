RSpec.describe "Article Workflow", type: :request do
  describe "complete article creation and management workflow" do
    it "creates person and adds multiple articles via API" do
      # Step 1: Create a person via POST
      person_params = { person: { name: "Article Author", email: "author@example.com" } }
      post "/people", params: person_params
      expect(response).to have_http_status(:created)
      
      person_id = json_response['id']
      
      # Step 2: Add first article via POST
      article1_params = { article: { title: "First Article", body: "First Content" } }
      post "/people/#{person_id}/articles", params: article1_params
      expect(response).to have_http_status(:created)
      
      article1_data = json_response
      expect(article1_data['title']).to eq("First Article")
      expect(article1_data['person_id']).to eq(person_id)
      
      # Step 3: Add second article via POST
      article2_params = { article: { title: "Second Article", body: "Second Content" } }
      post "/people/#{person_id}/articles", params: article2_params
      expect(response).to have_http_status(:created)
      
      # Step 4: Verify articles association via GET
      get "/people/#{person_id}/articles"
      expect(response).to have_http_status(:ok)
      
      articles = json_response
      expect(articles.size).to eq(2)
      expect(articles.map { |a| a['title'] }).to contain_exactly("First Article", "Second Article")
      
      # Verify each article belongs to the person
      articles.each do |article|
        expect(article['person_id']).to eq(person_id)
      end
    end
  end
  
  describe "article CRUD operations workflow" do
    let(:person) { create(:person) }
    
    it "manages complete article lifecycle via HTTP endpoints" do
      # Step 1: Create article via POST
      article_params = { article: { title: "CRUD Article", body: "CRUD Content" } }
      post "/people/#{person.id}/articles", params: article_params
      expect(response).to have_http_status(:created)
      
      article_id = json_response['id']
      
      # Step 2: Retrieve article via GET
      get "/articles/#{article_id}"
      expect(response).to have_http_status(:ok)
      expect(json_response['title']).to eq("CRUD Article")
      
      # Step 3: Update article via PATCH
      update_params = { article: { title: "Updated CRUD Article", published: true } }
      patch "/articles/#{article_id}", params: update_params
      expect(response).to have_http_status(:ok)
      expect(json_response['title']).to eq("Updated CRUD Article")
      
      # Step 4: Verify update persisted
      get "/articles/#{article_id}"
      updated_article = json_response
      expect(updated_article['title']).to eq("Updated CRUD Article")
      expect(updated_article['published']).to eq(true)
      
      # Step 5: Delete article via DELETE
      delete "/articles/#{article_id}"
      expect(response).to have_http_status(:no_content)
      
      # Step 6: Verify deletion
      get "/articles/#{article_id}"
      expect(response).to have_http_status(:not_found)
    end
  end
  
  describe "article validation workflow" do
    let(:person) { create(:person) }
    
    it "handles invalid article creation and validates input" do
      # Step 1: Attempt to create article with empty title
      invalid_params = { article: { title: "", body: "" } }
      post "/people/#{person.id}/articles", params: invalid_params
      expect(response).to have_http_status(:unprocessable_entity)
      
      # Verify error messages
      expect(json_response['errors']).to be_present
      expect(json_response['errors']['title']).to include("can't be blank")
      
      # Step 2: Create valid article
      valid_params = { article: { title: "Valid Article", body: "Valid Content" } }
      post "/people/#{person.id}/articles", params: valid_params
      expect(response).to have_http_status(:created)
      
      article_id = json_response['id']
      
      # Step 3: Attempt invalid update
      invalid_update = { article: { title: "" } }
      patch "/articles/#{article_id}", params: invalid_update
      expect(response).to have_http_status(:unprocessable_entity)
      
      # Step 4: Verify original data unchanged
      get "/articles/#{article_id}"
      expect(json_response['title']).to eq("Valid Article")
    end
  end
  
  describe "cascade delete workflow" do
    it "deletes articles when person is deleted" do
      # Step 1: Create person
      person_params = { person: { name: "Delete Author", email: "delete@example.com" } }
      post "/people", params: person_params
      person_id = json_response['id']
      
      # Step 2: Create multiple articles
      3.times do |i|
        post "/people/#{person_id}/articles", 
             params: { article: { title: "Article #{i+1}", body: "Content #{i+1}" } }
        expect(response).to have_http_status(:created)
      end
      
      # Step 3: Verify articles exist
      get "/people/#{person_id}/articles"
      articles = json_response
      expect(articles.size).to eq(3)
      article_ids = articles.map { |a| a['id'] }
      
      # Step 4: Delete person
      delete "/people/#{person_id}"
      expect(response).to have_http_status(:no_content)
      
      # Step 5: Verify all articles are also deleted
      article_ids.each do |article_id|
        get "/articles/#{article_id}"
        expect(response).to have_http_status(:not_found)
      end
    end
  end
  
  describe "published article filtering workflow" do
    let!(:author) { create(:person) }
    
    it "manages article publication status through workflow" do
      # Step 1: Create draft article
      draft_params = { article: { title: "Draft Article", body: "Draft", published: false } }
      post "/people/#{author.id}/articles", params: draft_params
      draft_id = json_response['id']
      
      # Step 2: Create published article
      published_params = { article: { title: "Published Article", body: "Published", published: true } }
      post "/people/#{author.id}/articles", params: published_params
      published_id = json_response['id']
      
      # Step 3: Get all articles (including drafts)
      get "/people/#{author.id}/articles"
      all_articles = json_response
      expect(all_articles.size).to eq(2)
      
      # Step 4: Get only published articles
      get "/articles?published=true"
      published_articles = json_response
      expect(published_articles.size).to eq(1)
      expect(published_articles.first['title']).to eq("Published Article")
      
      # Step 5: Publish the draft
      patch "/articles/#{draft_id}", params: { article: { published: true } }
      expect(response).to have_http_status(:ok)
      
      # Step 6: Verify both now published
      get "/articles?published=true"
      expect(json_response.size).to eq(2)
    end
  end
end
