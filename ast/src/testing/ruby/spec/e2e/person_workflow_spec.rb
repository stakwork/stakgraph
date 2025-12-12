RSpec.describe "Person Workflow", type: :request do
  describe "complete person CRUD workflow" do
    it "creates, retrieves, updates, and deletes a person via API" do
      # Step 1: Create a person via POST
      person_params = { person: { name: "Alice Smith", email: "alice@example.com" } }
      
      post "/people", params: person_params
      expect(response).to have_http_status(:created)
      
      created_person = json_response
      person_id = created_person['id']
      expect(created_person['name']).to eq("Alice Smith")
      expect(created_person['email']).to eq("alice@example.com")
      
      # Step 2: Retrieve the person via GET
      get "/people/#{person_id}"
      expect(response).to have_http_status(:ok)
      
      retrieved_person = json_response
      expect(retrieved_person['id']).to eq(person_id)
      expect(retrieved_person['name']).to eq("Alice Smith")
      expect(retrieved_person['email']).to eq("alice@example.com")
      
      # Step 3: Update the person via PATCH
      update_params = { person: { name: "Alice Johnson" } }
      
      patch "/people/#{person_id}", params: update_params
      expect(response).to have_http_status(:ok)
      
      updated_person = json_response
      expect(updated_person['name']).to eq("Alice Johnson")
      expect(updated_person['email']).to eq("alice@example.com") # unchanged
      
      # Step 4: Delete the person via DELETE
      delete "/people/#{person_id}"
      expect(response).to have_http_status(:no_content)
      
      # Step 5: Verify deletion - GET should return 404
      get "/people/#{person_id}"
      expect(response).to have_http_status(:not_found)
    end
  end
  
  describe "person with articles workflow" do
    it "creates person, adds articles, and retrieves full profile" do
      # Step 1: Create person
      person_params = { person: { name: "Bob Jones", email: "bob@example.com" } }
      post "/people", params: person_params
      
      person_id = json_response['id']
      
      # Step 2: Create first article for person
      article1_params = { article: { title: "First Article", body: "Content 1" } }
      post "/people/#{person_id}/articles", params: article1_params
      expect(response).to have_http_status(:created)
      
      article1_id = json_response['id']
      
      # Step 3: Create second article
      article2_params = { article: { title: "Second Article", body: "Content 2" } }
      post "/people/#{person_id}/articles", params: article2_params
      
      article2_id = json_response['id']
      
      # Step 4: Retrieve person with articles
      get "/people/#{person_id}"
      expect(response).to have_http_status(:ok)
      
      person_data = json_response
      expect(person_data['articles'].size).to eq(2)
      expect(person_data['articles'].map { |a| a['title'] }).to contain_exactly("First Article", "Second Article")
      
      # Step 5: List all articles for person
      get "/people/#{person_id}/articles"
      articles = json_response
      expect(articles.size).to eq(2)
      
      # Step 6: Delete person (cascade deletes articles)
      delete "/people/#{person_id}"
      expect(response).to have_http_status(:no_content)
      
      # Step 7: Verify articles are also deleted
      get "/articles/#{article1_id}"
      expect(response).to have_http_status(:not_found)
    end
  end
  
  describe "person profile page workflow" do
    let(:user) { create(:person) }
    
    it "navigates through person profile and updates settings" do
      # Step 1: Create person with articles
      person = create(:person, :with_articles, name: "Profile User")
      
      # Step 2: View profile page
      get "/people/#{person.id}/profile"
      expect(response).to have_http_status(:ok)
      expect(json_response['name']).to eq("Profile User")
      expect(json_response['articles']).to be_present
      
      # Step 3: Update profile settings
      settings_params = { person: { bio: "Updated bio", location: "New York" } }
      patch "/people/#{person.id}", params: settings_params
      expect(response).to have_http_status(:ok)
      
      # Step 4: Verify updated profile
      get "/people/#{person.id}/profile"
      profile_data = json_response
      expect(profile_data['bio']).to eq("Updated bio")
      expect(profile_data['location']).to eq("New York")
    end
  end
  
  describe "multi-person collaboration workflow" do
    it "creates multiple people and manages their interactions" do
      # Step 1: Create first author
      post "/people", params: { person: { name: "Author 1", email: "author1@example.com" } }
      author1_id = json_response['id']
      
      # Step 2: Create second author
      post "/people", params: { person: { name: "Author 2", email: "author2@example.com" } }
      author2_id = json_response['id']
      
      # Step 3: Each author creates articles
      post "/people/#{author1_id}/articles", params: { article: { title: "Author 1 Article", body: "Content" } }
      post "/people/#{author2_id}/articles", params: { article: { title: "Author 2 Article", body: "Content" } }
      
      # Step 4: List all articles across all authors
      get "/people/articles"
      all_articles = json_response
      expect(all_articles.size).to eq(2)
      
      # Step 5: Filter articles by author
      get "/people/#{author1_id}/articles"
      author1_articles = json_response
      expect(author1_articles.size).to eq(1)
      expect(author1_articles.first['title']).to eq("Author 1 Article")
    end
  end
end
