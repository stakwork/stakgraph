RSpec.describe "Articles API", type: :request do
  describe "GET /people/articles" do
    context "when no articles exist" do
      before { get "/people/articles" }
      
      it "returns success status" do
        expect(response).to have_http_status(:ok)
      end
      
      it "returns empty array" do
        expect(json_response).to eq([])
      end
    end

    context "when articles exist across multiple people" do
      let!(:person1) { create(:person, name: "Author 1") }
      let!(:person2) { create(:person, name: "Author 2") }
      let!(:article1) { create(:article, person: person1, title: "Article 1") }
      let!(:article2) { create(:article, person: person2, title: "Article 2") }
      let!(:article3) { create(:article, person: person1, title: "Article 3") }
      
      before { get "/people/articles" }
      
      it "returns success status" do
        expect(response).to have_http_status(:ok)
      end
      
      it "returns all articles" do
        expect(json_response.size).to eq(3)
      end
      
      it "includes article details" do
        titles = json_response.map { |a| a['title'] }
        expect(titles).to contain_exactly("Article 1", "Article 2", "Article 3")
      end
    end
  end

  describe "POST /people/:id/articles" do
    let(:person) { create(:person) }
    
    context "with valid params" do
      let(:article_params) { { article: { title: "Test Article", body: "Test Content" } } }
      
      it "creates a new article" do
        expect {
          post "/people/#{person.id}/articles", params: article_params
        }.to change(Article, :count).by(1)
      end
      
      it "associates article with person" do
        post "/people/#{person.id}/articles", params: article_params
        expect(person.reload.articles.count).to eq(1)
      end
      
      it "returns created status" do
        post "/people/#{person.id}/articles", params: article_params
        expect(response).to have_http_status(:created)
      end
      
      it "returns article data" do
        post "/people/#{person.id}/articles", params: article_params
        expect(json_response['title']).to eq("Test Article")
        expect(json_response['body']).to eq("Test Content")
      end
    end

    context "with invalid params" do
      let(:invalid_params) { { article: { title: "", body: "" } } }
      
      before { post "/people/#{person.id}/articles", params: invalid_params }
      
      it "does not create article" do
        expect(person.reload.articles.count).to eq(0)
      end
      
      it "returns unprocessable entity status" do
        expect(response).to have_http_status(:unprocessable_entity)
      end
      
      it "returns error messages" do
        expect(json_response['errors']).to be_present
      end
    end
    
    context "with multiple articles" do
      it "creates articles preserving association" do
        create(:article, person: person, title: "First Article")
        post "/people/#{person.id}/articles", params: { article: { title: "Second Article", body: "Content" } }
        
        expect(person.reload.articles.count).to eq(2)
        expect(person.articles.pluck(:title)).to include("First Article", "Second Article")
      end
    end
  end
end
