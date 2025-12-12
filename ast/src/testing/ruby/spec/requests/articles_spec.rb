# Integration (request)
RSpec.describe "Articles API", type: :request do
  describe "GET /articles" do
    context "with no articles" do
      before { get "/articles" }
      
      it "returns success" do
        expect(response).to have_http_status(:ok)
      end
      
      it "returns empty array" do
        expect(json_response).to eq([])
      end
    end
    
    context "with published articles" do
      let!(:person) { create(:person) }
      let!(:published1) { create(:article, :published, person: person, title: "Published 1") }
      let!(:published2) { create(:article, :published, person: person, title: "Published 2") }
      let!(:draft) { create(:article, person: person, title: "Draft", published: false) }
      
      before { get "/articles" }
      
      it "returns success" do
        expect(response).to have_http_status(:ok)
      end
      
      it "returns only published articles" do
        expect(json_response.size).to eq(2)
        titles = json_response.map { |a| a['title'] }
        expect(titles).to contain_exactly("Published 1", "Published 2")
      end
    end
  end
  
  describe "GET /articles/:id" do
    let(:person) { create(:person) }
    let(:article) { create(:article, person: person, title: "Test Article") }
    
    context "when article exists" do
      before { get "/articles/#{article.id}" }
      
      it "returns success" do
        expect(response).to have_http_status(:ok)
      end
      
      it "returns article data" do
        expect(json_response['id']).to eq(article.id)
        expect(json_response['title']).to eq("Test Article")
      end
    end
    
    context "when article does not exist" do
      before { get "/articles/99999" }
      
      it "returns not found" do
        expect(response).to have_http_status(:not_found)
      end
    end
  end
  
  describe "POST /articles" do
    let(:person) { create(:person) }
    
    context "with valid params" do
      let(:params) { { article: { title: "New Article", body: "Content", person_id: person.id } } }
      
      it "creates article" do
        expect {
          post "/articles", params: params
        }.to change(Article, :count).by(1)
      end
      
      it "returns created status" do
        post "/articles", params: params
        expect(response).to have_http_status(:created)
      end
      
      it "returns article data" do
        post "/articles", params: params
        expect(json_response['title']).to eq("New Article")
      end
    end
    
    context "with invalid params" do
      let(:params) { { article: { title: "", body: "" } } }
      
      before { post "/articles", params: params }
      
      it "returns unprocessable entity" do
        expect(response).to have_http_status(:unprocessable_entity)
      end
      
      it "returns errors" do
        expect(json_response['errors']).to be_present
      end
    end
  end
  
  describe "DELETE /articles/:id" do
    let!(:article) { create(:article, :published) }
    
    it "deletes the article" do
      expect {
        delete "/articles/#{article.id}"
      }.to change(Article, :count).by(-1)
    end
    
    it "returns no content status" do
      delete "/articles/#{article.id}"
      expect(response).to have_http_status(:no_content)
    end
  end
end
