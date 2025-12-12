RSpec.describe "People API", type: :request do
  describe "GET /person/:id" do
    let(:person) { create(:person) }
    
    context "when person exists" do
      before { get "/person/#{person.id}" }
      
      it "returns success status" do
        expect(response).to have_http_status(:ok)
      end
      
      it "returns person data" do
        expect(json_response['name']).to eq(person.name)
        expect(json_response['email']).to eq(person.email)
      end
    end
    
    context "when person not found" do
      before { get "/person/9999" }
      
      it "returns not found status" do
        expect(response).to have_http_status(:not_found)
      end
      
      it "returns error message" do
        expect(json_response['error']).to eq("Person not found")
      end
    end
  end

  describe "POST /person" do
    context "with valid params" do
      let(:valid_params) { { person: { name: "Jane Doe", email: "jane@example.com" } } }
      
      it "creates a new person" do
        expect {
          post "/person", params: valid_params
        }.to change(Person, :count).by(1)
      end
      
      it "returns created status" do
        post "/person", params: valid_params
        expect(response).to have_http_status(:created)
      end
      
      it "returns person data" do
        post "/person", params: valid_params
        expect(json_response['name']).to eq("Jane Doe")
        expect(json_response['email']).to eq("jane@example.com")
      end
    end
    
    context "with invalid params" do
      let(:invalid_params) { { person: { name: "", email: "" } } }
      
      it "does not create person" do
        expect {
          post "/person", params: invalid_params
        }.not_to change(Person, :count)
      end
      
      it "returns unprocessable entity status" do
        post "/person", params: invalid_params
        expect(response).to have_http_status(:unprocessable_entity)
      end
      
      it "returns error messages" do
        post "/person", params: invalid_params
        expect(json_response['errors']).to be_present
      end
    end
  end

  describe "DELETE /people/:id" do
    let!(:person) { create(:person) }
    
    context "when person exists" do
      it "deletes the person" do
        expect {
          delete "/people/#{person.id}"
        }.to change(Person, :count).by(-1)
      end
      
      it "returns success status" do
        delete "/people/#{person.id}"
        expect(response).to have_http_status(:ok)
      end
      
      it "returns success message" do
        delete "/people/#{person.id}"
        expect(json_response['message']).to eq('Person deleted successfully')
      end
    end
    
    context "when person not found" do
      before { delete "/people/9999" }
      
      it "returns not found status" do
        expect(response).to have_http_status(:not_found)
      end
      
      it "returns error message" do
        expect(json_response['error']).to eq("Person not found")
      end
    end
  end

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
    
    context "when articles exist" do
      let(:person) { create(:person) }
      let!(:articles) { create_list(:article, 3, person: person) }
      
      before { get "/people/articles" }
      
      it "returns all articles" do
        expect(json_response.length).to eq(3)
      end
    end
  end

  describe "POST /people/:id/articles" do
    let(:person) { create(:person) }
    
    context "with valid params" do
      let(:valid_params) { { article: { title: "New Article", body: "New Content" } } }
      
      it "creates article for person" do
        expect {
          post "/people/#{person.id}/articles", params: valid_params
        }.to change(person.articles, :count).by(1)
      end
      
      it "returns created status" do
        post "/people/#{person.id}/articles", params: valid_params
        expect(response).to have_http_status(:created)
      end
      
      it "returns article data" do
        post "/people/#{person.id}/articles", params: valid_params
        expect(json_response['title']).to eq("New Article")
        expect(json_response['body']).to eq("New Content")
      end
    end
    
    context "with invalid params" do
      let(:invalid_params) { { article: { title: "", body: "" } } }
      
      it "returns unprocessable entity status" do
        post "/people/#{person.id}/articles", params: invalid_params
        expect(response).to have_http_status(:unprocessable_entity)
      end
    end
  end
end
