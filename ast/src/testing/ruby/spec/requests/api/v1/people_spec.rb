RSpec.describe "API V1 People", type: :request do
  let(:base_url) { "/api/v1/people" }
  
  describe "GET /api/v1/people" do
    context "when not authenticated" do
      it "returns unauthorized" do
        get base_url
        expect(response).to have_http_status(:unauthorized)
      end
    end
    
    context "when authenticated" do
      let(:user) { create(:person) }
      
      before do
        get base_url, headers: auth_headers(user)
      end
      
      it "returns success" do
        expect(response).to have_http_status(:ok)
      end
      
      context "with no people" do
        it "returns empty array" do
          expect(json_response['data']).to eq([])
        end
      end
      
      context "with multiple people" do
        let!(:person1) { create(:person, name: "Alice") }
        let!(:person2) { create(:person, name: "Bob") }
        
        it "returns all people" do
          get base_url, headers: auth_headers(user)
          expect(json_response['data'].size).to eq(3) # user + person1 + person2
        end
        
        it "includes person attributes" do
          get base_url, headers: auth_headers(user)
          names = json_response['data'].map { |p| p['name'] }
          expect(names).to include("Alice", "Bob")
        end
      end
    end
  end
  
  describe "GET /api/v1/people/:id" do
    let(:person) { create(:person, :with_articles) }
    let(:user) { create(:person) }
    
    context "when not authenticated" do
      it "returns unauthorized" do
        get "#{base_url}/#{person.id}"
        expect(response).to have_http_status(:unauthorized)
      end
    end
    
    context "when authenticated" do
      before do
        get "#{base_url}/#{person.id}", headers: auth_headers(user)
      end
      
      it "returns success" do
        expect(response).to have_http_status(:ok)
      end
      
      it "returns person data" do
        expect(json_response['id']).to eq(person.id)
        expect(json_response['name']).to eq(person.name)
      end
      
      it "includes associated articles" do
        expect(json_response['articles']).to be_present
        expect(json_response['articles'].size).to eq(2)
      end
    end
    
    context "when person does not exist" do
      it "returns not found" do
        get "#{base_url}/99999", headers: auth_headers(user)
        expect(response).to have_http_status(:not_found)
      end
    end
  end
  
  describe "POST /api/v1/people" do
    let(:user) { create(:person) }
    let(:valid_params) do
      { person: { name: "New Person", email: "new@example.com" } }
    end
    
    context "when not authenticated" do
      it "returns unauthorized" do
        post base_url, params: valid_params
        expect(response).to have_http_status(:unauthorized)
      end
    end
    
    context "when authenticated with valid params" do
      it "creates new person" do
        expect {
          post base_url, params: valid_params, headers: auth_headers(user)
        }.to change(Person, :count).by(1)
      end
      
      it "returns created status" do
        post base_url, params: valid_params, headers: auth_headers(user)
        expect(response).to have_http_status(:created)
      end
      
      it "returns person data" do
        post base_url, params: valid_params, headers: auth_headers(user)
        expect(json_response['name']).to eq("New Person")
        expect(json_response['email']).to eq("new@example.com")
      end
    end
    
    context "when authenticated with invalid params" do
      let(:invalid_params) { { person: { name: "", email: "" } } }
      
      before do
        post base_url, params: invalid_params, headers: auth_headers(user)
      end
      
      it "does not create person" do
        expect(Person.count).to eq(1) # only the authenticated user
      end
      
      it "returns unprocessable entity" do
        expect(response).to have_http_status(:unprocessable_entity)
      end
      
      it "returns validation errors" do
        expect(json_response['errors']).to include('name')
        expect(json_response['errors']).to include('email')
      end
    end
  end
  
  describe "PATCH /api/v1/people/:id" do
    let(:person) { create(:person, name: "Original Name") }
    let(:user) { create(:person) }
    let(:update_params) { { person: { name: "Updated Name" } } }
    
    context "when not authenticated" do
      it "returns unauthorized" do
        patch "#{base_url}/#{person.id}", params: update_params
        expect(response).to have_http_status(:unauthorized)
      end
    end
    
    context "when authenticated" do
      it "updates the person" do
        patch "#{base_url}/#{person.id}", params: update_params, headers: auth_headers(user)
        expect(person.reload.name).to eq("Updated Name")
      end
      
      it "returns success" do
        patch "#{base_url}/#{person.id}", params: update_params, headers: auth_headers(user)
        expect(response).to have_http_status(:ok)
      end
      
      it "returns updated data" do
        patch "#{base_url}/#{person.id}", params: update_params, headers: auth_headers(user)
        expect(json_response['name']).to eq("Updated Name")
      end
    end
  end
  
  describe "DELETE /api/v1/people/:id" do
    let!(:person) { create(:person) }
    let(:user) { create(:person) }
    
    context "when not authenticated" do
      it "returns unauthorized" do
        delete "#{base_url}/#{person.id}"
        expect(response).to have_http_status(:unauthorized)
      end
    end
    
    context "when authenticated" do
      it "deletes the person" do
        expect {
          delete "#{base_url}/#{person.id}", headers: auth_headers(user)
        }.to change(Person, :count).by(-1)
      end
      
      it "returns no content status" do
        delete "#{base_url}/#{person.id}", headers: auth_headers(user)
        expect(response).to have_http_status(:no_content)
      end
    end
  end
end
