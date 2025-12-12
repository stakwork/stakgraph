RSpec.describe "Authenticated API Endpoints", type: :request do
  let(:user) { create(:person) }
  
  describe "GET /api/v1/dashboard" do
    let(:endpoint) { "/api/v1/dashboard" }
    
    it_behaves_like "an authenticated API endpoint", :get, "/api/v1/dashboard"
    
    context "when authenticated" do
      before { get endpoint, headers: auth_headers(user) }
      
      it "returns success" do
        expect(response).to have_http_status(:ok)
      end
      
      it "returns dashboard data" do
        expect(json_response['user_id']).to eq(user.id)
        expect(json_response['data']).to be_present
      end
    end
  end
  
  describe "GET /api/v1/profile" do
    let(:endpoint) { "/api/v1/profile" }
    
    it_behaves_like "an authenticated API endpoint", :get, "/api/v1/profile"
    
    context "when authenticated" do
      let!(:person_with_articles) { create(:person, :with_articles) }
      
      before do
        get endpoint, headers: auth_headers(person_with_articles)
      end
      
      it "returns success" do
        expect(response).to have_http_status(:ok)
      end
      
      it "returns current user profile" do
        expect(json_response['id']).to eq(person_with_articles.id)
        expect(json_response['name']).to eq(person_with_articles.name)
      end
      
      it "includes user articles" do
        expect(json_response['articles']).to be_present
        expect(json_response['articles'].size).to eq(2)
      end
    end
  end
  
  describe "POST /api/v1/settings" do
    let(:endpoint) { "/api/v1/settings" }
    let(:params) { { settings: { theme: "dark", notifications: true } } }
    
    it_behaves_like "an authenticated API endpoint", :post, "/api/v1/settings"
    
    context "when authenticated" do
      before do
        post endpoint, params: params, headers: auth_headers(user)
      end
      
      it "returns success" do
        expect(response).to have_http_status(:ok)
      end
      
      it "saves settings" do
        expect(json_response['theme']).to eq("dark")
        expect(json_response['notifications']).to eq(true)
      end
    end
  end
  
  describe "PUT /api/v1/profile" do
    let(:endpoint) { "/api/v1/profile" }
    let(:update_params) { { person: { name: "Updated Name" } } }
    
    it_behaves_like "an authenticated API endpoint", :put, "/api/v1/profile"
    
    context "when authenticated" do
      it "updates user profile" do
        put endpoint, params: update_params, headers: auth_headers(user)
        expect(user.reload.name).to eq("Updated Name")
      end
      
      it "returns updated data" do
        put endpoint, params: update_params, headers: auth_headers(user)
        expect(json_response['name']).to eq("Updated Name")
      end
    end
  end
  
  describe "DELETE /api/v1/account" do
    let(:endpoint) { "/api/v1/account" }
    
    it_behaves_like "an authenticated API endpoint", :delete, "/api/v1/account"
    
    context "when authenticated" do
      let!(:deletable_user) { create(:person) }
      
      it "deletes user account" do
        expect {
          delete endpoint, headers: auth_headers(deletable_user)
        }.to change(Person, :count).by(-1)
      end
      
      it "returns no content" do
        delete endpoint, headers: auth_headers(deletable_user)
        expect(response).to have_http_status(:no_content)
      end
    end
  end
  
  describe "nested resources with authentication" do
    let(:person) { create(:person) }
    
    describe "GET /api/v1/people/:id/articles" do
      let(:endpoint) { "/api/v1/people/#{person.id}/articles" }
      
      it_behaves_like "an authenticated API endpoint", :get, :dynamic
      
      context "when authenticated" do
        let!(:articles) { create_list(:article, 3, person: person) }
        
        before do
          get endpoint, headers: auth_headers(user)
        end
        
        it "returns success" do
          expect(response).to have_http_status(:ok)
        end
        
        it "returns person's articles" do
          expect(json_response.size).to eq(3)
          json_response.each do |article|
            expect(article['person_id']).to eq(person.id)
          end
        end
      end
    end
  end
end
