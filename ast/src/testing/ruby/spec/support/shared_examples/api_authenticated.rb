RSpec.shared_examples "an authenticated API endpoint" do |method, path|
  context "without authentication" do
    it "returns 401 unauthorized" do
      send(method, path)
      expect(response).to have_http_status(:unauthorized)
    end
    
    it "returns authentication error message" do
      send(method, path)
      expect(json_response['error']).to match(/unauthorized|authentication/i)
    end
  end
end

RSpec.shared_examples "requires authentication" do
  it "requires valid token" do
    expect(response).to have_http_status(:unauthorized)
  end
end
