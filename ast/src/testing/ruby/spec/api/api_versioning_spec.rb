# API integration
# @ast node: IntegrationTest "API versioning"
# @ast edge: Calls -> Endpoint "/" "routes.rb" [verb=GET]
RSpec.describe "API versioning", type: :request do
  it "pings v1" do
    get "/api/v1/ping"
    expect(response.status).to eq(200)
  end
end
