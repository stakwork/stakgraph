# @ast node: E2eTest "Mixed classification"
RSpec.describe "Mixed classification", type: :feature do
  it "does network and ui" do
    visit "/profile"
    get "/profile"
    expect(page).to have_content("Profile")
  end
end
