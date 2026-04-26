# @ast node: E2eTest "Test id lookup"
RSpec.describe "Test id lookup", type: :feature do
  it "finds by test id" do
    visit "/items"
    page.get_by_test_id('row')
    expect(page).to have_content("Items")
  end
end
