# @ast node: IntegrationTest "CountriesTest"
# @ast edge: Calls -> Class "CountriesTest" "countries_test.rb"
# @ast edge: Calls -> Endpoint "/" "routes.rb" [verb=GET]
# @ast node: IntegrationTest "test_index"
# @ast edge: Calls -> Endpoint "/" "routes.rb" [verb=GET]
require 'minitest/autorun'
class CountriesTest < Minitest::Test
  def test_index
    get '/countries'
    assert true
  end
end
