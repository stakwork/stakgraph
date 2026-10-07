module JsonHelpers
  # @ast node: Function "json_response"
  def json_response
    JSON.parse(response.body)
  rescue JSON::ParserError
    {}
  end

  # @ast node: Function "json_data"
  def json_data
    json_response['data']
  end

  # @ast node: Function "json_errors"
  def json_errors
    json_response['errors']
  end
end

RSpec.configure do |config|
  config.include JsonHelpers, type: :request
end
