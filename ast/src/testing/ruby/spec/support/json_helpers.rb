module JsonHelpers
  def json_response
    JSON.parse(response.body)
  rescue JSON::ParserError
    {}
  end
  
  def json_data
    json_response['data']
  end
  
  def json_errors
    json_response['errors']
  end
end

RSpec.configure do |config|
  config.include JsonHelpers, type: :request
end
