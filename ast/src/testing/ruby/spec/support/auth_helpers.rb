module AuthHelpers
  # @ast node: Function "sign_in"
  def sign_in(user)
    @current_user = user
    { "Authorization" => "Bearer #{generate_token(user)}" }
  end

  # @ast node: Function "auth_headers"
  def auth_headers(user)
    { "Authorization" => "Bearer #{generate_token(user)}" }
  end

  private

  # @ast node: Function "generate_token"
  def generate_token(user)
    "token_#{user.id}_#{user.email}"
  end
end

RSpec.configure do |config|
  config.include AuthHelpers, type: :request
  config.include AuthHelpers, type: :system
end
