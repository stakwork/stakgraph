require 'securerandom'
require 'json'

module Utils
  def self.generate_id
    SecureRandom.uuid
  end

  def self.json_response(body, status = 200)
    [
      status,
      { 'Content-Type' => 'application/json' },
      [body.to_json]
    ]
  end

  def self.symbolize_keys(hash)
    hash.transform_keys(&:to_sym)
  end
end
