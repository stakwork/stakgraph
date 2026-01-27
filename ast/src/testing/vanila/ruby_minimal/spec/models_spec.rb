require_relative '../lib/models'
require 'minitest/autorun'

class TestUser < Minitest::Test
  def setup
    DB.clear!
  end

  def test_valid_user
    user = Models::User.new(username: 'test', email: 'test@example.com')
    assert user.valid?
    assert user.save
  end

  def test_invalid_email
    user = Models::User.new(username: 'test', email: 'bad-email')
    refute user.valid?
    refute user.save
  end
end
