require_relative '../app/controllers/user_controller'
require 'minitest/autorun'

class TestUserController < Minitest::Test
  def setup
    DB.clear!
  end

  def test_create_success
    response = UserController.create({ username: 'test', email: 'test@example.com' })
    status, _, _ = response
    assert_equal 201, status
  end

  def test_create_fail
    response = UserController.create({ username: '', email: '' })
    status, _, _ = response
    assert_equal 422, status
  end
end
