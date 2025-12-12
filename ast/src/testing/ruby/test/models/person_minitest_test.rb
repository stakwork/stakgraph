require 'test_helper'

class PersonMinitestTest < Minitest::Test
  def setup
    @person = Person.new(name: "Alice", email: "alice@example.com")
  end

  def test_valid_person
    assert @person.valid?
  end

  def test_requires_name
    @person.name = nil
    refute @person.valid?
    assert_includes @person.errors[:name], "can't be blank"
  end

  def test_requires_email
    @person.email = nil
    refute @person.valid?
  end

  def test_email_uniqueness
    Person.create!(name: "Bob", email: "alice@example.com")
    duplicate = Person.new(name: "Alice Duplicate", email: "alice@example.com")
    refute duplicate.valid?
  end
end
