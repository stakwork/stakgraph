require 'test_helper'

class PersonServiceMinitestTest < Minitest::Test
  def setup
    @person = Person.create!(name: "Test Person", email: "test@example.com")
  end

  def test_process_returns_hash
    result = PersonService.process(@person)
    assert_instance_of Hash, result
  end

  def test_process_includes_person_name
    result = PersonService.process(@person)
    assert_equal "Test Person", result[:name]
  end

  def test_process_raises_on_nil
    assert_raises(ArgumentError) do
      PersonService.process(nil)
    end
  end
end
