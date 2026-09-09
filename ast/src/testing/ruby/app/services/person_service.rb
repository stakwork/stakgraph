class PersonService
  # @ast node: Function "get_person_by_id"
  def self.get_person_by_id(id)
    Person.find_by(id: id)
  end

  # @ast node: Function "new_person"
  def self.new_person(person_params)
    Person.create(person_params)
  end

  # @ast node: Function "delete"
  def self.delete(id)
    Person.destroy(id)
  end
end