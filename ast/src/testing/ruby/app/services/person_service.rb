class PersonService
  def self.get_person_by_id(id)
    Person.find_by(id: id)
  end

  def self.new_person(person_params)
    Person.create(person_params)
  end
end