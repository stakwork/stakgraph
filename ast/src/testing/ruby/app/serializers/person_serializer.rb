class PersonSerializer < ActiveModel::Serializer
  attributes :id, :name, :email, :age

  # @ast node: Function "email"
  def email
    object.email.downcase
  end
end
