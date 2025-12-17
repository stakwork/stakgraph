class PersonSerializer < ActiveModel::Serializer
  attributes :id, :name, :email, :age

  def email
    object.email.downcase
  end
end
