class PersonPolicy
  attr_reader :user, :person

  def initialize(user, person)
    @user = user
    @person = person
  end

  def show?
    true
  end

  def create?
    user.admin?
  end

  def update?
    user.admin? || user == person
  end

  def destroy?
    user.admin?
  end
end
