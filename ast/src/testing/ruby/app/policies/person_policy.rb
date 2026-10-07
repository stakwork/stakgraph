class PersonPolicy
  attr_reader :user, :person

  # @ast node: Function "initialize"
  def initialize(user, person)
    @user = user
    @person = person
  end

  # @ast node: Function "show?"
  def show?
    true
  end

  # @ast node: Function "create?"
  def create?
    user.admin?
  end

  # @ast node: Function "update?"
  def update?
    user.admin? || user == person
  end

  # @ast node: Function "destroy?"
  def destroy?
    user.admin?
  end
end
