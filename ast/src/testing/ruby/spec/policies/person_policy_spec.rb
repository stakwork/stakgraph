RSpec.describe PersonPolicy, type: :policy do
  subject { described_class }

  let(:admin) { create(:person, role: :admin) }
  let(:user) { create(:person, role: :user) }
  let(:other_user) { create(:person, role: :user) }

  permissions :index? do
    it "allows admins to list people" do
      expect(subject).to permit(admin, Person)
    end

    it "allows regular users to list people" do
      expect(subject).to permit(user, Person)
    end
  end

  permissions :show? do
    it "allows users to view their own profile" do
      expect(subject).to permit(user, user)
    end

    it "allows users to view other profiles" do
      expect(subject).to permit(user, other_user)
    end

    it "allows admins to view any profile" do
      expect(subject).to permit(admin, other_user)
    end
  end

  permissions :create? do
    it "allows admins to create people" do
      expect(subject).to permit(admin, Person)
    end

    it "denies regular users from creating people" do
      expect(subject).not_to permit(user, Person)
    end
  end

  permissions :update? do
    it "allows users to update their own profile" do
      expect(subject).to permit(user, user)
    end

    it "denies users from updating other profiles" do
      expect(subject).not_to permit(user, other_user)
    end

    it "allows admins to update any profile" do
      expect(subject).to permit(admin, other_user)
    end
  end

  permissions :destroy? do
    it "allows users to delete their own account" do
      expect(subject).to permit(user, user)
    end

    it "denies users from deleting other accounts" do
      expect(subject).not_to permit(user, other_user)
    end

    it "allows admins to delete any account" do
      expect(subject).to permit(admin, other_user)
    end

    it "denies admins from deleting themselves" do
      expect(subject).not_to permit(admin, admin)
    end
  end

  describe "scope" do
    it "returns all people for admins" do
      policy_scope = Pundit.policy_scope!(admin, Person)
      expect(policy_scope).to eq(Person.all)
    end

    it "returns only active people for regular users" do
      policy_scope = Pundit.policy_scope!(user, Person)
      expect(policy_scope).to eq(Person.where(active: true))
    end
  end
end
