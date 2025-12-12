require 'rails_helper'

describe Country do
  let(:country) { build(:country) }

  it "validates presence of name" do
    country.name = nil
    expect(country).not_to be_valid
  end

  it "validates uniqueness of iso_code" do
    create(:country, iso_code: "US")
    duplicate = build(:country, iso_code: "US")
    expect(duplicate).not_to be_valid
  end
end
