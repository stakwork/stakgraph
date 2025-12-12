FactoryBot.define do
  factory :country do
    name { "United States" }
    code { "US" }
    
    trait :european do
      name { "Germany" }
      code { "DE" }
    end
  end
end
