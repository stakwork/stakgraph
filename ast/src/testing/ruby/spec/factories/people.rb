FactoryBot.define do
  factory :person do
    name { "John Doe" }
    email { "john.doe@example.com" }
    
    trait :with_articles do
      after(:create) do |person|
        create_list(:article, 2, person: person)
      end
    end
    
    trait :inactive do
      active { false }
    end
  end
end
