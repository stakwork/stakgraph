FactoryBot.define do
  factory :article do
    title { "Sample Article Title" }
    body { "This is the article content with detailed information." }
    association :person
    
    trait :published do
      published { true }
    end
    
    trait :draft do
      published { false }
    end
  end
end
