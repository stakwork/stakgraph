require 'test_helper'

class ArticleMinitestDslTest < Minitest::Test
  test "creates valid article with title and body" do
    article = Article.new(title: "Test Article", body: "Content")
    assert article.valid?
  end

  test "requires title" do
    article = Article.new(body: "Content only")
    refute article.valid?
    assert_includes article.errors[:title], "can't be blank"
  end

  test "belongs to person" do
    person = Person.create!(name: "Author", email: "author@example.com")
    article = Article.new(title: "Article", body: "Body", person: person)
    assert_equal person, article.person
  end
end
