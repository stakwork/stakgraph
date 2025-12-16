require 'test_helper'

class UserMailerTest < ActionMailer::TestCase
  test "welcome_email sends to new user" do
    person = Person.create!(name: "Jane Smith", email: "jane@example.com")
    email = UserMailer.welcome_email(person)

    assert_emails 1 do
      email.deliver_now
    end

    assert_equal ["jane@example.com"], email.to
    assert_equal "Welcome to Our App", email.subject
    assert_match "Welcome Jane Smith", email.body.encoded
  end

  test "welcome_email includes signup confirmation" do
    person = Person.create!(name: "Bob", email: "bob@example.com")
    email = UserMailer.welcome_email(person)

    assert_match "Thanks for signing up", email.body.encoded
    assert_match person.name, email.body.encoded
  end

  test "password_reset includes reset token" do
    person = Person.create!(name: "Alice", email: "alice@example.com")
    token = "abc123reset"
    email = UserMailer.password_reset(person, token)

    assert_equal "Password Reset Instructions", email.subject
    assert_equal [person.email], email.to
    assert_match token, email.body.encoded
    assert_match "reset", email.body.encoded.downcase
  end

  test "password_reset expires in 2 hours" do
    person = Person.create!(name: "Charlie", email: "charlie@example.com")
    email = UserMailer.password_reset(person, "token456")

    assert_match "2 hours", email.body.encoded
  end

  test "notification_email contains article title" do
    person = Person.create!(name: "Diana", email: "diana@example.com")
    article = Article.create!(title: "Important Update", body: "News content", person: person)
    email = UserMailer.notification_email(person, article)

    assert_includes email.subject, "Important Update"
    assert_match article.title, email.body.encoded
    assert_equal [person.email], email.to
  end
end
