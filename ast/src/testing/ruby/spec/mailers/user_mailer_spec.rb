RSpec.describe UserMailer, type: :mailer do
  describe "#welcome_email" do
    let(:person) { create(:person, email: "newuser@example.com", name: "John Doe") }
    let(:mail) { UserMailer.welcome_email(person) }

    it "renders the headers" do
      expect(mail.subject).to eq("Welcome to Our App")
      expect(mail.to).to eq(["newuser@example.com"])
      expect(mail.from).to eq(["noreply@example.com"])
    end

    it "renders the body" do
      expect(mail.body.encoded).to match("Welcome John Doe")
      expect(mail.body.encoded).to match("Thanks for signing up")
    end

    it "includes the person's name" do
      expect(mail.body.encoded).to include(person.name)
    end
  end

  describe "#password_reset" do
    let(:person) { create(:person, email: "user@example.com") }
    let(:token) { "reset_token_123" }
    let(:mail) { UserMailer.password_reset(person, token) }

    it "renders the subject" do
      expect(mail.subject).to eq("Password Reset Instructions")
    end

    it "sends to the person's email" do
      expect(mail.to).to eq([person.email])
    end

    it "includes reset link with token" do
      expect(mail.body.encoded).to include("reset_token_123")
      expect(mail.body.encoded).to match(/reset.*password/i)
    end

    it "expires in 2 hours" do
      expect(mail.body.encoded).to include("2 hours")
    end
  end

  describe "#notification_email" do
    let(:person) { create(:person, email: "recipient@example.com") }
    let(:article) { create(:article, title: "Breaking News") }
    let(:mail) { UserMailer.notification_email(person, article) }

    it "sends notification to person" do
      expect(mail.to).to eq([person.email])
    end

    it "includes article title in subject" do
      expect(mail.subject).to include("Breaking News")
    end

    it "contains article information in body" do
      expect(mail.body.encoded).to include(article.title)
    end
  end
end
