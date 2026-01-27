module Validators
  EMAIL_REGEX = /\A[\w+\-.]+@[a-z\d\-]+(\.[a-z\d\-]+)*\.[a-z]+\z/i

  def self.valid_email?(email)
    return false unless email
    !!(email =~ EMAIL_REGEX)
  end

  def self.present?(value)
    !value.nil? && !value.to_s.strip.empty?
  end
end
