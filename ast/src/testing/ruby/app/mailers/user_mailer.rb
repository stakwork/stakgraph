class UserMailer < ApplicationMailer
  default from: 'notifications@example.com'

  # @ast node: Function "welcome_email"
  def welcome_email(user)
    @user = user
    @url  = 'http://example.com/login'
    mail(to: @user.email, subject: 'Welcome to My Awesome Site')
  end

  # @ast node: Function "password_reset"
  def password_reset(user)
    @user = user
    @token = user.reset_token
    mail(to: @user.email, subject: 'Password Reset Instructions')
  end
end
