class UserMailer < ApplicationMailer
  default from: 'notifications@example.com'

  def welcome_email(user)
    @user = user
    @url  = 'http://example.com/login'
    mail(to: @user.email, subject: 'Welcome to My Awesome Site')
  end

  def password_reset(user)
    @user = user
    @token = user.reset_token
    mail(to: @user.email, subject: 'Password Reset Instructions')
  end
end
