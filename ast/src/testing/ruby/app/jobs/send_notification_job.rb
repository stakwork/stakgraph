class SendNotificationJob < ApplicationJob
  queue_as :urgent

  def perform(user_id, message)
    user = User.find(user_id)
    NotificationService.send(user, message)
  end
end
