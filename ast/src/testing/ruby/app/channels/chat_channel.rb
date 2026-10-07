class ChatChannel < ApplicationCable::Channel
  # @ast node: Function "subscribed"
  def subscribed
    stream_from "chat_#{params[:room_id]}"
  end

  # @ast node: Function "unsubscribed"
  def unsubscribed
    stop_all_streams
  end

  # @ast node: Function "speak"
  def speak(data)
    ActionCable.server.broadcast("chat_#{params[:room_id]}", message: data['message'])
  end
end
