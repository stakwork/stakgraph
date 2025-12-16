class ChatChannel < ApplicationCable::Channel
  def subscribed
    stream_from "chat_#{params[:room_id]}"
  end

  def unsubscribed
    stop_all_streams
  end

  def speak(data)
    ActionCable.server.broadcast("chat_#{params[:room_id]}", message: data['message'])
  end
end
