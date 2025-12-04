class HomeController < ApplicationController
  def index
    render json: { message: 'Welcome to the homepage' }, status: :ok
  end
end
