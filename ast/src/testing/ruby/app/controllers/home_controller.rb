class HomeController < ApplicationController
  # @ast node: Function "index"
  def index
    render json: { message: 'Welcome to the homepage' }, status: :ok
  end
end
