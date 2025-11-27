class ProfilesController < ApplicationController
  def show
    profile = fetch_user_profile
    render json: profile, status: :ok
  end

  def edit
    profile = fetch_user_profile
    render json: profile, status: :ok
  end

  def update
    updated_profile = update_user_profile(params)
    render json: updated_profile, status: :ok
  end

  private

  def fetch_user_profile
    {}
  end

  def update_user_profile(params)
    {}
  end
end
