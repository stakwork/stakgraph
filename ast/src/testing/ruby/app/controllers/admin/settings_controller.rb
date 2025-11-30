module Admin
  class SettingsController < ApplicationController
    def index
      settings = fetch_all_settings
      render json: settings, status: :ok
    end

    def update
      setting = update_setting(params[:id])
      render json: setting, status: :ok
    end

    private

    def fetch_all_settings
      []
    end

    def update_setting(id)
      {}
    end
  end
end
