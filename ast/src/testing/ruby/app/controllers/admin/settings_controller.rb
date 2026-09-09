module Admin
  class SettingsController < ApplicationController
    # @ast node: Function "index"
    def index
      settings = fetch_all_settings
      render json: settings, status: :ok
    end

    # @ast node: Function "update"
    def update
      setting = update_setting(params[:id])
      render json: setting, status: :ok
    end

    private

    # @ast node: Function "fetch_all_settings"
    def fetch_all_settings
      []
    end

    # @ast node: Function "update_setting"
    def update_setting(id)
      {}
    end
  end
end
