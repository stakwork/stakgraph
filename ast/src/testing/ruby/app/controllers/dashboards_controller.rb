class DashboardsController < ApplicationController
  def show
    dashboard_data = fetch_dashboard_data
    render json: dashboard_data, status: :ok
  end

  def update
    updated_dashboard = update_dashboard_settings(params)
    render json: updated_dashboard, status: :ok
  end

  private

  def fetch_dashboard_data
    {}
  end

  def update_dashboard_settings(params)
    {}
  end
end
